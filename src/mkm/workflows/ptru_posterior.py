"""PtRu posterior provenance, checkpoint validation, and loading."""

from dataclasses import dataclass
import numpy as np

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.posterior import (
    compute_posterior_deterministics,
    load_inference_data,
)
from mkm.models.ptru_registry import ptru_model_metadata
from mkm.provenance import (
    build_fit_metadata,
    read_run_metadata,
    sha256_file,
    sha256_mapping,
)
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE, RUN_STATUS_SAMPLED
from mkm.workflows.pt100_fit import build_pt100_fit


PT100_SOURCE_FILES = (
    "src/mkm/mechanisms/pt100.py",
    "src/mkm/mechanisms/co_ssa.py",
    "src/mkm/mechanisms/agpd_basic/electrochem.py",
    "src/mkm/models/pt100.py",
    "src/mkm/models/ptru_registry.py",
    "src/mkm/workflows/pt100_fit.py",
    "src/mkm/workflows/configuration.py",
    "src/mkm/workflows/likelihood_configuration.py",
    "src/mkm/model_data.py",
    "src/mkm/model_inputs.py",
    "src/mkm/inference/model.py",
    "src/mkm/inference/likelihoods.py",
    "src/mkm/inference/priors.py",
)


@dataclass(frozen=True)
class Pt100PosteriorRun:
    specification: object
    metadata: dict
    output_dir: object
    model_data: object
    inputs: object
    built_model: object
    inference_data: object
    parameter_specs: dict
    free_parameter_names: tuple[str, ...]


def pt100_model_specification(paths, fit):
    return {
        "family": "Pt100",
        "medium": fit.specification.medium,
        "model_name": fit.specification.model_name,
        "assumptions": fit.config["mechanism_assumptions"],
        "registry": ptru_model_metadata(fit.specification.model_name),
        "source_sha256": {
            filename: sha256_file(paths.root / filename)
            for filename in PT100_SOURCE_FILES
        },
    }


def build_pt100_run_metadata(paths, fit, sampler):
    return build_fit_metadata(
        root=paths.root,
        fit_scope=fit.specification.fit_scope,
        materials=fit.inputs.materials,
        model_name=fit.specification.model_name,
        error_structure=fit.specification.error_structure,
        data_path=fit.data_path,
        model_config_path=fit.model_config_path,
        model_config_paths=(fit.model_config_path, fit.prior_config_path),
        sampler=sampler,
        parameterization=fit.specification.parameterization,
        model_specification=pt100_model_specification(paths, fit),
        prior_specification=fit.parameter_specs,
    )


def validate_pt100_run_metadata(
    metadata, paths, fit, *, allowed_statuses=(RUN_STATUS_COMPLETE,),
):
    model_specification = pt100_model_specification(paths, fit)
    prior_specification = fit.parameter_specs
    spec = fit.specification

    expected = {
        "fit_scope": spec.fit_scope,
        "materials": list(fit.inputs.materials),
        "model": spec.model_name,
        "likelihood": RATE_NORMAL,
        "error_structure": spec.error_structure,
        "parameterization": spec.parameterization,
        "model_specification": model_specification,
        "prior_specification": prior_specification,
        "resolved_fit_sha256": sha256_mapping({
            "model_specification": model_specification,
            "parameterization_specification": None,
            "prior_specification": prior_specification,
            "error_structure": spec.error_structure,
            "likelihood": RATE_NORMAL,
        }),
    }

    mismatches = [
        key for key, value in expected.items()
        if metadata.get(key) != value
    ]

    input_metadata = metadata.get("inputs", {})
    config_paths = (fit.model_config_path, fit.prior_config_path)
    expected_inputs = {
        "data_path": str(fit.data_path.relative_to(paths.root)),
        "data_sha256": sha256_file(fit.data_path),
        "model_config_path": str(fit.model_config_path.relative_to(paths.root)),
        "model_config_paths": [
            str(path.relative_to(paths.root)) for path in config_paths
        ],
        "model_config_sha256": sha256_mapping({
            str(path.relative_to(paths.root)): sha256_file(path)
            for path in config_paths
        }),
    }

    mismatches.extend(
        f"inputs.{key}"
        for key, value in expected_inputs.items()
        if input_metadata.get(key) != value
    )

    if metadata.get("status") not in allowed_statuses:
        mismatches.append("status")

    if mismatches:
        raise ValueError(
            "Posterior checkpoint does not match the current Pt100 fit: "
            + ", ".join(mismatches)
        )



def load_pt100_posterior_run(
    paths,
    medium,
    model_name,
    *,
    reconstruct_model_rate=False,
    reconstruct_pointwise=False,
    draws_per_chain=None,
    progressbar=True,
):
    fit = build_pt100_fit(paths, medium, model_name)
    posterior_path = fit.output_dir / "posterior.nc"
    metadata_path = fit.output_dir / "run_metadata.yaml"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    metadata = read_run_metadata(metadata_path)
    validate_pt100_run_metadata(metadata, paths, fit)

    inference_data = load_inference_data(posterior_path)
    built = fit.built_model
    free_parameter_names = tuple(variable.name for variable in built.model.free_RVs)

    missing_free = [
        name for name in free_parameter_names
        if name not in inference_data.posterior
    ]
    if missing_free:
        raise ValueError(f"Posterior is missing fitted variables: {missing_free}")

    if draws_per_chain is not None:
        draws_per_chain = int(draws_per_chain)
        if draws_per_chain < 1:
            raise ValueError("draws_per_chain must be positive.")

        n_draws = int(inference_data.posterior.sizes["draw"])
        stride = max(1, int(np.ceil(n_draws / draws_per_chain)))

        inference_data = inference_data.copy()

        for group_name in ("posterior", "sample_stats", "log_likelihood"):
            if not hasattr(inference_data, group_name):
                continue

            group = getattr(inference_data, group_name)
            if hasattr(group, "to_dataset"):
                group = group.to_dataset()
            if "draw" not in group.sizes:
                continue

            group = group.isel(draw=slice(None, None, stride))
            group = group.assign_coords(draw=np.arange(group.sizes["draw"]))
            setattr(inference_data, group_name, group)

    deterministic_names = []
    if reconstruct_model_rate or reconstruct_pointwise:
        deterministic_names.append("ln_rate_model")
    if reconstruct_pointwise:
        deterministic_names.extend(built.mechanism_result.pointwise)

    missing = [
        name for name in dict.fromkeys(deterministic_names)
        if name not in inference_data.posterior
    ]

    if missing:
        posterior = compute_posterior_deterministics(
            inference_data,
            built,
            var_names=missing,
            backend="numba",
            progressbar=progressbar,
        )
        if hasattr(posterior, "to_dataset"):
            posterior = posterior.to_dataset()

        inference_data = inference_data.copy()
        inference_data.posterior = posterior

    if not hasattr(inference_data, "log_likelihood"):
        raise ValueError(
            "Posterior does not contain pointwise log likelihood. "
            "Resume the fit before postprocessing."
        )

    if "rate_observed" not in inference_data.log_likelihood:
        raise ValueError(
            "Posterior log likelihood does not contain 'rate_observed'. "
            "Resume the fit before postprocessing."
        )

    return Pt100PosteriorRun(
        specification=fit.specification,
        metadata=metadata,
        output_dir=fit.output_dir,
        model_data=fit.model_data,
        inputs=fit.inputs,
        built_model=built,
        inference_data=inference_data,
        parameter_specs=fit.parameter_specs,
        free_parameter_names=free_parameter_names,
    )
