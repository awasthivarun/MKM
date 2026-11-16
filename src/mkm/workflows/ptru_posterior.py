"""Unified PtRu posterior provenance, validation, and deterministic reconstruction."""

from copy import deepcopy
from dataclasses import dataclass

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.posterior import (
    compute_posterior_deterministics, load_inference_data, validate_inference_data_alignment,
)
from mkm.models.ptru import ptru_parameterization_metadata
from mkm.provenance import (
    PTRU_MODEL_SOURCE_FILES, build_fit_metadata, read_run_metadata, sha256_mapping,
    source_file_hashes, validate_fit_input_identity,
)
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE
from mkm.workflows.ptru_fit import build_ptru_fit


@dataclass(frozen=True)
class PtRuPosteriorRun:
    specification: object
    metadata: dict
    output_dir: object
    model_data: object
    inputs: object
    built_model: object
    inference_data: object
    parameter_specs: dict
    free_parameter_names: tuple[str, ...]


def ptru_model_specification(paths, fit):
    spec = fit.specification
    return {
        "family": "PtRu", "medium": spec.medium, "model_name": spec.model_name,
        "materials": list(spec.materials), "assumptions": deepcopy(fit.config["mechanism_assumptions"]),
        "surface_composition": {material: deepcopy(fit.config["surface_composition"][material])
                                for material in spec.materials},
        "source_sha256": source_file_hashes(paths.root, PTRU_MODEL_SOURCE_FILES),
    }


def build_ptru_run_metadata(paths, fit, sampler):
    spec = fit.specification
    return build_fit_metadata(
        root=paths.root, fit_scope=spec.fit_scope, materials=fit.inputs.materials,
        model_name=spec.model_name, error_structure=spec.error_structure,
        data_path=fit.data_path, model_config_path=fit.model_config_path,
        model_config_paths=(fit.model_config_path, fit.prior_config_path),
        sampler=sampler, parameterization=spec.parameterization,
        parameterization_specification=ptru_parameterization_metadata(fit.config, spec),
        model_specification=ptru_model_specification(paths, fit), prior_specification=fit.parameter_specs,
    )


def validate_ptru_run_metadata(metadata, paths, fit, *, allowed_statuses=(RUN_STATUS_COMPLETE,)):
    spec = fit.specification
    model_specification = ptru_model_specification(paths, fit)
    parameterization_specification = ptru_parameterization_metadata(fit.config, spec)
    resolved = {
        "model_specification": model_specification,
        "parameterization_specification": parameterization_specification,
        "prior_specification": fit.parameter_specs,
        "error_structure": spec.error_structure,
        "likelihood": RATE_NORMAL,
    }
    expected = {
        "fit_scope": spec.fit_scope, "materials": list(fit.inputs.materials),
        "model": spec.model_name, "likelihood": RATE_NORMAL,
        "error_structure": spec.error_structure, "parameterization": spec.parameterization,
        "parameterization_specification": parameterization_specification,
        "model_specification": model_specification, "prior_specification": fit.parameter_specs,
        "resolved_fit_sha256": sha256_mapping(resolved),
    }
    mismatches = [name for name, value in expected.items() if metadata.get(name) != value]
    mismatches.extend(validate_fit_input_identity(
        metadata, paths.root, fit.data_path, fit.model_config_path,
        (fit.model_config_path, fit.prior_config_path),
    ))
    if metadata.get("status") not in allowed_statuses:
        mismatches.append("status")
    if mismatches:
        raise ValueError("PtRu checkpoint mismatch: " + ", ".join(mismatches))


def load_ptru_posterior_run(
    paths, specification, *, reconstruct_model_rate=False, reconstruct_pointwise=False, progressbar=True,
):
    fit = build_ptru_fit(paths, specification)
    posterior_path = fit.output_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")
    metadata = read_run_metadata(fit.output_dir / "run_metadata.yaml")
    validate_ptru_run_metadata(metadata, paths, fit)
    inference_data = load_inference_data(posterior_path)
    names = tuple(variable.name for variable in fit.built_model.model.free_RVs)
    missing = [name for name in names if name not in inference_data.posterior]
    if missing:
        raise ValueError(f"PtRu posterior is missing fitted variables: {missing}")
    deterministic_names = []
    if reconstruct_model_rate or reconstruct_pointwise:
        deterministic_names.append("ln_rate_model")
    if reconstruct_pointwise:
        deterministic_names.extend(fit.built_model.mechanism_result.pointwise)
    absent = [name for name in dict.fromkeys(deterministic_names) if name not in inference_data.posterior]
    if absent:
        posterior = compute_posterior_deterministics(
            inference_data, fit.built_model, var_names=absent, backend="numba", progressbar=progressbar,
        )
        if hasattr(posterior, "to_dataset"):
            posterior = posterior.to_dataset()
        inference_data = inference_data.copy()
        inference_data.posterior = posterior
    if not hasattr(inference_data, "log_likelihood") or "rate_observed" not in inference_data.log_likelihood:
        raise ValueError("PtRu posterior lacks rate_observed log likelihood; resume fitting first.")
    validate_inference_data_alignment(inference_data)
    return PtRuPosteriorRun(
        specification=specification, metadata=metadata, output_dir=fit.output_dir,
        model_data=fit.model_data, inputs=fit.inputs, built_model=fit.built_model,
        inference_data=inference_data, parameter_specs=fit.parameter_specs, free_parameter_names=names,
    )
