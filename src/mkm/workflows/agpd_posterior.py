"""Load, validate, and reconstruct AgPd posterior runs."""

from dataclasses import dataclass
import warnings

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.posterior import (
    compute_posterior_deterministics,
    load_inference_data,
)
from mkm.provenance import read_run_metadata, sha256_file
from mkm.workflows.agpd_basic import build_agpd_model_data
from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    build_agpd_fit_model,
    fit_materials,
    fit_output_dir,
    resolved_parameterization_metadata,
)
from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
)


@dataclass(frozen=True)
class AgPdPosteriorRun:
    specification: object
    metadata: dict
    output_dir: object
    model_data: object
    inputs: object
    built_model: object
    inference_data: object
    parameter_specs: dict
    free_parameter_names: tuple[str, ...]


def expected_agpd_run_metadata(specification, materials, config):
    return {
        "fit_scope": specification.fit_scope,
        "materials": list(materials),
        "model": specification.model_name,
        "likelihood": RATE_NORMAL,
        "error_structure": specification.error_structure,
        "parameterization": specification.parameterization,
        "parameterization_specification": resolved_parameterization_metadata(
            specification,
            config,
        ),
        "prior_material": specification.prior_material,
    }


def validate_agpd_run_metadata(
    metadata,
    paths,
    specification,
    materials,
    config,
    *,
    allowed_statuses=(RUN_STATUS_COMPLETE,),
):
    expected = expected_agpd_run_metadata(specification, materials, config)
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }

    input_metadata = metadata.get("inputs", {})

    current_data_hash = sha256_file(paths.agpd_selected_path)
    if input_metadata.get("data_sha256") != current_data_hash:
        mismatches["inputs.data_sha256"] = (
            input_metadata.get("data_sha256"),
            current_data_hash,
        )

    current_config_hash = sha256_file(paths.agpd_model_config_path)
    stored_config_hash = input_metadata.get("model_config_sha256")
    if stored_config_hash != current_config_hash:
        warnings.warn(
            "Model config file hash differs from the hash stored for this posterior. "
            "The requested run identity and resolved parameterization specification "
            "still match, so loading will continue. Do not interpret this as proof "
            "that other run-defining config entries, such as base priors, are unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )

    status = metadata.get("status")
    if status not in allowed_statuses:
        mismatches["status"] = (status, tuple(allowed_statuses))

    if mismatches:
        details = "; ".join(
            f"{key}: stored={stored!r}, expected={expected_value!r}"
            for key, (stored, expected_value) in mismatches.items()
        )
        raise ValueError(f"Posterior run metadata do not match the requested run: {details}")


def load_agpd_posterior_run(
    paths,
    config,
    specification,
    *,
    reconstruct_model_rate=False,
    reconstruct_pointwise=False,
    progressbar=True,
):
    output_dir = fit_output_dir(paths, specification)
    posterior_path = output_dir / "posterior.nc"
    metadata_path = output_dir / "run_metadata.yaml"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    materials = fit_materials(specification, config)
    model_data = build_agpd_model_data(paths, materials)
    fit = build_agpd_fit_model(specification, model_data, config)
    built = fit.built_model

    metadata = read_run_metadata(metadata_path)
    validate_agpd_run_metadata(
        metadata,
        paths,
        specification,
        fit.inputs.materials,
        config,
        allowed_statuses=(RUN_STATUS_COMPLETE,),
    )

    inference_data = load_inference_data(posterior_path)
    free_parameter_names = tuple(variable.name for variable in built.model.free_RVs)
    missing_free = [
        name for name in free_parameter_names if name not in inference_data.posterior
    ]
    if missing_free:
        raise ValueError(f"Posterior is missing fitted variables: {missing_free}")

    deterministic_names = []
    if reconstruct_model_rate or reconstruct_pointwise:
        deterministic_names.append("ln_rate_model")
    if reconstruct_pointwise:
        deterministic_names.extend(built.mechanism_result.pointwise)

    missing = [
        name for name in deterministic_names if name not in inference_data.posterior
    ]
    if missing:
        posterior = compute_posterior_deterministics(
            inference_data,
            built,
            var_names=missing,
            backend="numba",
            progressbar=progressbar,
        )
        inference_data = inference_data.copy()
        inference_data.posterior = posterior

    if not hasattr(inference_data, "log_likelihood"):
        raise ValueError(
            "Posterior does not contain pointwise log likelihood. "
            "Resume the fit before postprocessing."
        )
    if "rate_observed" not in inference_data.log_likelihood:
        raise ValueError(
            "Posterior log_likelihood does not contain 'rate_observed'. "
            "Resume the fit before postprocessing."
        )

    return AgPdPosteriorRun(
        specification=specification,
        metadata=metadata,
        output_dir=output_dir,
        model_data=model_data,
        inputs=fit.inputs,
        built_model=built,
        inference_data=inference_data,
        parameter_specs=all_parameter_specs(specification, config),
        free_parameter_names=free_parameter_names,
    )
