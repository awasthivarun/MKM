"""Load, validate, and reconstruct AgPd posterior runs."""

from dataclasses import dataclass

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.posterior import (
    compute_posterior_deterministics,
    load_inference_data,
)
from mkm.provenance import (
    AGPD_MODEL_SOURCE_FILES, read_run_metadata, sha256_mapping,
    source_file_hashes, validate_fit_input_identity,
)
from mkm.workflows.agpd_basic import build_agpd_model_data
from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    build_agpd_fit_model,
    fit_materials,
    fit_output_dir,
    resolved_model_metadata,
    resolved_parameterization_metadata,
    resolved_prior_metadata,
)
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE


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


def agpd_model_specification(paths, specification):
    return {**resolved_model_metadata(specification),
            "source_sha256": source_file_hashes(paths.root, AGPD_MODEL_SOURCE_FILES)}


def expected_agpd_run_metadata(specification, materials, config, paths=None):
    model_specification = (agpd_model_specification(paths, specification) if paths is not None
                           else resolved_model_metadata(specification))
    parameterization_specification = resolved_parameterization_metadata(specification, config)
    prior_specification = resolved_prior_metadata(specification, config)
    resolved_fit = {
        "model_specification": model_specification,
        "parameterization_specification": parameterization_specification,
        "prior_specification": prior_specification,
        "error_structure": specification.error_structure,
        "likelihood": RATE_NORMAL,
    }
    return {
        "fit_scope": specification.fit_scope,
        "materials": list(materials),
        "model": specification.model_name,
        "likelihood": RATE_NORMAL,
        "error_structure": specification.error_structure,
        "parameterization": specification.parameterization,
        "parameterization_specification": parameterization_specification,
        "prior_material": specification.prior_material,
        "model_specification": model_specification,
        "prior_specification": prior_specification,
        "resolved_fit_sha256": sha256_mapping(resolved_fit),
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
    expected = expected_agpd_run_metadata(specification, materials, config, paths=paths)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    mismatches.extend(validate_fit_input_identity(
        metadata, paths.root, paths.agpd_selected_path,
        paths.agpd_model_config_path, paths.agpd_model_config_paths,
    ))
    if metadata.get("status") not in allowed_statuses:
        mismatches.append("status")
    if mismatches:
        raise ValueError("Posterior run metadata do not match the requested run: " + ", ".join(mismatches))


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

    from mkm.inference.posterior import validate_inference_data_alignment
    validate_inference_data_alignment(inference_data)
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
