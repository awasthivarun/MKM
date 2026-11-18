"""Load, validate, and reconstruct AgPd posterior runs."""

from copy import deepcopy
from dataclasses import dataclass

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.models.agpd_registry import AGPD_MODEL_REGISTRY, get_agpd_model_definition
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


def agpd_model_specification(paths, specification, config):
    definition = get_agpd_model_definition(specification.model_name)
    registry = "src/mkm/models/agpd_registry.py"
    evaluator = getattr(definition.evaluator, "func", definition.evaluator)
    mechanism_file = "src/" + evaluator.__module__.replace(".", "/") + ".py"
    dynamic = specification.model_name not in AGPD_MODEL_REGISTRY
    if dynamic:
        # Dynamic definitions have no literal registry entry or source dataclass declaration.
        # Hash the complete defining files until symbol-scoped dynamic hashes are supported.
        dependencies = {}
    else:
        dependencies = {
            registry: ("AGPD_MODEL_REGISTRY", "get_agpd_model_definition", "agpd_model_metadata",
                       "available_agpd_models", "available_agpd_all_material_models"),
            mechanism_file: (evaluator.__name__, definition.parameter_class.__name__),
        }
    excluded = {"src/mkm/mechanisms/agpd_basic/__init__.py"}
    if not dynamic:
        excluded.add("src/mkm/mechanisms/agpd_basic/legacy_qea.py")
    for name in ("src/mkm/mechanisms/agpd_basic/finite_co.py",
                 "src/mkm/mechanisms/agpd_basic/full_ssa.py"):
        if name != mechanism_file:
            excluded.add(name)
    model_config = {
        "temperature_K": config["temperature_K"], "gas": deepcopy(config["gas"]),
        "electrolyte": deepcopy(config["electrolyte"]),
        "surface_composition": {material: deepcopy(config["surface_composition"][material])
                                for material in (specification.material,) if material is not None}
                               if not specification.is_all_materials else
                               {material: deepcopy(config["surface_composition"][material])
                                for material in fit_materials(specification, config)},
        "surface_assumptions": deepcopy(config.get("surface_assumptions", {})),
    }
    if definition.coverage_cap_mode == "config":
        model_config["co_coverage_cap"] = {material: config["co_coverage_cap"][material]
                                            for material in model_config["surface_composition"]}
    if definition.config_model_name == "CO_BF_ER_LH_3_OH":
        model_config["full_ssa"] = deepcopy(config["full_ssa"])
    if evaluator.__name__ == "evaluate_agpd_mixed_adsorption":
        model_config["mixed_adsorption"] = deepcopy(config.get("mixed_adsorption", {}))
    elif evaluator.__name__ == "evaluate_agpd_reversible_subset":
        model_config["reversible_ssa"] = deepcopy(config.get("reversible_ssa", {}))
    return {
        **resolved_model_metadata(specification), "kinetic_config": model_config,
        "source_sha256": source_file_hashes(
            paths.root, (*AGPD_MODEL_SOURCE_FILES, mechanism_file,
                         *(("src/mkm/models/agpd_physics.py",) if dynamic else ())),
            scoped_symbols=dependencies,
            scoped_registry_entries={} if dynamic else {registry: {"AGPD_MODEL_REGISTRY": specification.model_name}},
            skip_files=excluded,
        ),
    }


def expected_agpd_run_metadata(specification, materials, config, paths=None):
    model_specification = (agpd_model_specification(paths, specification, config) if paths is not None
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
