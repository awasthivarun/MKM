"""Shared fit specification and model assembly for AgPd posterior workflows."""

from dataclasses import dataclass

from mkm.inference.likelihoods import RATE_NORMAL, available_error_structures
from mkm.inference.model import build_pymc_model
from mkm.models.agpd_basic import (
    available_agpd_all_material_models,
    available_agpd_models,
    available_agpd_parameterizations,
    build_agpd_all_material_mechanism,
    build_agpd_mechanism,
    get_agpd_all_material_parameter_specs,
    get_agpd_prior_profile,
)
from mkm.workflows.agpd_basic import (
    available_agpd_materials,
    build_agpd_inputs,
    validate_agpd_material,
)


@dataclass(frozen=True)
class AgPdFitSpecification:
    fit_scope: str
    model_name: str
    material: str | None
    parameterization: str | None
    error_structure: str
    prior_material: str | None

    @property
    def is_all_materials(self):
        return self.fit_scope == "all_materials"


@dataclass(frozen=True)
class AgPdBuiltFit:
    specification: AgPdFitSpecification
    inputs: object
    built_model: object


def resolve_agpd_fit_specification(
    config,
    *,
    model_name,
    all_materials=False,
    material="Ag10Pd90",
    parameterization="shared",
    error_structure="material",
    prior_material="Ag10Pd90",
):
    supported_errors = tuple(config["likelihood"]["supported_error_structures"])
    unknown_errors = set(supported_errors) - set(available_error_structures())
    if unknown_errors:
        raise ValueError(
            f"Configuration contains unsupported error structures: {sorted(unknown_errors)}."
        )
    if error_structure not in supported_errors:
        raise ValueError(
            f"Error structure '{error_structure}' is not enabled. "
            f"Configured structures: {supported_errors}."
        )

    if all_materials:
        if model_name not in available_agpd_all_material_models():
            raise ValueError(
                f"Model '{model_name}' is not available for all-material fitting. "
                f"Available models: {available_agpd_all_material_models()}."
            )

        available_parameterizations = available_agpd_parameterizations(config, model_name)
        if parameterization not in available_parameterizations:
            raise ValueError(
                f"Parameterization '{parameterization}' is not configured for model "
                f"'{model_name}'. Available parameterizations: {available_parameterizations}."
            )
        if prior_material not in config.get("prior_profiles", {}):
            raise ValueError(f"No prior profile is configured for '{prior_material}'.")

        get_agpd_all_material_parameter_specs(
            config=config,
            prior_material=prior_material,
            model_name=model_name,
            parameterization=parameterization,
        )

        return AgPdFitSpecification(
            fit_scope="all_materials",
            model_name=model_name,
            material=None,
            parameterization=parameterization,
            error_structure=error_structure,
            prior_material=prior_material,
        )

    if model_name not in available_agpd_models():
        raise ValueError(
            f"Unknown AgPd model '{model_name}'. Available models: {available_agpd_models()}."
        )
    validate_agpd_material(config, material)
    if error_structure != "material":
        raise ValueError(
            "Individual-material fits use one material-indexed pair of error parameters. "
            "Use error_structure='material'."
        )

    get_agpd_prior_profile(config, material, model_name)

    return AgPdFitSpecification(
        fit_scope="individual",
        model_name=model_name,
        material=material,
        parameterization=None,
        error_structure="material",
        prior_material=material,
    )


def fit_materials(specification, config):
    if specification.is_all_materials:
        return available_agpd_materials(config)
    return (specification.material,)


def build_fit_mechanism(specification, inputs, config, *, prediction_only=False):
    if specification.is_all_materials:
        return build_agpd_all_material_mechanism(
            model_name=specification.model_name,
            materials=tuple(inputs.materials),
            config=config,
            prior_material=specification.prior_material,
            parameterization=specification.parameterization,
            prediction_only=prediction_only,
        )

    if prediction_only:
        raise ValueError("prediction_only is only used by all-material validation fits.")

    return build_agpd_mechanism(
        model_name=specification.model_name,
        material=specification.material,
        config=config,
    )


def fit_parameter_specs(specification, config):
    if specification.is_all_materials:
        return get_agpd_all_material_parameter_specs(
            config=config,
            prior_material=specification.prior_material,
            model_name=specification.model_name,
            parameterization=specification.parameterization,
        )

    return dict(
        get_agpd_prior_profile(
            config,
            specification.material,
            specification.model_name,
        )["parameters"]
    )


def rate_normal_likelihood_kwargs(config):
    likelihood = config["likelihood"]
    if likelihood.get("name") != RATE_NORMAL:
        raise ValueError(
            f"AgPd configuration must use likelihood '{RATE_NORMAL}', "
            f"not '{likelihood.get('name')}'."
        )

    configured_form = likelihood.get("sigma_form")
    expected_form = "sigma_abs + sigma_rel * model_rate"
    if configured_form != expected_form:
        raise ValueError(
            f"Configured sigma form must be '{expected_form}', not '{configured_form}'."
        )

    keys = (
        "sigma_abs_prior_median_s_inv",
        "sigma_abs_prior_log_sd",
        "sigma_rel_prior_median",
        "sigma_rel_prior_log_sd",
    )
    return {key: likelihood[key] for key in keys}


def error_parameter_specs(config):
    likelihood = config["likelihood"]
    return {
        "sigma_rate_abs": {
            "distribution": "lognormal",
            "median": float(likelihood["sigma_abs_prior_median_s_inv"]),
            "log_sd": float(likelihood["sigma_abs_prior_log_sd"]),
        },
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": float(likelihood["sigma_rel_prior_median"]),
            "log_sd": float(likelihood["sigma_rel_prior_log_sd"]),
        },
    }


def all_parameter_specs(specification, config):
    return {
        **fit_parameter_specs(specification, config),
        **error_parameter_specs(config),
    }


def build_agpd_fit_model(specification, model_data, config):
    inputs = build_agpd_inputs(model_data)
    mechanism = build_fit_mechanism(specification, inputs, config)
    built_model = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        likelihood_name=RATE_NORMAL,
        error_structure=specification.error_structure,
        likelihood_kwargs=rate_normal_likelihood_kwargs(config),
    )
    return AgPdBuiltFit(
        specification=specification,
        inputs=inputs,
        built_model=built_model,
    )


def fit_output_dir(paths, specification):
    return paths.agpd_posterior_output_dir(
        fit_scope=specification.fit_scope,
        model_name=specification.model_name,
        material=specification.material,
        parameterization=specification.parameterization,
        error_structure=specification.error_structure,
    )
