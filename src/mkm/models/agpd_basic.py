"""AgPd model assembly, prior resolution, and composition parameterization."""

from dataclasses import fields

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from mkm.inference.priors import build_named_priors
from mkm.mechanisms.agpd_basic import build_agpd_point_state, evaluate_agpd_co_bf_er_lh
from mkm.mechanisms.pd_basic import PdCOERLHParameters, evaluate_pd_co_er_lh
from mkm.models.agpd_registry import (
    FITTED_CAPS_MODEL,
    AgPdModelDefinition,
    agpd_model_metadata,
    available_agpd_all_material_models,
    available_agpd_models,
    get_agpd_model_definition,
)

_NORMALIZED_BOUNDED_LINEAR_PARAMETERS = {"beta_2_BF", "beta_2_ER", "q"}


def get_agpd_fixed_parameters(model_name):
    definition = get_agpd_model_definition(model_name)
    fixed_parameters = dict(definition.fixed_parameters)
    mechanism_parameters = {field.name for field in fields(definition.parameter_class)}
    unknown = set(fixed_parameters) - mechanism_parameters
    if unknown:
        raise ValueError(f"Model '{model_name}' fixes unknown mechanism parameters: {sorted(unknown)}.")
    return fixed_parameters


def get_agpd_prior_variant(model_name):
    return get_agpd_model_definition(model_name).prior_variant


def _apply_prior_variant(config, definition, profile):
    variant_name = definition.prior_variant
    if variant_name is None:
        return profile
    try:
        variant = config["prior_variants"][variant_name]
    except KeyError as error:
        raise ValueError(f"Model prior variant '{variant_name}' is not configured.") from error
    overrides = variant.get("parameters", {})
    projected = dict(profile)
    projected["parameters"] = {name: dict(spec) for name, spec in profile["parameters"].items()}
    unknown = set(overrides) - set(projected["parameters"])
    if unknown:
        raise ValueError(f"Prior variant '{variant_name}' overrides unknown parameters: {sorted(unknown)}.")
    for name, specification in overrides.items():
        projected["parameters"][name] = dict(specification)
    projected["provenance"] = f"{profile.get('provenance', 'unspecified')}+{variant_name}"
    projected["prior_variant"] = variant_name
    return projected


def _project_profile_to_definition(profile, definition, profile_label):
    parameter_names = tuple(field.name for field in fields(definition.parameter_class))
    missing = [name for name in parameter_names if name not in profile["parameters"]]
    if missing:
        raise ValueError(f"Prior source '{profile_label}' is missing parameters required by the model: {missing}.")
    projected = dict(profile)
    projected["parameters"] = {name: profile["parameters"][name] for name in parameter_names}
    return projected


def get_agpd_prior_profile(config, material, model_name):
    definition = get_agpd_model_definition(model_name)
    if definition.coverage_cap_mode == "fitted":
        profile = config["fitted_cap_calibration"]
    else:
        config_model_name = definition.config_model_name
        try:
            profile = config["prior_profiles"][material][config_model_name]
        except KeyError as error:
            raise ValueError(
                f"No prior profile is defined for material '{material}' and model '{model_name}'."
            ) from error
    profile = _apply_prior_variant(config, definition, profile)

    # Full-model aliases reuse the canonical profile verbatim. Reduced pathway models
    # project that canonical profile onto their smaller parameter dataclass. Validation
    # of a full profile belongs where it is consumed to build a fit, not at lookup time.
    canonical_definition = get_agpd_model_definition(definition.config_model_name)
    if definition.parameter_class is canonical_definition.parameter_class:
        return profile
    return _project_profile_to_definition(profile, definition, f"{material}/{definition.config_model_name}")


def _validate_parameter_specs(definition, parameter_specs, profile_label):
    expected_parameters = {field.name for field in fields(definition.parameter_class)}
    configured_parameters = set(parameter_specs)
    if configured_parameters != expected_parameters:
        missing = expected_parameters - configured_parameters
        extra = configured_parameters - expected_parameters
        raise ValueError(
            f"Prior profile for {profile_label} does not match the mechanism parameters. "
            f"Missing: {sorted(missing)}; extra: {sorted(extra)}."
        )


def _free_parameter_specs(model_name, parameter_specs):
    fixed_parameters = get_agpd_fixed_parameters(model_name)
    return {name: specification for name, specification in parameter_specs.items() if name not in fixed_parameters}


def available_agpd_parameterizations(config, model_name=None):
    if model_name is not None and get_agpd_model_definition(model_name).coverage_cap_mode == "fitted":
        return ("linear_xAg",)
    parameterizations = config.get("composition_parameterizations", {})
    config_model_name = None if model_name is None else get_agpd_model_definition(model_name).config_model_name
    names = []
    for name, specification in parameterizations.items():
        models = specification.get("models", {})
        if config_model_name is None or config_model_name in models:
            names.append(name)
    return tuple(names)


def get_agpd_parameterization(config, model_name, parameterization="shared"):
    definition = get_agpd_model_definition(model_name)
    if definition.coverage_cap_mode == "fitted":
        if parameterization != "linear_xAg":
            raise ValueError(
                f"AgPd fitted-cap calibration only supports parameterization 'linear_xAg', got '{parameterization}'."
            )
        calibration = config["fitted_cap_calibration"]
        return float(calibration["x_reference"]), dict(calibration["slopes"])
    try:
        specification = config["composition_parameterizations"][parameterization]
    except KeyError as error:
        raise ValueError(
            f"Unknown AgPd parameterization '{parameterization}'. "
            f"Available parameterizations: {available_agpd_parameterizations(config)}."
        ) from error
    try:
        model_specification = specification["models"][definition.config_model_name]
    except KeyError as error:
        raise ValueError(
            f"AgPd parameterization '{parameterization}' is not configured for model '{model_name}'. "
            f"Available parameterizations for this model: {available_agpd_parameterizations(config, model_name)}."
        ) from error
    x_reference = float(specification.get("x_reference", 0.5))
    if not 0.0 <= x_reference <= 1.0:
        raise ValueError(f"Parameterization '{parameterization}' x_reference must lie in [0, 1].")
    slope_specs = dict(model_specification.get("slopes", {}))
    mechanism_parameters = {field.name for field in fields(definition.parameter_class)}
    unknown = set(slope_specs) - mechanism_parameters
    if unknown:
        raise ValueError(
            f"Parameterization '{parameterization}' contains unknown mechanism parameters: {sorted(unknown)}."
        )
    fixed_parameters = get_agpd_fixed_parameters(model_name)
    return x_reference, {name: spec for name, spec in slope_specs.items() if name not in fixed_parameters}


def get_agpd_parameterization_metadata(config, model_name, parameterization):
    x_reference, slope_specs = get_agpd_parameterization(config, model_name, parameterization)
    metadata = {"name": parameterization, "x_reference": x_reference, "slopes": slope_specs}
    fixed_parameters = get_agpd_fixed_parameters(model_name)
    if fixed_parameters:
        metadata["fixed_parameters"] = fixed_parameters
    return metadata


def get_agpd_all_material_parameter_specs(config, prior_material, model_name, parameterization="shared"):
    profile = get_agpd_prior_profile(config, prior_material, model_name)
    definition = get_agpd_model_definition(model_name)
    _validate_parameter_specs(definition, profile["parameters"], f"{prior_material}/{model_name}")
    parameter_specs = _free_parameter_specs(model_name, profile["parameters"])
    _, slope_specs = get_agpd_parameterization(config, model_name, parameterization)
    for parameter_name, specification in slope_specs.items():
        parameter_specs[f"{parameter_name}_xAg_slope"] = specification
    if definition.coverage_cap_mode == "fitted":
        cap_spec = config["fitted_cap_calibration"]["theta_CO_max_prior"]
        for material in config["surface_composition"]:
            parameter_specs[f"theta_CO_max_{material}"] = dict(cap_spec)
    return parameter_specs


def build_agpd_mechanism(model_name, material, config):
    definition = get_agpd_model_definition(model_name)
    if not definition.supports_individual:
        raise ValueError(f"Model '{model_name}' is defined only for all-material fitting.")
    if material not in config["surface_composition"]:
        raise ValueError(f"Material '{material}' has no surface-composition configuration.")
    profile = get_agpd_prior_profile(config, material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(definition, parameter_specs, f"{material}/{model_name}")

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != (material,):
            raise ValueError(
                f"AgPd individual-material model '{model_name}' for '{material}' "
                "must be fit to exactly that one material."
            )
        prior_values = build_named_priors(parameter_specs)
        parameters = definition.parameter_class(**prior_values)
        state = build_agpd_point_state(inputs=point_inputs, config=config)
        evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])
        return evaluated.mechanism_result

    return mechanism


def _material_activity(state, disabled_materials):
    if not disabled_materials:
        return None
    if state.materials is None or state.material_index is None:
        raise ValueError("Material-specific pathway modifiers require material identity in the point state.")
    disabled = set(disabled_materials)
    by_material = np.asarray([0.0 if material in disabled else 1.0 for material in state.materials], dtype=float)
    return by_material[state.material_index]


def _evaluate_all_material_state(
    *,
    model_name,
    definition,
    state,
    effective_values,
    config,
    prediction_only,
    theta_CO_max_override=None,
):
    """Evaluate an all-material model, reducing pure-Pd prediction to the identifiable ER+LH model."""
    if prediction_only and definition.pure_pd_prediction_reduction and np.all(state.Ag_fraction == 0.0):
        pd_parameter_names = {field.name for field in fields(PdCOERLHParameters)}
        pd_parameters = PdCOERLHParameters(**{name: effective_values[name] for name in pd_parameter_names})
        theta_CO_max = None
        if definition.coverage_cap_mode == "config":
            theta_CO_max = state.theta_CO_max
        elif definition.coverage_cap_mode == "fitted":
            theta_CO_max = theta_CO_max_override
        return evaluate_pd_co_er_lh(
            state=state,
            parameters=pd_parameters,
            temperature_K=config["temperature_K"],
            theta_CO_max=theta_CO_max,
        )

    parameters = definition.parameter_class(**effective_values)
    if definition.coverage_cap_mode == "fitted":
        return evaluate_agpd_co_bf_er_lh(
            state=state,
            parameters=parameters,
            temperature_K=config["temperature_K"],
            theta_CO_max=theta_CO_max_override,
            bf_activity=_material_activity(state, definition.bf_disabled_materials),
            er_activity=_material_activity(state, definition.er_disabled_materials),
        )
    return definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])


def _bounded_linear_max_abs_slope(parameter_name, base_value, parameter_specs, x_reference):
    if not np.isclose(x_reference, 0.5):
        raise ValueError("Bounded linear_xAg parameters currently require x_reference = 0.5.")
    specification = parameter_specs[parameter_name]
    try:
        lower = float(specification["lower"])
        upper = float(specification["upper"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Bounded linear_xAg parameter '{parameter_name}' requires finite lower and upper prior bounds."
        ) from error
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"Bounded linear_xAg parameter '{parameter_name}' has invalid prior bounds [{lower}, {upper}]."
        )
    return 2.0 * pt.minimum(base_value - lower, upper - base_value)


def build_agpd_all_material_mechanism(
    model_name,
    materials,
    config,
    *,
    prior_material="Ag10Pd90",
    parameterization="shared",
    prediction_only=False,
):
    definition = get_agpd_model_definition(model_name)
    if not definition.supports_all_materials:
        raise ValueError(
            f"Model '{model_name}' is not enabled for all-material fitting. "
            f"Available models: {available_agpd_all_material_models()}."
        )
    materials = tuple(materials)
    if not prediction_only and len(materials) < 2:
        raise ValueError("An all-material model requires at least two materials for fitting.")
    if prediction_only and len(materials) < 1:
        raise ValueError("An all-material prediction requires at least one material.")
    if len(set(materials)) != len(materials):
        raise ValueError("All-material model materials must be unique.")
    missing = [material for material in materials if material not in config["surface_composition"]]
    if missing:
        raise ValueError(f"Missing surface-composition configuration for materials: {missing}.")

    profile = get_agpd_prior_profile(config, prior_material, model_name)
    _validate_parameter_specs(definition, profile["parameters"], f"{prior_material}/{model_name}")
    parameter_specs = _free_parameter_specs(model_name, profile["parameters"])
    fixed_parameters = get_agpd_fixed_parameters(model_name)
    x_reference, slope_specs = get_agpd_parameterization(config, model_name, parameterization)
    slope_prior_specs = {f"{name}_xAg_slope": spec for name, spec in slope_specs.items()}
    cap_prior_specs = {}
    if definition.coverage_cap_mode == "fitted":
        cap_spec = config["fitted_cap_calibration"]["theta_CO_max_prior"]
        cap_prior_specs = {f"theta_CO_max_{material}": dict(cap_spec) for material in config["surface_composition"]}

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != materials:
            raise ValueError(
                f"AgPd all-material model expected materials {materials}, "
                f"but model inputs contain {tuple(point_inputs.materials)}."
            )
        prior_values = build_named_priors(parameter_specs)
        slope_values = build_named_priors(slope_prior_specs)
        cap_values = build_named_priors(cap_prior_specs)
        fixed_values = {
            name: pm.Deterministic(name, pt.as_tensor_variable(float(value)))
            for name, value in fixed_parameters.items()
        }
        state = build_agpd_point_state(inputs=point_inputs, config=config)
        effective_values = {**fixed_values, **prior_values}
        if slope_specs:
            x_shift = pt.as_tensor_variable(state.Ag_fraction) - x_reference
            for parameter_name in slope_specs:
                slope_name = f"{parameter_name}_xAg_slope"
                slope = slope_values[slope_name]
                if parameter_name in _NORMALIZED_BOUNDED_LINEAR_PARAMETERS:
                    slope = slope * _bounded_linear_max_abs_slope(
                        parameter_name=parameter_name,
                        base_value=prior_values[parameter_name],
                        parameter_specs=parameter_specs,
                        x_reference=x_reference,
                    )
                effective_values[parameter_name] = prior_values[parameter_name] + slope * x_shift

        theta_CO_max_override = None
        if definition.coverage_cap_mode == "fitted":
            theta_CO_max_override = pt.zeros_like(pt.as_tensor_variable(state.Ag_fraction))
            for material_index, material in enumerate(point_inputs.materials):
                material_mask = pt.as_tensor_variable((state.material_index == material_index).astype(float))
                theta_CO_max_override = (
                    theta_CO_max_override + cap_values[f"theta_CO_max_{material}"] * material_mask
                )

        evaluated = _evaluate_all_material_state(
            model_name=model_name,
            definition=definition,
            state=state,
            effective_values=effective_values,
            config=config,
            prediction_only=prediction_only,
            theta_CO_max_override=theta_CO_max_override,
        )
        return evaluated.mechanism_result

    return mechanism


__all__ = [
    "AgPdModelDefinition",
    "agpd_model_metadata",
    "available_agpd_all_material_models",
    "available_agpd_models",
    "available_agpd_parameterizations",
    "build_agpd_all_material_mechanism",
    "build_agpd_mechanism",
    "get_agpd_all_material_parameter_specs",
    "get_agpd_fixed_parameters",
    "get_agpd_model_definition",
    "get_agpd_parameterization",
    "get_agpd_parameterization_metadata",
    "get_agpd_prior_profile",
    "get_agpd_prior_variant",
]
