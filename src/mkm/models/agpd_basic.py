from dataclasses import dataclass, fields
from typing import Callable

import pytensor.tensor as pt

from mkm.inference.priors import build_named_priors
from mkm.mechanisms.agpd_basic import (
    AgPdBFParameters,
    AgPdBFLHParameters,
    AgPdCOBFERRLHParameters,
    build_agpd_point_state,
    evaluate_agpd_bf,
    evaluate_agpd_bf_lh,
    evaluate_agpd_co_bf_er_lh,
)


@dataclass(frozen=True)
class AgPdModelDefinition:
    parameter_class: type
    evaluator: Callable


_AGPD_MODEL_REGISTRY = {
    "BF": AgPdModelDefinition(parameter_class=AgPdBFParameters, evaluator=evaluate_agpd_bf),
    "BF_LH": AgPdModelDefinition(parameter_class=AgPdBFLHParameters, evaluator=evaluate_agpd_bf_lh),
    "CO_BF_ER_LH": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh,
    ),
}

_AGPD_COMPOSITION_MODELS = ("BF_LH", "CO_BF_ER_LH")
_AGPD_COMPOSITION_PARAMETERIZATIONS = ("shared", "linear_xAg")


def available_agpd_models():
    return tuple(_AGPD_MODEL_REGISTRY)


def available_agpd_composition_models():
    return _AGPD_COMPOSITION_MODELS


def available_agpd_composition_parameterizations():
    return _AGPD_COMPOSITION_PARAMETERIZATIONS


def get_agpd_model_definition(model_name):
    try:
        return _AGPD_MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown AgPd model '{model_name}'. Available models: {available_agpd_models()}.") from error


def get_agpd_prior_profile(config, material, model_name):
    try:
        return config["prior_profiles"][material][model_name]
    except KeyError as error:
        raise ValueError(f"No prior profile is defined for material '{material}' and model '{model_name}'.") from error


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


def _get_linear_xag_config(config, model_name):
    try:
        linear_config = config["composition_parameterizations"]["linear_xAg"]
        model_config = linear_config["models"][model_name]
    except KeyError as error:
        raise ValueError(
            f"No linear_xAg composition parameterization is configured for AgPd model '{model_name}'."
        ) from error

    x_reference = float(linear_config["x_reference"])
    if not 0.0 <= x_reference <= 1.0:
        raise ValueError("linear_xAg x_reference must lie in [0, 1].")

    slope_specs = model_config.get("slopes", {})
    if not slope_specs:
        raise ValueError(f"linear_xAg for '{model_name}' must configure at least one parameter slope.")

    definition = get_agpd_model_definition(model_name)
    mechanism_parameters = {field.name for field in fields(definition.parameter_class)}
    unknown = set(slope_specs) - mechanism_parameters
    if unknown:
        raise ValueError(f"linear_xAg contains unknown mechanism parameters: {sorted(unknown)}.")

    return x_reference, slope_specs


def get_agpd_composition_parameterization(config, model_name, composition_model="shared"):
    """Return x-reference and configured slope specs for a composition parameterization."""
    if composition_model == "shared":
        return None, {}
    if composition_model == "linear_xAg":
        return _get_linear_xag_config(config, model_name)

    raise ValueError(
        f"Unknown AgPd composition parameterization '{composition_model}'. "
        f"Available parameterizations: {available_agpd_composition_parameterizations()}."
    )


def get_agpd_composition_parameter_specs(config, prior_material, model_name, composition_model="shared"):
    profile = get_agpd_prior_profile(config, prior_material, model_name)
    parameter_specs = dict(profile["parameters"])

    _, slope_specs = get_agpd_composition_parameterization(
        config=config,
        model_name=model_name,
        composition_model=composition_model,
    )
    for parameter_name, spec in slope_specs.items():
        parameter_specs[f"{parameter_name}_xAg_slope"] = spec
    return parameter_specs


def build_agpd_mechanism(model_name, material, config):
    definition = get_agpd_model_definition(model_name)
    if material not in config["surface_composition"]:
        raise ValueError(f"Material '{material}' has no surface-composition configuration.")

    profile = get_agpd_prior_profile(config, material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(definition, parameter_specs, f"{material}/{model_name}")

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != (material,):
            raise ValueError(
                f"AgPd individual-material model '{model_name}' for '{material}' must be fit to exactly that one material."
            )
        prior_values = build_named_priors(parameter_specs)
        parameters = definition.parameter_class(**prior_values)
        state = build_agpd_point_state(inputs=point_inputs, config=config)
        evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])
        return evaluated.mechanism_result

    return mechanism


def build_agpd_composition_mechanism(
    model_name,
    materials,
    config,
    prior_material="Ag10Pd90",
    composition_model="shared",
    prediction_only=False,
):
    """Build an AgPd composition model with shared or linear-in-xAg energetics."""
    if model_name not in _AGPD_COMPOSITION_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not enabled for the composition workflow. "
            f"Available models: {available_agpd_composition_models()}."
        )
    if composition_model not in _AGPD_COMPOSITION_PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown AgPd composition parameterization '{composition_model}'. "
            f"Available parameterizations: {available_agpd_composition_parameterizations()}."
        )

    materials = tuple(materials)
    if not prediction_only and len(materials) < 2:
        raise ValueError("A composition model requires at least two materials for fitting.")
    if prediction_only and len(materials) < 1:
        raise ValueError("A composition prediction requires at least one material.")
    if len(set(materials)) != len(materials):
        raise ValueError("Composition-model materials must be unique.")

    missing = [material for material in materials if material not in config["surface_composition"]]
    if missing:
        raise ValueError(f"Missing surface-composition configuration for materials: {missing}.")

    definition = get_agpd_model_definition(model_name)
    profile = get_agpd_prior_profile(config, prior_material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(definition, parameter_specs, f"{prior_material}/{model_name}")

    x_reference, slope_specs = get_agpd_composition_parameterization(
        config=config,
        model_name=model_name,
        composition_model=composition_model,
    )
    slope_prior_specs = {
        f"{parameter_name}_xAg_slope": spec for parameter_name, spec in slope_specs.items()
    }

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != materials:
            raise ValueError(
                f"AgPd composition model expected materials {materials}, "
                f"but model inputs contain {tuple(point_inputs.materials)}."
            )

        prior_values = build_named_priors(parameter_specs)
        state = build_agpd_point_state(inputs=point_inputs, config=config)

        if composition_model == "linear_xAg":
            slope_values = build_named_priors(slope_prior_specs)
            x_shift = pt.as_tensor_variable(state.Ag_fraction) - x_reference
            effective_values = dict(prior_values)
            for parameter_name in slope_specs:
                slope_name = f"{parameter_name}_xAg_slope"
                effective_values[parameter_name] = prior_values[parameter_name] + slope_values[slope_name] * x_shift
            parameters = definition.parameter_class(**effective_values)
        else:
            parameters = definition.parameter_class(**prior_values)

        evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])
        return evaluated.mechanism_result

    return mechanism
