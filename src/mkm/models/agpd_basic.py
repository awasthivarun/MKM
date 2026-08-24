from dataclasses import dataclass, fields
from typing import Callable

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


def available_agpd_models():
    return tuple(_AGPD_MODEL_REGISTRY)


def available_agpd_composition_models():
    return _AGPD_COMPOSITION_MODELS


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


def build_agpd_mechanism(model_name, material, config):
    definition = get_agpd_model_definition(model_name)

    if material not in config["surface_composition"]:
        raise ValueError(f"Material '{material}' has no surface-composition configuration.")

    profile = get_agpd_prior_profile(config, material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(definition, parameter_specs, f"{material}/{model_name}")

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != (material,):
            raise ValueError(f"AgPd individual-material model '{model_name}' for '{material}' must be fit to exactly that one material.")

        prior_values = build_named_priors(parameter_specs)
        parameters = definition.parameter_class(**prior_values)
        state = build_agpd_point_state(inputs=point_inputs, config=config)
        evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])
        return evaluated.mechanism_result

    return mechanism


def build_agpd_composition_mechanism(model_name, materials, config, prior_material="Ag10Pd90"):
    """Build the shared-energetics baseline over multiple AgPd compositions.

    All thermodynamic and kinetic parameters are shared across materials. Composition enters only through
    the explicit Ag/Pd surface fractions already present in the mechanism. This is therefore a bifunctional/
    site-availability baseline rather than an electronic-effect model.
    """
    if model_name not in _AGPD_COMPOSITION_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not enabled for the shared-composition workflow. "
            f"Available models: {available_agpd_composition_models()}."
        )

    materials = tuple(materials)
    if len(materials) < 2:
        raise ValueError("A composition model requires at least two materials.")
    if len(set(materials)) != len(materials):
        raise ValueError("Composition-model materials must be unique.")

    missing = [material for material in materials if material not in config["surface_composition"]]
    if missing:
        raise ValueError(f"Missing surface-composition configuration for materials: {missing}.")

    definition = get_agpd_model_definition(model_name)
    profile = get_agpd_prior_profile(config, prior_material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(definition, parameter_specs, f"{prior_material}/{model_name}")

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != materials:
            raise ValueError(
                f"Shared AgPd composition model expected materials {materials}, "
                f"but model inputs contain {tuple(point_inputs.materials)}."
            )

        prior_values = build_named_priors(parameter_specs)
        parameters = definition.parameter_class(**prior_values)
        state = build_agpd_point_state(inputs=point_inputs, config=config)
        evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])
        return evaluated.mechanism_result

    return mechanism
