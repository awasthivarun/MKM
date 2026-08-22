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


def available_agpd_models():
    return tuple(_AGPD_MODEL_REGISTRY)


def get_agpd_model_definition(model_name):
    try:
        return _AGPD_MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown AgPd model '{model_name}'. Available models: {available_agpd_models()}.") from error


def build_agpd_mechanism(model_name, material, config):
    definition = get_agpd_model_definition(model_name)

    if material not in config["surface_composition"]:
        raise ValueError(f"Material '{material}' has no surface-composition configuration.")

    try:
        profile = config["prior_profiles"][material][model_name]
    except KeyError as error:
        raise ValueError(f"No prior profile is defined for material '{material}' and model '{model_name}'.") from error

    parameter_specs = profile["parameters"]

    expected_parameters = {field.name for field in fields(definition.parameter_class)}
    configured_parameters = set(parameter_specs)

    if configured_parameters != expected_parameters:
        missing = expected_parameters - configured_parameters
        extra = configured_parameters - expected_parameters

        raise ValueError(
            f"Prior profile for {material}/{model_name} does not match the mechanism parameters. "
            f"Missing: {sorted(missing)}; extra: {sorted(extra)}."
        )

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != (material,):
            raise ValueError(f"AgPd legacy-style model '{model_name}' for '{material}' must be fit to exactly that one material.")

        prior_values = build_named_priors(parameter_specs)
        parameters = definition.parameter_class(**prior_values)
        state = build_agpd_point_state(inputs=point_inputs, config=config)

        evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])

        return evaluated.mechanism_result

    return mechanism