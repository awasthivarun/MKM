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
    "BF": AgPdModelDefinition(
        parameter_class=AgPdBFParameters,
        evaluator=evaluate_agpd_bf,
    ),
    "BF_LH": AgPdModelDefinition(
        parameter_class=AgPdBFLHParameters,
        evaluator=evaluate_agpd_bf_lh,
    ),
    "CO_BF_ER_LH": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh,
    ),
}


def available_agpd_models():
    return tuple(_AGPD_MODEL_REGISTRY)


def available_agpd_all_material_models():
    # The configured all-material dataset includes Pd100. A BF-only rate is
    # structurally zero there and the BF evaluator requires a positive Ag fraction.
    return ("BF_LH", "CO_BF_ER_LH")


def get_agpd_model_definition(model_name):
    try:
        return _AGPD_MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown AgPd model '{model_name}'. Available models: {available_agpd_models()}."
        ) from error


def get_agpd_prior_profile(config, material, model_name):
    try:
        return config["prior_profiles"][material][model_name]
    except KeyError as error:
        raise ValueError(
            f"No prior profile is defined for material '{material}' and model '{model_name}'."
        ) from error


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


def available_agpd_parameterizations(config, model_name=None):
    parameterizations = config.get("composition_parameterizations", {})
    names = []

    for name, specification in parameterizations.items():
        models = specification.get("models", {})
        if model_name is None or model_name in models:
            names.append(name)

    return tuple(names)


def get_agpd_parameterization(config, model_name, parameterization="shared"):
    try:
        specification = config["composition_parameterizations"][parameterization]
    except KeyError as error:
        raise ValueError(
            f"Unknown AgPd parameterization '{parameterization}'. "
            f"Available parameterizations: {available_agpd_parameterizations(config)}."
        ) from error

    try:
        model_specification = specification["models"][model_name]
    except KeyError as error:
        raise ValueError(
            f"AgPd parameterization '{parameterization}' is not configured for model '{model_name}'. "
            f"Available parameterizations for this model: "
            f"{available_agpd_parameterizations(config, model_name)}."
        ) from error

    x_reference = float(specification.get("x_reference", 0.5))
    if not 0.0 <= x_reference <= 1.0:
        raise ValueError(
            f"Parameterization '{parameterization}' x_reference must lie in [0, 1]."
        )

    slope_specs = dict(model_specification.get("slopes", {}))
    definition = get_agpd_model_definition(model_name)
    mechanism_parameters = {field.name for field in fields(definition.parameter_class)}
    unknown = set(slope_specs) - mechanism_parameters

    if unknown:
        raise ValueError(
            f"Parameterization '{parameterization}' contains unknown mechanism parameters: "
            f"{sorted(unknown)}."
        )

    return x_reference, slope_specs


def get_agpd_all_material_parameter_specs(
    config,
    prior_material,
    model_name,
    parameterization="shared",
):
    profile = get_agpd_prior_profile(config, prior_material, model_name)
    parameter_specs = dict(profile["parameters"])

    _, slope_specs = get_agpd_parameterization(
        config=config,
        model_name=model_name,
        parameterization=parameterization,
    )
    for parameter_name, specification in slope_specs.items():
        parameter_specs[f"{parameter_name}_xAg_slope"] = specification

    return parameter_specs


def build_agpd_mechanism(model_name, material, config):
    definition = get_agpd_model_definition(model_name)

    if material not in config["surface_composition"]:
        raise ValueError(
            f"Material '{material}' has no surface-composition configuration."
        )

    profile = get_agpd_prior_profile(config, material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(
        definition,
        parameter_specs,
        f"{material}/{model_name}",
    )

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != (material,):
            raise ValueError(
                f"AgPd individual-material model '{model_name}' for '{material}' "
                "must be fit to exactly that one material."
            )

        prior_values = build_named_priors(parameter_specs)
        parameters = definition.parameter_class(**prior_values)
        state = build_agpd_point_state(inputs=point_inputs, config=config)
        evaluated = definition.evaluator(
            state=state,
            parameters=parameters,
            temperature_K=config["temperature_K"],
        )
        return evaluated.mechanism_result

    return mechanism


def build_agpd_all_material_mechanism(
    model_name,
    materials,
    config,
    *,
    prior_material="Ag10Pd90",
    parameterization="shared",
    prediction_only=False,
):
    if model_name not in available_agpd_all_material_models():
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

    missing = [
        material
        for material in materials
        if material not in config["surface_composition"]
    ]
    if missing:
        raise ValueError(
            f"Missing surface-composition configuration for materials: {missing}."
        )

    definition = get_agpd_model_definition(model_name)
    profile = get_agpd_prior_profile(config, prior_material, model_name)
    parameter_specs = profile["parameters"]
    _validate_parameter_specs(
        definition,
        parameter_specs,
        f"{prior_material}/{model_name}",
    )

    x_reference, slope_specs = get_agpd_parameterization(
        config=config,
        model_name=model_name,
        parameterization=parameterization,
    )
    slope_prior_specs = {
        f"{parameter_name}_xAg_slope": specification
        for parameter_name, specification in slope_specs.items()
    }

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != materials:
            raise ValueError(
                f"AgPd all-material model expected materials {materials}, "
                f"but model inputs contain {tuple(point_inputs.materials)}."
            )

        prior_values = build_named_priors(parameter_specs)
        slope_values = build_named_priors(slope_prior_specs)
        state = build_agpd_point_state(inputs=point_inputs, config=config)

        effective_values = dict(prior_values)
        if slope_specs:
            x_shift = pt.as_tensor_variable(state.Ag_fraction) - x_reference
            for parameter_name in slope_specs:
                slope_name = f"{parameter_name}_xAg_slope"
                effective_values[parameter_name] = (
                    prior_values[parameter_name]
                    + slope_values[slope_name] * x_shift
                )

        parameters = definition.parameter_class(**effective_values)
        evaluated = definition.evaluator(
            state=state,
            parameters=parameters,
            temperature_K=config["temperature_K"],
        )
        return evaluated.mechanism_result

    return mechanism
