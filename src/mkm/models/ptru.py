"""Unified PtRu mechanism registry, parameterizations, and PyMC assembly."""

from dataclasses import fields

import numpy as np
import pytensor.tensor as pt

from mkm.inference.priors import build_named_priors
from mkm.mechanisms.ptru import PURE_PT_MODELS, PTRU_PARAMETER_CLASSES, evaluate_ptru_co


def available_ptru_models():
    return tuple(PTRU_PARAMETER_CLASSES)


def ptru_parameter_names(model_name, material=None):
    if model_name not in PTRU_PARAMETER_CLASSES:
        raise ValueError(f"Unknown PtRu model: {model_name}")
    if material == "Pt100":
        if model_name == "CO_BF":
            raise ValueError("BF-only fitting is unavailable for Pt100.")
        active = {field.name for field in fields(PTRU_PARAMETER_CLASSES["CO_ER_LH"])}
    else:
        active = {field.name for field in fields(PTRU_PARAMETER_CLASSES[model_name])}
    return tuple(field.name for field in fields(PTRU_PARAMETER_CLASSES[model_name]) if field.name in active)


def ptru_parameter_specs(priors, medium, model_name, material=None):
    profile = priors["prior_profiles"][medium]["parameters"]
    names = ptru_parameter_names(model_name, material)
    missing = set(names) - set(profile)
    if missing:
        raise ValueError(f"Missing PtRu {medium} priors: {sorted(missing)}")
    return {name: dict(profile[name]) for name in names}


def ptru_all_parameter_specs(priors, config, spec):
    names = ptru_parameter_names(spec.model_name)
    if spec.is_all_materials and spec.parameterization == "independent":
        return {
            f"{name}_{material}": definition
            for material in spec.materials
            for name, definition in ptru_parameter_specs(priors, spec.medium, spec.model_name, material).items()
        }
    result = ptru_parameter_specs(priors, spec.medium, spec.model_name, spec.material)
    if spec.is_all_materials and spec.parameterization == "linear_xRu":
        slopes = config["parameterizations"]["linear_xRu"]["slopes"]
        result.update({f"{name}_xRu_slope": dict(slopes[name]) for name in names})
    return result


def ptru_parameterization_metadata(config, spec):
    if not spec.is_all_materials:
        return None
    if spec.parameterization == "independent":
        return {"name": "independent", "independent_materials": True, "materials": list(spec.materials)}
    if spec.parameterization == "shared":
        return {"name": "shared", "materials": list(spec.materials)}
    return {
        "name": "linear_xRu", "materials": list(spec.materials),
        "x_reference": float(config["parameterizations"]["linear_xRu"]["x_reference"]),
        "slopes": {name: config["parameterizations"]["linear_xRu"]["slopes"][name]
                   for name in ptru_parameter_names(spec.model_name)},
        "bounded_normalized_slopes": ["beta_2_ER"] if "ER" in spec.model_name else [],
    }


def ptru_pointwise_ru_fraction(point_inputs, config):
    fractions = np.array([config["surface_composition"][name]["Ru_fraction"]
                          for name in point_inputs.materials], dtype=float)
    return fractions[point_inputs.material_index]


def build_ptru_mechanism(spec, config, priors):
    """Construct one PyMC graph for pure Pt, alloy, or pooled PtRu data."""
    model_name = spec.model_name
    param_names = ptru_parameter_names(model_name)
    prior_specs = ptru_all_parameter_specs(priors, config, spec)
    parameterization = spec.parameterization

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != tuple(spec.materials):
            raise ValueError(f"PtRu expected {spec.materials}; observed {point_inputs.materials}.")
        sampled = build_named_priors(prior_specs)
        point_index = np.asarray(point_inputs.material_index, dtype=int)
        ru_fraction = ptru_pointwise_ru_fraction(point_inputs, config)
        n_points = len(point_index)
        params = {}
        if spec.is_all_materials and parameterization == "independent":
            for name in param_names:
                available = [material for material in spec.materials
                             if name in ptru_parameter_names(model_name, material)]
                # BF values on pure Pt are irrelevant because BF availability is exactly zero.
                fallback = sampled[f"{name}_{available[0]}"] if available else pt.as_tensor_variable(0.0)
                params[name] = pt.zeros(n_points)
                for index, material in enumerate(spec.materials):
                    value = sampled[f"{name}_{material}"] if material in available else fallback
                    params[name] = params[name] + value * (point_index == index).astype(float)
        else:
            params = {name: sampled[name] for name in param_names}
            if spec.is_all_materials and parameterization == "linear_xRu":
                x_reference = float(config["parameterizations"]["linear_xRu"]["x_reference"])
                x_shift = ru_fraction - x_reference
                for name in param_names:
                    slope = sampled[f"{name}_xRu_slope"]
                    if name == "beta_2_ER":
                        limits = prior_specs[name]
                        margin = pt.minimum(params[name] - limits["lower"], limits["upper"] - params[name])
                        # The fractional slope is scaled to preserve beta in [0, 1] over all compositions.
                        slope = slope * margin / max(abs(x_reference), abs(0.5 - x_reference))
                    params[name] = params[name] + slope * x_shift
        parameters = PTRU_PARAMETER_CLASSES[model_name](**params)
        return evaluate_ptru_co(
            point_inputs, parameters, config["temperature_K"], spec.medium, model_name, ru_fraction,
        )

    return mechanism
