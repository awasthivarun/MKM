
"""Pt100 mechanism assembly using shared AgPd/PtRu inference infrastructure."""

from dataclasses import fields

from mkm.inference.priors import build_named_priors
from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES, evaluate_pt100_co


def build_pt100_mechanism(model_name, medium, temperature_K, parameter_specs):
    """Build a PyMC-compatible Pt100 mechanism with configurable parameter priors."""
    if model_name not in PT100_PARAMETER_CLASSES:
        raise ValueError(f"Unknown Pt100 model: {model_name}")
    if medium not in ("acidic", "basic"):
        raise ValueError(f"Unknown Pt100 medium: {medium}")

    parameter_class = PT100_PARAMETER_CLASSES[model_name]
    expected = {field.name for field in fields(parameter_class)}
    configured = set(parameter_specs)

    if configured != expected:
        raise ValueError(
            f"Pt100 {model_name} prior mismatch: "
            f"missing={sorted(expected - configured)}, "
            f"extra={sorted(configured - expected)}"
        )

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != ("Pt100",):
            raise ValueError("Pt100 fitting requires exactly one material: Pt100.")

        prior_values = build_named_priors(parameter_specs)
        parameters = parameter_class(**prior_values)

        return evaluate_pt100_co(
            state=point_inputs,
            parameters=parameters,
            temperature_K=temperature_K,
            medium=medium,
            model_name=model_name,
        )

    return mechanism
