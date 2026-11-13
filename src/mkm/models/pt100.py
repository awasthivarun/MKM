"""Pt100 mechanism assembly using the PtRu registry and shared inference infrastructure."""

from dataclasses import fields

from mkm.inference.priors import build_named_priors
from mkm.models.ptru_registry import get_ptru_model_definition


def build_pt100_mechanism(model_name, medium, temperature_K, parameter_specs):
    """Build a PyMC-compatible Pt100 mechanism with configurable parameter priors."""
    definition = get_ptru_model_definition(model_name)

    if medium not in definition.media:
        raise ValueError(f"Unknown Pt100 medium: {medium}")

    parameter_class = definition.parameter_class
    expected = {field.name for field in fields(parameter_class)}
    configured = set(parameter_specs)

    if configured != expected:
        raise ValueError(
            f"Pt100 {model_name} prior mismatch: "
            f"missing={sorted(expected - configured)}, "
            f"extra={sorted(configured - expected)}"
        )

    def mechanism(point_inputs):
        if tuple(point_inputs.materials) != (definition.material,):
            raise ValueError("Pt100 fitting requires exactly one material: Pt100.")

        prior_values = build_named_priors(parameter_specs)
        parameters = parameter_class(**prior_values)

        return definition.evaluator(
            state=point_inputs,
            parameters=parameters,
            temperature_K=temperature_K,
            medium=medium,
            model_name=model_name,
        )

    return mechanism