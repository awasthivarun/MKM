from dataclasses import dataclass

import pymc as pm
import pytensor.tensor as pt

from mkm.inference.likelihoods import RATE_NORMAL, RateNormalLikelihood, add_likelihood
from mkm.mechanisms.base import MechanismResult, validate_mechanism_result
from mkm.model_inputs import ModelInputArrays, build_model_coords, build_model_point_inputs


@dataclass(frozen=True)
class BuiltModel:
    model: pm.Model
    mechanism_result: MechanismResult
    likelihood_name: str
    error_structure: str
    likelihood: RateNormalLikelihood


def _specify_model_point_vector(value, n_model_points, name):
    tensor = pt.as_tensor_variable(value)
    if tensor.ndim != 1:
        raise ValueError(f"'{name}' must be one-dimensional.")

    static_length = tensor.type.shape[0]
    if static_length is not None and static_length != n_model_points:
        raise ValueError(
            f"'{name}' has length {static_length}, but there are {n_model_points} model points."
        )
    return pt.specify_shape(tensor, (n_model_points,))


def build_pymc_model(
    inputs: ModelInputArrays,
    mechanism,
    *,
    likelihood_name=RATE_NORMAL,
    error_structure="material",
    likelihood_kwargs=None,
):
    coords = build_model_coords(inputs)
    point_inputs = build_model_point_inputs(inputs)
    n_model_points = len(point_inputs.E_V_SHE)

    with pm.Model(coords=coords) as model:
        result = mechanism(point_inputs)
        if not isinstance(result, MechanismResult):
            raise TypeError("Mechanism must return a MechanismResult.")
        validate_mechanism_result(result)

        ln_rate = _specify_model_point_vector(result.ln_rate, n_model_points, "ln_rate")
        ln_rate_model = pm.Deterministic("ln_rate_model", ln_rate, dims="model_point")

        registered_pointwise = {}
        for name, value in result.pointwise.items():
            if name in model.named_vars:
                raise ValueError(
                    f"Mechanism pointwise output '{name}' conflicts with an existing PyMC variable."
                )
            registered_pointwise[name] = pm.Deterministic(
                name,
                _specify_model_point_vector(value, n_model_points, name),
                dims="model_point",
            )

        registered_result = MechanismResult(
            ln_rate=ln_rate_model,
            pointwise=registered_pointwise,
        )

        likelihood = add_likelihood(
            likelihood_name,
            rate_model=pt.exp(ln_rate_model),
            inputs=inputs,
            error_structure=error_structure,
            likelihood_kwargs=likelihood_kwargs,
        )

    return BuiltModel(
        model=model,
        mechanism_result=registered_result,
        likelihood_name=likelihood_name,
        error_structure=error_structure,
        likelihood=likelihood,
    )
