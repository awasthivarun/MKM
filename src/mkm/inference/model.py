from dataclasses import dataclass

import pymc as pm
import pytensor.tensor as pt

from mkm.inference.likelihoods import (
    LogRateLikelihood,
    RateNormalLikelihood,
    add_log_rate_likelihood,
    add_rate_normal_likelihood,
)
from mkm.mechanisms.base import MechanismResult, validate_mechanism_result
from mkm.model_inputs import ModelInputArrays, build_model_coords, build_model_point_inputs


@dataclass(frozen=True)
class BuiltModel:
    model: pm.Model
    mechanism_result: MechanismResult
    likelihood: LogRateLikelihood | RateNormalLikelihood


def _specify_model_point_vector(value, n_model_points, name):
    tensor = pt.as_tensor_variable(value)

    if tensor.ndim != 1:
        raise ValueError(f"'{name}' must be one-dimensional.")

    static_length = tensor.type.shape[0]

    if static_length is not None and static_length != n_model_points:
        raise ValueError(f"'{name}' has length {static_length}, but there are {n_model_points} model points.")

    return pt.specify_shape(tensor, (n_model_points,))


def build_pymc_model(
    inputs: ModelInputArrays,
    mechanism,
    sigma_prior_median=0.20,
    sigma_prior_log_sd=0.75,
    setup_intercept=False,
    setup_prior_median=0.10,
    setup_prior_log_sd=0.75,
    correlated_potential=False,
    correlation_length_prior_median_V=0.020,
    correlation_length_prior_log_sd=1.0,
    rate_normal=False,
    sigma_abs_prior_median_s_inv=2.0e-4,
    sigma_abs_prior_log_sd=1.0,
    sigma_rel_prior_median=0.18,
    sigma_rel_prior_log_sd=0.75,
):
    coords = build_model_coords(inputs)
    point_inputs = build_model_point_inputs(inputs)
    n_model_points = len(point_inputs.E_V_SHE)

    with pm.Model(coords=coords) as model:
        result = mechanism(point_inputs)

        if not isinstance(result, MechanismResult):
            raise TypeError("Mechanism must return a MechanismResult.")

        validate_mechanism_result(result)

        ln_rate = _specify_model_point_vector(value=result.ln_rate, n_model_points=n_model_points, name="ln_rate")
        ln_rate_model = pm.Deterministic("ln_rate_model", ln_rate, dims="model_point")
        rate_model = pm.Deterministic("rate_model", pt.exp(ln_rate_model), dims="model_point")

        registered_pointwise = {}

        for name, value in result.pointwise.items():
            if name in model.named_vars:
                raise ValueError(f"Mechanism pointwise output '{name}' conflicts with an existing PyMC variable.")

            pointwise_value = _specify_model_point_vector(value=value, n_model_points=n_model_points, name=name)
            registered_pointwise[name] = pm.Deterministic(name, pointwise_value, dims="model_point")

        registered_result = MechanismResult(ln_rate=ln_rate_model, pointwise=registered_pointwise)

        if rate_normal:
            if setup_intercept or correlated_potential:
                raise ValueError(
                    "Rate-normal, setup-intercept, and potential-correlated likelihoods are separate "
                    "likelihood models and cannot be enabled together."
                )

            likelihood = add_rate_normal_likelihood(
                rate_model=rate_model,
                inputs=inputs,
                sigma_abs_prior_median_s_inv=sigma_abs_prior_median_s_inv,
                sigma_abs_prior_log_sd=sigma_abs_prior_log_sd,
                sigma_rel_prior_median=sigma_rel_prior_median,
                sigma_rel_prior_log_sd=sigma_rel_prior_log_sd,
            )
        else:
            likelihood = add_log_rate_likelihood(
                ln_rate_model=ln_rate_model,
                inputs=inputs,
                sigma_prior_median=sigma_prior_median,
                sigma_prior_log_sd=sigma_prior_log_sd,
                setup_intercept=setup_intercept,
                setup_prior_median=setup_prior_median,
                setup_prior_log_sd=setup_prior_log_sd,
                correlated_potential=correlated_potential,
                correlation_length_prior_median_V=correlation_length_prior_median_V,
                correlation_length_prior_log_sd=correlation_length_prior_log_sd,
            )

    return BuiltModel(model=model, mechanism_result=registered_result, likelihood=likelihood)