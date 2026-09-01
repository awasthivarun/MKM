from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from mkm.model_inputs import ModelInputArrays


RATE_NORMAL = "rate_normal"
ERROR_STRUCTURES = ("shared", "material")


@dataclass(frozen=True)
class RateNormalLikelihood:
    error_structure: str
    sigma_abs: object
    sigma_rel: object
    mean_observation: object
    sigma_observation: object
    observed: object


def available_likelihoods():
    return (RATE_NORMAL,)


def available_error_structures():
    return ERROR_STRUCTURES


def validate_error_structure(error_structure: str):
    if error_structure not in ERROR_STRUCTURES:
        raise ValueError(
            f"Unknown error structure '{error_structure}'. "
            f"Available structures: {ERROR_STRUCTURES}."
        )
    return error_structure


def get_observation_material_index(inputs: ModelInputArrays):
    observation_model_point_index = np.asarray(inputs.observation_model_point_index, dtype=np.int64)
    model_point_condition_index = np.asarray(inputs.model_point_condition_index, dtype=np.int64)
    condition_material_index = np.asarray(inputs.condition_material_index, dtype=np.int64)

    observation_condition_index = model_point_condition_index[observation_model_point_index]
    observation_material_index = condition_material_index[observation_condition_index]
    return np.asarray(observation_material_index, dtype=np.int64)


def _positive_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _build_lognormal_error_parameter(name, median, log_sd, error_structure):
    kwargs = {"mu": np.log(median), "sigma": log_sd}
    if error_structure == "material":
        kwargs["dims"] = "material"
    return pm.LogNormal(name, **kwargs)


def add_rate_normal_likelihood(
    rate_model,
    inputs: ModelInputArrays,
    *,
    error_structure="material",
    sigma_abs_prior_median_s_inv=2.0e-4,
    sigma_abs_prior_log_sd=1.0,
    sigma_rel_prior_median=0.18,
    sigma_rel_prior_log_sd=0.75,
):
    error_structure = validate_error_structure(error_structure)
    sigma_abs_prior_median_s_inv = _positive_finite(
        sigma_abs_prior_median_s_inv, "sigma_abs_prior_median_s_inv"
    )
    sigma_abs_prior_log_sd = _positive_finite(
        sigma_abs_prior_log_sd, "sigma_abs_prior_log_sd"
    )
    sigma_rel_prior_median = _positive_finite(
        sigma_rel_prior_median, "sigma_rel_prior_median"
    )
    sigma_rel_prior_log_sd = _positive_finite(
        sigma_rel_prior_log_sd, "sigma_rel_prior_log_sd"
    )

    observation_model_point_index = np.asarray(
        inputs.observation_model_point_index, dtype=np.int64
    )
    observed_rate = np.asarray(inputs.observation_rate, dtype=float)

    if len(observation_model_point_index) != len(observed_rate):
        raise ValueError("Observation rates do not align with model-point indices.")
    if not np.all(np.isfinite(observed_rate)) or np.any(observed_rate <= 0):
        raise ValueError("Observed rates must be finite and positive.")

    rate_model = pt.as_tensor_variable(rate_model)
    if rate_model.ndim != 1:
        raise ValueError("rate_model must be one-dimensional.")

    mean_observation = rate_model[observation_model_point_index]

    sigma_abs = _build_lognormal_error_parameter(
        "sigma_rate_abs",
        sigma_abs_prior_median_s_inv,
        sigma_abs_prior_log_sd,
        error_structure,
    )
    sigma_rel = _build_lognormal_error_parameter(
        "sigma_rate_rel",
        sigma_rel_prior_median,
        sigma_rel_prior_log_sd,
        error_structure,
    )

    if error_structure == "material":
        observation_material_index = get_observation_material_index(inputs)
        sigma_abs_observation = sigma_abs[observation_material_index]
        sigma_rel_observation = sigma_rel[observation_material_index]
    else:
        sigma_abs_observation = sigma_abs
        sigma_rel_observation = sigma_rel

    sigma_observation = sigma_abs_observation + sigma_rel_observation * mean_observation

    observed = pm.Normal(
        "rate_observed",
        mu=mean_observation,
        sigma=sigma_observation,
        observed=observed_rate,
        dims="observation",
    )

    return RateNormalLikelihood(
        error_structure=error_structure,
        sigma_abs=sigma_abs,
        sigma_rel=sigma_rel,
        mean_observation=mean_observation,
        sigma_observation=sigma_observation,
        observed=observed,
    )


def add_likelihood(
    likelihood_name,
    *,
    rate_model,
    inputs: ModelInputArrays,
    error_structure="material",
    likelihood_kwargs=None,
):
    likelihood_kwargs = dict(likelihood_kwargs or {})

    if likelihood_name == RATE_NORMAL:
        return add_rate_normal_likelihood(
            rate_model=rate_model,
            inputs=inputs,
            error_structure=error_structure,
            **likelihood_kwargs,
        )

    raise ValueError(
        f"Unknown likelihood '{likelihood_name}'. "
        f"Available likelihoods: {available_likelihoods()}."
    )
