from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from mkm.model_inputs import ModelInputArrays


@dataclass(frozen=True)
class LogRateLikelihood:
    sigma_material: object
    mu_observation: object
    sigma_observation: object
    observed: object


def get_observation_material_index(inputs: ModelInputArrays):
    observation_model_point_index = inputs.observation_model_point_index
    model_point_condition_index = inputs.model_point_condition_index
    condition_material_index = inputs.condition_material_index

    observation_condition_index = model_point_condition_index[observation_model_point_index]
    observation_material_index = condition_material_index[observation_condition_index]

    return np.asarray(observation_material_index, dtype=np.int64)


def add_material_log_rate_likelihood(
    ln_rate_model, inputs: ModelInputArrays, sigma_prior_median=0.20, sigma_prior_log_sd=0.75
):
    sigma_prior_median = float(sigma_prior_median)
    sigma_prior_log_sd = float(sigma_prior_log_sd)

    if not np.isfinite(sigma_prior_median) or sigma_prior_median <= 0:
        raise ValueError("sigma_prior_median must be finite and positive.")

    if not np.isfinite(sigma_prior_log_sd) or sigma_prior_log_sd <= 0:
        raise ValueError("sigma_prior_log_sd must be finite and positive.")

    observation_material_index = get_observation_material_index(inputs)
    observation_model_point_index = np.asarray(inputs.observation_model_point_index, dtype=np.int64)
    observed_ln_rate = np.asarray(inputs.observation_ln_rate, dtype=float)

    ln_rate_model = pt.as_tensor_variable(ln_rate_model)

    sigma_material = pm.LogNormal(
        "sigma_ln_rate_material", mu=np.log(sigma_prior_median), sigma=sigma_prior_log_sd, dims="material"
    )

    mu_observation = pm.Deterministic(
        "ln_rate_observation_mean", ln_rate_model[observation_model_point_index], dims="observation"
    )

    sigma_observation = pm.Deterministic(
        "ln_rate_observation_sigma", sigma_material[observation_material_index], dims="observation"
    )

    observed = pm.Normal(
        "ln_rate_observed", mu=mu_observation, sigma=sigma_observation, observed=observed_ln_rate, dims="observation"
    )

    return LogRateLikelihood(
        sigma_material=sigma_material,
        mu_observation=mu_observation,
        sigma_observation=sigma_observation,
        observed=observed,
    )