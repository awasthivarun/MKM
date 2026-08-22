from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from mkm.model_inputs import ModelInputArrays


@dataclass(frozen=True)
class LogRateLikelihood:
    sigma_material: object
    setup_sigma_material: object | None
    setup_z: object | None
    setup_offset: object | None
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


def add_log_rate_likelihood(
    ln_rate_model,
    inputs: ModelInputArrays,
    sigma_prior_median=0.20,
    sigma_prior_log_sd=0.75,
    setup_intercept=False,
    setup_prior_median=0.10,
    setup_prior_log_sd=0.75,
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
        "sigma_ln_rate_material",
        mu=np.log(sigma_prior_median),
        sigma=sigma_prior_log_sd,
        dims="material",
    )

    mu_values = ln_rate_model[observation_model_point_index]

    setup_sigma_material = None
    setup_z = None
    setup_offset = None

    if setup_intercept:
        setup_prior_median = float(setup_prior_median)
        setup_prior_log_sd = float(setup_prior_log_sd)

        if not np.isfinite(setup_prior_median) or setup_prior_median <= 0:
            raise ValueError("setup_prior_median must be finite and positive.")

        if not np.isfinite(setup_prior_log_sd) or setup_prior_log_sd <= 0:
            raise ValueError("setup_prior_log_sd must be finite and positive.")

        if (
            inputs.setup_labels is None
            or inputs.setup_material_index is None
            or inputs.observation_setup_index is None
        ):
            raise ValueError("Setup-intercept likelihood requires setup-indexed model inputs.")

        setup_material_index = np.asarray(inputs.setup_material_index, dtype=np.int64)
        observation_setup_index = np.asarray(inputs.observation_setup_index, dtype=np.int64)

        if len(setup_material_index) != len(inputs.setup_labels):
            raise ValueError("Setup material index does not align with setup coordinates.")

        if len(observation_setup_index) != len(observed_ln_rate):
            raise ValueError("Observation setup index does not align with observations.")

        if np.any(setup_material_index < 0) or np.any(setup_material_index >= len(inputs.materials)):
            raise ValueError("Setup material indices are invalid.")

        if np.any(observation_setup_index < 0) or np.any(observation_setup_index >= len(inputs.setup_labels)):
            raise ValueError("Observation setup indices are invalid.")

        setup_sigma_material = pm.LogNormal(
            "sigma_ln_rate_setup_material",
            mu=np.log(setup_prior_median),
            sigma=setup_prior_log_sd,
            dims="material",
        )

        setup_z = pm.Normal(
            "z_ln_rate_setup",
            mu=0.0,
            sigma=1.0,
            dims="setup",
        )

        if inputs.setup_experiment_index is None or inputs.setup_experiment_size is None:
            raise ValueError("Zero-sum setup likelihood requires setup-experiment indexing.")

        setup_experiment_index = np.asarray(inputs.setup_experiment_index, dtype=np.int64)
        setup_experiment_size = np.asarray(inputs.setup_experiment_size, dtype=np.int64)

        n_setups = len(inputs.setup_labels)

        if len(setup_experiment_index) != n_setups:
            raise ValueError("Setup experiment index does not align with setup coordinates.")

        if np.any(setup_experiment_size < 2):
            raise ValueError("Every zero-sum setup experiment must contain at least two setups.")

        centering_matrix = np.eye(n_setups, dtype=float)

        for experiment_id, experiment_size in enumerate(setup_experiment_size):
            setup_indices = np.flatnonzero(setup_experiment_index == experiment_id)

            if len(setup_indices) != experiment_size:
                raise ValueError("Setup experiment size is inconsistent with setup indexing.")

            centering_matrix[np.ix_(setup_indices, setup_indices)] -= 1.0 / experiment_size

        setup_scale = np.sqrt(
            setup_experiment_size[setup_experiment_index]
            / (setup_experiment_size[setup_experiment_index] - 1.0)
        )

        centered_setup_z = pt.dot(
            pt.as_tensor_variable(centering_matrix),
            setup_z,
        )

        setup_offset = pm.Deterministic(
            "ln_rate_setup_offset",
            setup_sigma_material[setup_material_index] * centered_setup_z * setup_scale,
            dims="setup",
        )

        mu_values = mu_values + setup_offset[observation_setup_index]

    mu_observation = pm.Deterministic(
        "ln_rate_observation_mean",
        mu_values,
        dims="observation",
    )

    sigma_observation = pm.Deterministic(
        "ln_rate_observation_sigma",
        sigma_material[observation_material_index],
        dims="observation",
    )

    observed = pm.Normal(
        "ln_rate_observed",
        mu=mu_observation,
        sigma=sigma_observation,
        observed=observed_ln_rate,
        dims="observation",
    )

    return LogRateLikelihood(
        sigma_material=sigma_material,
        setup_sigma_material=setup_sigma_material,
        setup_z=setup_z,
        setup_offset=setup_offset,
        mu_observation=mu_observation,
        sigma_observation=sigma_observation,
        observed=observed,
    )


def add_material_log_rate_likelihood(*args, **kwargs):
    return add_log_rate_likelihood(*args, **kwargs)