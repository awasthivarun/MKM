from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from mkm.model_inputs import ModelInputArrays


@dataclass(frozen=True)
class PotentialCurveStructure:
    curve_index: np.ndarray
    previous_observation_index: np.ndarray
    delta_E_V: np.ndarray
    is_transition: np.ndarray
    ordered_observation_index: np.ndarray
    n_curves: int


@dataclass(frozen=True)
class LogRateLikelihood:
    sigma_material: object
    correlation_length_material: object | None
    rho_observation: object | None
    setup_sigma_material: object | None
    setup_z: object | None
    setup_offset: object | None
    mu_observation: object
    sigma_observation: object
    observed: object


@dataclass(frozen=True)
class RateNormalLikelihood:
    sigma_abs: object
    sigma_rel: object
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


def get_observation_curve_structure(inputs: ModelInputArrays):
    observation_model_point_index = np.asarray(inputs.observation_model_point_index, dtype=np.int64)
    model_point_condition_index = np.asarray(inputs.model_point_condition_index, dtype=np.int64)
    model_point_E_V_SHE = np.asarray(inputs.model_point_E_V_SHE, dtype=float)
    observation_replicate = np.asarray(inputs.observation_replicate).astype(str)

    n_observations = len(observation_model_point_index)
    if len(observation_replicate) != n_observations:
        raise ValueError("Observation replicate labels do not align with observations.")

    observation_condition_index = model_point_condition_index[observation_model_point_index]
    observation_E_V_SHE = model_point_E_V_SHE[observation_model_point_index]

    if not np.all(np.isfinite(observation_E_V_SHE)):
        raise ValueError("Observation potentials contain non-finite values.")

    keys = sorted(
        set(zip(observation_condition_index.tolist(), observation_replicate.tolist())),
        key=lambda value: (value[0], value[1]),
    )
    key_to_curve_index = {key: index for index, key in enumerate(keys)}

    curve_index = np.empty(n_observations, dtype=np.int64)
    previous_observation_index = np.arange(n_observations, dtype=np.int64)
    delta_E_V = np.zeros(n_observations, dtype=float)
    is_transition = np.zeros(n_observations, dtype=bool)
    ordered_observation_index = []

    for key in keys:
        condition_index, replicate = key
        indices = np.flatnonzero(
            (observation_condition_index == condition_index)
            & (observation_replicate == replicate)
        )
        order = indices[np.argsort(observation_E_V_SHE[indices], kind="stable")]
        potentials = observation_E_V_SHE[order]

        if len(order) > 1:
            differences = np.diff(potentials)
            if np.any(differences <= 0):
                raise ValueError(
                    "Potential-correlated likelihood requires strictly increasing potentials "
                    "within every condition/replicate curve."
                )

            previous_observation_index[order[1:]] = order[:-1]
            delta_E_V[order[1:]] = differences
            is_transition[order[1:]] = True

        curve_index[order] = key_to_curve_index[key]
        ordered_observation_index.extend(order.tolist())

    return PotentialCurveStructure(
        curve_index=curve_index,
        previous_observation_index=previous_observation_index,
        delta_E_V=delta_E_V,
        is_transition=is_transition,
        ordered_observation_index=np.asarray(ordered_observation_index, dtype=np.int64),
        n_curves=len(keys),
    )


def add_log_rate_likelihood(
    ln_rate_model,
    inputs: ModelInputArrays,
    sigma_prior_median=0.20,
    sigma_prior_log_sd=0.75,
    setup_intercept=False,
    setup_prior_median=0.10,
    setup_prior_log_sd=0.75,
    correlated_potential=False,
    correlation_length_prior_median_V=0.020,
    correlation_length_prior_log_sd=1.0,
):
    sigma_prior_median = float(sigma_prior_median)
    sigma_prior_log_sd = float(sigma_prior_log_sd)

    if not np.isfinite(sigma_prior_median) or sigma_prior_median <= 0:
        raise ValueError("sigma_prior_median must be finite and positive.")

    if not np.isfinite(sigma_prior_log_sd) or sigma_prior_log_sd <= 0:
        raise ValueError("sigma_prior_log_sd must be finite and positive.")

    if setup_intercept and correlated_potential:
        raise ValueError(
            "Setup-intercept and potential-correlated likelihoods are separate likelihood models "
            "and cannot be enabled together."
        )

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

    correlation_length_material = None
    rho_observation = None
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

    sigma_values = sigma_material[observation_material_index]

    if correlated_potential:
        correlation_length_prior_median_V = float(correlation_length_prior_median_V)
        correlation_length_prior_log_sd = float(correlation_length_prior_log_sd)

        if not np.isfinite(correlation_length_prior_median_V) or correlation_length_prior_median_V <= 0:
            raise ValueError("correlation_length_prior_median_V must be finite and positive.")

        if not np.isfinite(correlation_length_prior_log_sd) or correlation_length_prior_log_sd <= 0:
            raise ValueError("correlation_length_prior_log_sd must be finite and positive.")

        curve_structure = get_observation_curve_structure(inputs)
        previous_index = curve_structure.previous_observation_index
        delta_E_V = pt.as_tensor_variable(curve_structure.delta_E_V)
        transition = pt.as_tensor_variable(curve_structure.is_transition.astype(float))

        correlation_length_material = pm.LogNormal(
            "ell_E_V_material",
            mu=np.log(correlation_length_prior_median_V),
            sigma=correlation_length_prior_log_sd,
            dims="material",
        )
        correlation_length_observation = correlation_length_material[observation_material_index]

        rho_values = transition * pt.exp(-delta_E_V / correlation_length_observation)
        rho_observation = pm.Deterministic(
            "ln_rate_observation_rho",
            rho_values,
            dims="observation",
        )

        previous_residual = observed_ln_rate[previous_index] - mu_values[previous_index]
        likelihood_mu_values = mu_values + rho_values * previous_residual

        transition_variance_fraction = -pt.expm1(
            -2.0 * delta_E_V / correlation_length_observation
        )
        variance_fraction = (1.0 - transition) + transition * transition_variance_fraction
        likelihood_sigma_values = sigma_values * pt.sqrt(variance_fraction)
    else:
        likelihood_mu_values = mu_values
        likelihood_sigma_values = sigma_values

    mu_observation = pm.Deterministic(
        "ln_rate_observation_mean",
        likelihood_mu_values,
        dims="observation",
    )

    sigma_observation = pm.Deterministic(
        "ln_rate_observation_sigma",
        likelihood_sigma_values,
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
        correlation_length_material=correlation_length_material,
        rho_observation=rho_observation,
        setup_sigma_material=setup_sigma_material,
        setup_z=setup_z,
        setup_offset=setup_offset,
        mu_observation=mu_observation,
        sigma_observation=sigma_observation,
        observed=observed,
    )


def add_rate_normal_likelihood(
    rate_model,
    inputs: ModelInputArrays,
    sigma_abs_prior_median_s_inv=2.0e-4,
    sigma_abs_prior_log_sd=1.0,
    sigma_rel_prior_median=0.18,
    sigma_rel_prior_log_sd=0.75,
):
    sigma_abs_prior_median_s_inv = float(sigma_abs_prior_median_s_inv)
    sigma_abs_prior_log_sd = float(sigma_abs_prior_log_sd)
    sigma_rel_prior_median = float(sigma_rel_prior_median)
    sigma_rel_prior_log_sd = float(sigma_rel_prior_log_sd)

    for name, value in (
        ("sigma_abs_prior_median_s_inv", sigma_abs_prior_median_s_inv),
        ("sigma_abs_prior_log_sd", sigma_abs_prior_log_sd),
        ("sigma_rel_prior_median", sigma_rel_prior_median),
        ("sigma_rel_prior_log_sd", sigma_rel_prior_log_sd),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    observation_model_point_index = np.asarray(
        inputs.observation_model_point_index,
        dtype=np.int64,
    )
    observed_rate = np.asarray(inputs.observation_rate, dtype=float)

    if len(observation_model_point_index) != len(observed_rate):
        raise ValueError("Observation rate values do not align with model-point indices.")

    if not np.all(np.isfinite(observed_rate)) or np.any(observed_rate <= 0):
        raise ValueError("Observed rates must be finite and positive.")

    rate_model = pt.as_tensor_variable(rate_model)
    mu_values = rate_model[observation_model_point_index]

    sigma_abs = pm.LogNormal(
        "sigma_rate_abs",
        mu=np.log(sigma_abs_prior_median_s_inv),
        sigma=sigma_abs_prior_log_sd,
    )
    sigma_rel = pm.LogNormal(
        "sigma_rate_rel",
        mu=np.log(sigma_rel_prior_median),
        sigma=sigma_rel_prior_log_sd,
    )

    sigma_values = sigma_abs + sigma_rel * mu_values

    mu_observation = pm.Deterministic(
        "rate_observation_mean",
        mu_values,
        dims="observation",
    )
    sigma_observation = pm.Deterministic(
        "rate_observation_sigma",
        sigma_values,
        dims="observation",
    )

    observed = pm.Normal(
        "rate_observed",
        mu=mu_observation,
        sigma=sigma_observation,
        observed=observed_rate,
        dims="observation",
    )

    return RateNormalLikelihood(
        sigma_abs=sigma_abs,
        sigma_rel=sigma_rel,
        mu_observation=mu_observation,
        sigma_observation=sigma_observation,
        observed=observed,
    )


add_material_log_rate_likelihood = add_log_rate_likelihood
