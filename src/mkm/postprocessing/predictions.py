from dataclasses import dataclass

import numpy as np

from mkm.inference.likelihoods import get_observation_curve_structure, get_observation_material_index
from mkm.postprocessing.diagnostics import summarize_samples


SUPPORTED_LIKELIHOODS = {
    "iid",
    "setup_intercept",
    "mvn",
    "rate_normal",
}


@dataclass(frozen=True)
class ObservationDistributionDraws:
    mechanism_mu: np.ndarray
    conditional_mu: np.ndarray
    sigma: np.ndarray
    conditional_sigma: np.ndarray
    correlation_length: np.ndarray | None


def build_observation_distribution_draws(
    inference_data,
    inputs,
    likelihood_name,
    observed_ln_rate=None,
):
    if likelihood_name not in SUPPORTED_LIKELIHOODS:
        raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")

    posterior = inference_data.posterior

    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior does not contain 'ln_rate_model'.")

    ln_rate_model = np.asarray(posterior["ln_rate_model"], dtype=float)
    observation_model_point_index = np.asarray(inputs.observation_model_point_index, dtype=np.int64)
    mechanism_mu = ln_rate_model[..., observation_model_point_index]

    correlation_length = None

    if likelihood_name == "rate_normal":
        if "sigma_rate_abs" not in posterior or "sigma_rate_rel" not in posterior:
            raise ValueError(
                "Rate-normal posterior must contain 'sigma_rate_abs' and 'sigma_rate_rel'."
            )

        mechanism_mu = np.exp(mechanism_mu)
        sigma_abs = np.asarray(posterior["sigma_rate_abs"], dtype=float)[..., None]
        sigma_rel = np.asarray(posterior["sigma_rate_rel"], dtype=float)[..., None]
        sigma = sigma_abs + sigma_rel * mechanism_mu
        conditional_mu = mechanism_mu
        conditional_sigma = sigma

    else:
        if "sigma_ln_rate_material" not in posterior:
            raise ValueError("Posterior does not contain 'sigma_ln_rate_material'.")

        sigma_material = np.asarray(posterior["sigma_ln_rate_material"], dtype=float)
        observation_material_index = np.asarray(get_observation_material_index(inputs), dtype=np.int64)
        sigma = sigma_material[..., observation_material_index]

    if likelihood_name == "setup_intercept":
        if "ln_rate_setup_offset" not in posterior:
            raise ValueError("Setup-intercept posterior is missing 'ln_rate_setup_offset'.")
        if inputs.observation_setup_index is None:
            raise ValueError("Observation setup index is missing.")

        setup_offset = np.asarray(posterior["ln_rate_setup_offset"], dtype=float)
        observation_setup_index = np.asarray(inputs.observation_setup_index, dtype=np.int64)
        conditional_mu = mechanism_mu + setup_offset[..., observation_setup_index]
        conditional_sigma = sigma

    elif likelihood_name == "iid":
        conditional_mu = mechanism_mu
        conditional_sigma = sigma

    elif likelihood_name == "rate_normal":
        pass

    elif likelihood_name == "mvn":
        if "ell_E_V_material" not in posterior:
            raise ValueError("MVN posterior does not contain 'ell_E_V_material'.")

        ell_material = np.asarray(posterior["ell_E_V_material"], dtype=float)
        correlation_length = ell_material[..., observation_material_index]

        structure = get_observation_curve_structure(inputs)
        previous_index = structure.previous_observation_index
        if observed_ln_rate is None:
            if not hasattr(inputs, "observation_ln_rate"):
                raise ValueError("MVN observation diagnostics require observed log rates.")
            observed_ln_rate = inputs.observation_ln_rate

        observed_ln_rate = np.asarray(observed_ln_rate, dtype=float)
        if observed_ln_rate.shape != observation_model_point_index.shape:
            raise ValueError("Observed log rates do not align with observation indices.")

        delta_E_V = np.asarray(structure.delta_E_V, dtype=float)
        transition = np.asarray(structure.is_transition, dtype=float)

        rho = transition * np.exp(-delta_E_V / correlation_length)

        previous_residual = (
            observed_ln_rate[previous_index]
            - mechanism_mu[..., previous_index]
        )
        conditional_mu = mechanism_mu + rho * previous_residual

        transition_variance_fraction = -np.expm1(
            -2.0 * delta_E_V / correlation_length
        )
        variance_fraction = (
            (1.0 - transition)
            + transition * transition_variance_fraction
        )
        conditional_sigma = sigma * np.sqrt(variance_fraction)

    else:
        raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")

    if (
        mechanism_mu.shape != conditional_mu.shape
        or conditional_mu.shape != sigma.shape
        or conditional_sigma.shape != sigma.shape
    ):
        raise ValueError(
            "Posterior observation-distribution arrays do not align: "
            f"mechanism={mechanism_mu.shape}, conditional={conditional_mu.shape}, "
            f"sigma={sigma.shape}, conditional_sigma={conditional_sigma.shape}."
        )

    if correlation_length is not None and correlation_length.shape != conditional_mu.shape:
        raise ValueError(
            "Posterior MVN correlation-length array does not align with observation means: "
            f"ell={correlation_length.shape}, conditional={conditional_mu.shape}."
        )

    return ObservationDistributionDraws(
        mechanism_mu=mechanism_mu,
        conditional_mu=conditional_mu,
        sigma=sigma,
        conditional_sigma=conditional_sigma,
        correlation_length=correlation_length,
    )


def draw_mvn_posterior_predictive(conditional_mu, sigma, correlation_length, inputs, rng):
    conditional_mu = np.asarray(conditional_mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    correlation_length = np.asarray(correlation_length, dtype=float)

    if conditional_mu.shape != sigma.shape or sigma.shape != correlation_length.shape:
        raise ValueError("MVN posterior-predictive arrays must have identical shapes.")
    if conditional_mu.ndim != 2:
        raise ValueError("MVN posterior-predictive arrays must be sample x observation.")
    if np.any(sigma <= 0) or np.any(correlation_length <= 0):
        raise ValueError("MVN posterior-predictive scales and correlation lengths must be positive.")

    structure = get_observation_curve_structure(inputs)
    n_observations = conditional_mu.shape[1]
    if len(structure.previous_observation_index) != n_observations:
        raise ValueError("MVN curve structure does not align with posterior observations.")

    innovations = rng.standard_normal(conditional_mu.shape)
    residual = np.zeros_like(conditional_mu)

    for observation_index in structure.ordered_observation_index:
        if not structure.is_transition[observation_index]:
            residual[:, observation_index] = (
                sigma[:, observation_index] * innovations[:, observation_index]
            )
            continue

        previous_index = structure.previous_observation_index[observation_index]
        delta_E_V = structure.delta_E_V[observation_index]
        ell = correlation_length[:, observation_index]
        rho = np.exp(-delta_E_V / ell)
        innovation_scale = sigma[:, observation_index] * np.sqrt(
            -np.expm1(-2.0 * delta_E_V / ell)
        )
        residual[:, observation_index] = (
            rho * residual[:, previous_index]
            + innovation_scale * innovations[:, observation_index]
        )

    return conditional_mu + residual


def _add_distribution_summary(observations, prefix, draws):
    summary = summarize_samples(draws)

    for statistic, values in summary.items():
        observations[f"{prefix}_{statistic}"] = values


def build_observation_diagnostics(
    inference_data,
    model_data,
    inputs,
    likelihood_name,
    random_seed=20260821,
):
    observed_ln_rate = None
    if "ln_rate" in model_data.observations.columns:
        observed_ln_rate = model_data.observations["ln_rate"].to_numpy(dtype=float)

    draws = build_observation_distribution_draws(
        inference_data=inference_data,
        inputs=inputs,
        likelihood_name=likelihood_name,
        observed_ln_rate=observed_ln_rate,
    )

    n_observations = draws.conditional_mu.shape[-1]
    mechanism_mu = draws.mechanism_mu.reshape(-1, n_observations)
    conditional_mu = draws.conditional_mu.reshape(-1, n_observations)
    sigma_observation = draws.sigma.reshape(-1, n_observations)
    conditional_sigma = draws.conditional_sigma.reshape(-1, n_observations)

    rng = np.random.default_rng(random_seed)
    if likelihood_name == "mvn":
        correlation_length = draws.correlation_length.reshape(-1, n_observations)
        posterior_predictive = draw_mvn_posterior_predictive(
            conditional_mu=mechanism_mu,
            sigma=sigma_observation,
            correlation_length=correlation_length,
            inputs=inputs,
            rng=rng,
        )
    else:
        posterior_predictive = conditional_mu + conditional_sigma * rng.standard_normal(conditional_mu.shape)

    observations = model_data.observations.copy()

    if len(observations) != n_observations:
        raise ValueError(
            f"Posterior contains {n_observations} observations, but model data contain {len(observations)}."
        )

    if likelihood_name == "rate_normal":
        mechanism_rate = mechanism_mu
        conditional_rate = conditional_mu
        predictive_rate = posterior_predictive

        if not all(
            np.all(np.isfinite(values))
            for values in (mechanism_rate, conditional_rate, predictive_rate)
        ):
            raise RuntimeError("Rate-space posterior draws contain non-finite values.")

        observations["rate"] = observations["rate_s_inv"].to_numpy(dtype=float)

        _add_distribution_summary(observations, "rate_mechanism", mechanism_rate)
        _add_distribution_summary(observations, "rate_conditional", conditional_rate)
        _add_distribution_summary(observations, "rate_predictive", predictive_rate)
        _add_distribution_summary(observations, "sigma_rate", sigma_observation)

        # Mechanistic log-rate summaries remain available for kinetic observables and comparison.
        ln_rate_mechanism = np.log(mechanism_rate)
        _add_distribution_summary(observations, "ln_rate_mechanism", ln_rate_mechanism)
        _add_distribution_summary(observations, "ln_rate_conditional", ln_rate_mechanism)

        observations["residual_mechanism"] = (
            observations["rate"] - observations["rate_mechanism_median"]
        )
        observations["residual_conditional"] = (
            observations["rate"] - observations["rate_conditional_median"]
        )
        observations["standardized_residual_conditional"] = (
            observations["residual_conditional"]
            / observations["sigma_rate_median"]
        )
        observations["observed_inside_predictive_95_hdi"] = (
            (observations["rate"] >= observations["rate_predictive_hdi95_lower"])
            & (observations["rate"] <= observations["rate_predictive_hdi95_upper"])
        )
        return observations

    mechanism_rate = np.exp(mechanism_mu)
    conditional_rate = np.exp(conditional_mu)
    predictive_rate = np.exp(posterior_predictive)

    if not all(
        np.all(np.isfinite(values))
        for values in (mechanism_rate, conditional_rate, predictive_rate)
    ):
        raise RuntimeError("Rate-space posterior draws contain non-finite values after exponentiation.")

    observations["rate"] = np.exp(observations["ln_rate"].to_numpy(dtype=float))

    _add_distribution_summary(observations, "ln_rate_mechanism", mechanism_mu)
    _add_distribution_summary(observations, "ln_rate_conditional", conditional_mu)
    _add_distribution_summary(observations, "ln_rate_predictive", posterior_predictive)
    _add_distribution_summary(observations, "rate_mechanism", mechanism_rate)
    _add_distribution_summary(observations, "rate_conditional", conditional_rate)
    _add_distribution_summary(observations, "rate_predictive", predictive_rate)
    _add_distribution_summary(observations, "sigma_ln_rate", sigma_observation)
    _add_distribution_summary(
        observations,
        "sigma_ln_rate_conditional",
        conditional_sigma,
    )

    observations["residual_mechanism"] = (
        observations["ln_rate"] - observations["ln_rate_mechanism_median"]
    )
    observations["residual_conditional"] = (
        observations["ln_rate"] - observations["ln_rate_conditional_median"]
    )
    observations["standardized_residual_conditional"] = (
        observations["residual_conditional"]
        / observations["sigma_ln_rate_conditional_median"]
    )

    observations["observed_inside_predictive_95_hdi"] = (
        (observations["ln_rate"] >= observations["ln_rate_predictive_hdi95_lower"])
        & (observations["ln_rate"] <= observations["ln_rate_predictive_hdi95_upper"])
    )

    return observations
