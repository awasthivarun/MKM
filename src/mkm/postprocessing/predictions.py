from dataclasses import dataclass

import numpy as np

from mkm.inference.likelihoods import get_observation_material_index
from mkm.postprocessing.diagnostics import summarize_samples


@dataclass(frozen=True)
class ObservationDistributionDraws:
    mechanism_mu: np.ndarray
    conditional_mu: np.ndarray
    sigma: np.ndarray


def build_observation_distribution_draws(inference_data, inputs, likelihood_name):
    posterior = inference_data.posterior

    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior does not contain 'ln_rate_model'.")

    ln_rate_model = np.asarray(posterior["ln_rate_model"], dtype=float)
    observation_model_point_index = np.asarray(inputs.observation_model_point_index, dtype=np.int64)
    mechanism_mu = ln_rate_model[..., observation_model_point_index]

    if likelihood_name == "setup_intercept":
        if "ln_rate_setup_offset" not in posterior:
            raise ValueError("Setup-intercept posterior is missing 'ln_rate_setup_offset'.")
        if inputs.observation_setup_index is None:
            raise ValueError("Observation setup index is missing.")

        setup_offset = np.asarray(posterior["ln_rate_setup_offset"], dtype=float)
        observation_setup_index = np.asarray(inputs.observation_setup_index, dtype=np.int64)
        conditional_mu = mechanism_mu + setup_offset[..., observation_setup_index]
    elif likelihood_name == "iid":
        conditional_mu = mechanism_mu
    else:
        raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")

    if "sigma_ln_rate_material" not in posterior:
        raise ValueError("Posterior does not contain 'sigma_ln_rate_material'.")

    sigma_material = np.asarray(posterior["sigma_ln_rate_material"], dtype=float)
    observation_material_index = np.asarray(get_observation_material_index(inputs), dtype=np.int64)
    sigma = sigma_material[..., observation_material_index]

    if mechanism_mu.shape != conditional_mu.shape or conditional_mu.shape != sigma.shape:
        raise ValueError(
            "Posterior observation-distribution arrays do not align: "
            f"mechanism={mechanism_mu.shape}, conditional={conditional_mu.shape}, sigma={sigma.shape}."
        )

    return ObservationDistributionDraws(
        mechanism_mu=mechanism_mu,
        conditional_mu=conditional_mu,
        sigma=sigma,
    )


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
    draws = build_observation_distribution_draws(
        inference_data=inference_data,
        inputs=inputs,
        likelihood_name=likelihood_name,
    )

    n_observations = draws.conditional_mu.shape[-1]
    mechanism_mu = draws.mechanism_mu.reshape(-1, n_observations)
    conditional_mu = draws.conditional_mu.reshape(-1, n_observations)
    sigma_observation = draws.sigma.reshape(-1, n_observations)

    rng = np.random.default_rng(random_seed)
    posterior_predictive = conditional_mu + sigma_observation * rng.standard_normal(conditional_mu.shape)

    mechanism_rate = np.exp(mechanism_mu)
    conditional_rate = np.exp(conditional_mu)
    predictive_rate = np.exp(posterior_predictive)

    if not all(
        np.all(np.isfinite(values))
        for values in (mechanism_rate, conditional_rate, predictive_rate)
    ):
        raise RuntimeError("Rate-space posterior draws contain non-finite values after exponentiation.")

    observations = model_data.observations.copy()

    if len(observations) != n_observations:
        raise ValueError(
            f"Posterior contains {n_observations} observations, but model data contain {len(observations)}."
        )

    observations["rate"] = np.exp(observations["ln_rate"].to_numpy(dtype=float))

    _add_distribution_summary(observations, "ln_rate_mechanism", mechanism_mu)
    _add_distribution_summary(observations, "ln_rate_conditional", conditional_mu)
    _add_distribution_summary(observations, "ln_rate_predictive", posterior_predictive)
    _add_distribution_summary(observations, "rate_mechanism", mechanism_rate)
    _add_distribution_summary(observations, "rate_conditional", conditional_rate)
    _add_distribution_summary(observations, "rate_predictive", predictive_rate)
    _add_distribution_summary(observations, "sigma_ln_rate", sigma_observation)

    observations["residual_mechanism"] = (
        observations["ln_rate"] - observations["ln_rate_mechanism_median"]
    )
    observations["residual_conditional"] = (
        observations["ln_rate"] - observations["ln_rate_conditional_median"]
    )
    observations["standardized_residual_conditional"] = (
        observations["residual_conditional"] / observations["sigma_ln_rate_median"]
    )

    observations["observed_inside_predictive_95_hdi"] = (
        (observations["ln_rate"] >= observations["ln_rate_predictive_hdi95_lower"])
        & (observations["ln_rate"] <= observations["ln_rate_predictive_hdi95_upper"])
    )

    return observations
