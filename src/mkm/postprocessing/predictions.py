import numpy as np

from mkm.inference.likelihoods import get_observation_material_index
from mkm.postprocessing.diagnostics import (
    flatten_posterior_samples,
    summarize_samples,
)


def build_observation_diagnostics(
    inference_data,
    model_data,
    inputs,
    likelihood_name,
    random_seed=20260821,
):
    posterior = inference_data.posterior

    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior does not contain 'ln_rate_model'.")

    ln_rate_model = flatten_posterior_samples(
        posterior["ln_rate_model"]
    )

    observation_model_point_index = np.asarray(
        inputs.observation_model_point_index,
        dtype=np.int64,
    )

    mechanism_mu = ln_rate_model[:, observation_model_point_index]

    if likelihood_name == "setup_intercept":
        if "ln_rate_setup_offset" not in posterior:
            raise ValueError(
                "Setup-intercept posterior is missing "
                "'ln_rate_setup_offset'."
            )

        if inputs.observation_setup_index is None:
            raise ValueError("Observation setup index is missing.")

        setup_offset = flatten_posterior_samples(
            posterior["ln_rate_setup_offset"]
        )

        observation_setup_index = np.asarray(
            inputs.observation_setup_index,
            dtype=np.int64,
        )

        conditional_mu = (
            mechanism_mu
            + setup_offset[:, observation_setup_index]
        )

    elif likelihood_name == "iid":
        conditional_mu = mechanism_mu

    else:
        raise ValueError(
            f"Unsupported likelihood '{likelihood_name}'."
        )

    if "sigma_ln_rate_material" not in posterior:
        raise ValueError(
            "Posterior does not contain 'sigma_ln_rate_material'."
        )

    sigma_material = flatten_posterior_samples(
        posterior["sigma_ln_rate_material"]
    )

    observation_material_index = get_observation_material_index(
        inputs
    )

    sigma_observation = sigma_material[
        :, observation_material_index
    ]

    rng = np.random.default_rng(random_seed)

    posterior_predictive = (
        conditional_mu
        + sigma_observation
        * rng.standard_normal(conditional_mu.shape)
    )

    mechanism_summary = summarize_samples(mechanism_mu)
    conditional_summary = summarize_samples(conditional_mu)
    predictive_summary = summarize_samples(posterior_predictive)
    sigma_summary = summarize_samples(sigma_observation)

    observations = model_data.observations.copy()

    if len(observations) != mechanism_mu.shape[1]:
        raise ValueError(
            "Posterior observation mapping is inconsistent."
        )

    observations["ln_rate_mechanism_q025"] = (
        mechanism_summary["q025"]
    )
    observations["ln_rate_mechanism_q50"] = (
        mechanism_summary["q50"]
    )
    observations["ln_rate_mechanism_q975"] = (
        mechanism_summary["q975"]
    )

    observations["ln_rate_conditional_q025"] = (
        conditional_summary["q025"]
    )
    observations["ln_rate_conditional_q50"] = (
        conditional_summary["q50"]
    )
    observations["ln_rate_conditional_q975"] = (
        conditional_summary["q975"]
    )

    observations["ln_rate_predictive_q025"] = (
        predictive_summary["q025"]
    )
    observations["ln_rate_predictive_q50"] = (
        predictive_summary["q50"]
    )
    observations["ln_rate_predictive_q975"] = (
        predictive_summary["q975"]
    )

    observations["sigma_ln_rate_q50"] = sigma_summary["q50"]

    observations["residual_mechanism"] = (
        observations["ln_rate"]
        - observations["ln_rate_mechanism_q50"]
    )

    observations["residual_conditional"] = (
        observations["ln_rate"]
        - observations["ln_rate_conditional_q50"]
    )

    observations["standardized_residual_conditional"] = (
        observations["residual_conditional"]
        / observations["sigma_ln_rate_q50"]
    )

    observations["observed_inside_predictive_95"] = (
        (
            observations["ln_rate"]
            >= observations["ln_rate_predictive_q025"]
        )
        & (
            observations["ln_rate"]
            <= observations["ln_rate_predictive_q975"]
        )
    )

    return observations