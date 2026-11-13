
"""Posterior summaries for kinetic-model points, shared across catalyst families."""

import numpy as np

from mkm.postprocessing.diagnostics import flatten_posterior_samples, summarize_samples


def summarize_model_points(inference_data, model_points, pointwise_variables=()):
    posterior = inference_data.posterior

    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior is missing 'ln_rate_model'.")

    result = model_points.copy()
    ln_rate = flatten_posterior_samples(posterior["ln_rate_model"])

    if ln_rate.ndim != 2 or ln_rate.shape[1] != len(result):
        raise ValueError("Posterior log-rates do not align with model points.")

    variables = (
        ("ln_rate_model", ln_rate),
        ("rate_model", np.exp(ln_rate)),
    )
    for name, values in variables:
        for statistic, data in summarize_samples(values).items():
            result[f"{name}_{statistic}"] = data

    for name in dict.fromkeys(pointwise_variables):
        if name not in posterior:
            continue

        values = flatten_posterior_samples(posterior[name])
        if values.ndim != 2 or values.shape[1] != len(result):
            raise ValueError(f"Posterior variable '{name}' does not align with model points.")

        for statistic, data in summarize_samples(values).items():
            result[f"{name}_{statistic}"] = data

    return result
