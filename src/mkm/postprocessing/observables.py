from dataclasses import dataclass

import numpy as np
import pandas as pd

from mkm.observable_maps import evaluate_linear_observable_map_draws
from mkm.postprocessing.diagnostics import summarize_samples


@dataclass(frozen=True)
class PosteriorObservableSummary:
    pooled: pd.DataFrame
    by_chain: pd.DataFrame


def _get_posterior(inference_data):
    if hasattr(inference_data, "posterior"):
        return inference_data.posterior

    try:
        return inference_data["posterior"]
    except Exception as error:
        raise ValueError("Inference data does not contain a posterior group.") from error


def summarize_pointwise_posterior_variable(inference_data, model_points, variable_name):
    posterior = _get_posterior(inference_data)

    if variable_name not in posterior:
        raise ValueError(f"Posterior does not contain '{variable_name}'.")

    values = np.asarray(posterior[variable_name], dtype=float)

    if values.ndim != 3:
        raise ValueError(
            f"Posterior variable '{variable_name}' must have shape (chain, draw, model_point)."
        )

    if values.shape[-1] != len(model_points):
        raise ValueError(f"Posterior variable '{variable_name}' does not align with model points.")

    flattened = values.reshape((-1, values.shape[-1]))
    summary = summarize_samples(flattened)

    result = model_points.copy()

    for statistic, statistic_values in summary.items():
        result[statistic] = statistic_values

    return result


def summarize_posterior_model_variable(inference_data, model_points, variable_name):
    result = summarize_pointwise_posterior_variable(
        inference_data=inference_data,
        model_points=model_points,
        variable_name=variable_name,
    )

    statistics = (
        "mean", "sd", "median", "hdi80_lower", "hdi80_upper", "hdi95_lower", "hdi95_upper"
    )

    return result.rename(
        columns={statistic: f"{variable_name}_{statistic}" for statistic in statistics}
    )


def summarize_posterior_linear_observable(inference_data, observable_map):
    posterior = _get_posterior(inference_data)

    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior does not contain 'ln_rate_model'.")

    ln_rate = np.asarray(posterior["ln_rate_model"], dtype=float)
    observable_draws = evaluate_linear_observable_map_draws(ln_rate, observable_map)

    if observable_draws.ndim != 3:
        raise ValueError("Expected observable draws with shape (chain, draw, observable).")

    pooled_values = observable_draws.reshape((-1, observable_draws.shape[-1]))
    pooled_summary = summarize_samples(pooled_values)

    pooled = observable_map.outputs.copy()

    for statistic, values in pooled_summary.items():
        pooled[statistic] = values

    chain_records = []

    for chain in range(observable_draws.shape[0]):
        summary = summarize_samples(observable_draws[chain])
        frame = observable_map.outputs.copy()
        frame.insert(0, "chain", chain)

        for statistic, values in summary.items():
            frame[statistic] = values

        chain_records.append(frame)

    by_chain = pd.concat(chain_records, ignore_index=True)

    return PosteriorObservableSummary(pooled=pooled, by_chain=by_chain)


def summarize_pointwise_pathway_fractions(inference_data, model_points):
    posterior = _get_posterior(inference_data)
    results = {}

    for name in ["rate_fraction_BF", "rate_fraction_ER", "rate_fraction_LH"]:
        if name not in posterior:
            continue

        results[name] = summarize_posterior_model_variable(
            inference_data=inference_data,
            model_points=model_points,
            variable_name=name,
        )

    return results
