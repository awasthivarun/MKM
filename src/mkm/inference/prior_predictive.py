from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm


@dataclass(frozen=True)
class PriorPredictiveSummary:
    parameter_summary: pd.DataFrame
    model_point_summary: pd.DataFrame
    observation_summary: pd.DataFrame


def _get_group(prior_predictive, group_name):
    if hasattr(prior_predictive, group_name):
        return getattr(prior_predictive, group_name)

    try:
        return prior_predictive[group_name]
    except Exception as error:
        raise ValueError(f"Prior-predictive result does not contain group '{group_name}'.") from error


def _flatten_draw_dimensions(values):
    values = np.asarray(values)

    if values.ndim < 2:
        raise ValueError("Expected at least chain and draw dimensions.")

    return values.reshape((-1, *values.shape[2:]))


def _summarize_draws(values):
    values = _flatten_draw_dimensions(values)

    return {
        "mean": np.mean(values, axis=0),
        "sd": np.std(values, axis=0, ddof=1),
        "q025": np.quantile(values, 0.025, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q975": np.quantile(values, 0.975, axis=0),
    }


def sample_prior_predictive(built_model, draws=1000, random_seed=None):
    if draws < 2:
        raise ValueError("Prior-predictive sampling requires at least two draws.")

    with built_model.model:
        return pm.sample_prior_predictive(draws=draws, random_seed=random_seed, return_inferencedata=True)


def summarize_prior_parameters(prior_predictive):
    prior = _get_group(prior_predictive, "prior")
    records = []

    for name in prior.data_vars:
        values = np.asarray(prior[name])

        if values.ndim == 2:
            parameter_values = values
        elif values.ndim == 3 and values.shape[2] == 1:
            parameter_values = values[:, :, 0]
        else:
            continue

        summary = _summarize_draws(parameter_values)

        records.append(
            {
                "parameter": name,
                "mean": float(summary["mean"]),
                "sd": float(summary["sd"]),
                "q025": float(summary["q025"]),
                "q50": float(summary["q50"]),
                "q975": float(summary["q975"]),
            }
        )

    return pd.DataFrame(records)


def summarize_prior_model_points(prior_predictive, model_points):
    prior = _get_group(prior_predictive, "prior")
    result = model_points.copy()

    variables = [
        "ln_rate_model",
        "theta_CO",
        "theta_OH_Pd",
        "theta_empty_Pd",
        "theta_OH_Ag",
        "theta_empty_Ag",
        "rate_fraction_BF",
        "rate_fraction_ER",
        "rate_fraction_LH",
    ]

    for name in variables:
        if name not in prior:
            continue

        values = np.asarray(prior[name])
        summary = _summarize_draws(values)

        if summary["mean"].ndim != 1 or len(summary["mean"]) != len(result):
            raise ValueError(f"Prior variable '{name}' does not align with model points.")

        for statistic in ["mean", "sd", "q025", "q50", "q975"]:
            result[f"{name}_{statistic}"] = summary[statistic]

    return result


def summarize_prior_observations(prior_predictive, observations):
    result = observations.copy()

    try:
        predictive = _get_group(prior_predictive, "prior_predictive")
    except ValueError:
        return result

    candidate_names = ["ln_rate_observed"]

    for name in candidate_names:
        if name not in predictive:
            continue

        values = np.asarray(predictive[name])
        summary = _summarize_draws(values)

        if summary["mean"].ndim != 1 or len(summary["mean"]) != len(result):
            raise ValueError(f"Prior-predictive variable '{name}' does not align with observations.")

        for statistic in ["mean", "sd", "q025", "q50", "q975"]:
            result[f"{name}_{statistic}"] = summary[statistic]

    return result


def summarize_prior_predictive(prior_predictive, model_data):
    return PriorPredictiveSummary(
        parameter_summary=summarize_prior_parameters(prior_predictive),
        model_point_summary=summarize_prior_model_points(prior_predictive, model_data.model_points),
        observation_summary=summarize_prior_observations(prior_predictive, model_data.observations),
    )