from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm

from mkm.observable_maps import evaluate_linear_observable_map_draws


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
        raise ValueError(
            f"Prior-predictive result does not contain group '{group_name}'."
        ) from error


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
        return pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
            return_inferencedata=True,
        )


def _component_label(data, indexers):
    labels = []
    for dim, index in indexers.items():
        if dim in data.coords and data.coords[dim].ndim == 1:
            value = data.coords[dim].values[index]
        else:
            value = index
        labels.append(f"{dim}={value}")
    return ",".join(labels)


def summarize_prior_parameters(prior_predictive, parameter_names=None):
    prior = _get_group(prior_predictive, "prior")
    names = tuple(parameter_names) if parameter_names is not None else tuple(prior.data_vars)
    records = []

    for name in names:
        if name not in prior:
            raise ValueError(f"Prior group is missing configured parameter '{name}'.")

        data = prior[name].squeeze(drop=True)
        extra_dims = tuple(dim for dim in data.dims if dim not in {"chain", "draw"})
        if not extra_dims:
            components = [(name, data)]
        else:
            components = []
            shape = tuple(data.sizes[dim] for dim in extra_dims)
            for index in np.ndindex(shape):
                indexers = dict(zip(extra_dims, index, strict=True))
                label = _component_label(data, indexers)
                components.append((f"{name}[{label}]", data.isel(indexers, drop=True)))

        for label, component in components:
            summary = _summarize_draws(np.asarray(component, dtype=float))
            records.append(
                {
                    "parameter": label,
                    "variable": name,
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

        summary = _summarize_draws(np.asarray(prior[name]))
        if summary["mean"].ndim != 1 or len(summary["mean"]) != len(result):
            raise ValueError(f"Prior variable '{name}' does not align with model points.")

        for statistic in ("mean", "sd", "q025", "q50", "q975"):
            result[f"{name}_{statistic}"] = summary[statistic]

        if name == "ln_rate_model":
            rate_summary = _summarize_draws(np.exp(np.asarray(prior[name], dtype=float)))
            for statistic in ("mean", "sd", "q025", "q50", "q975"):
                result[f"rate_model_{statistic}"] = rate_summary[statistic]

    return result


def summarize_prior_observations(prior_predictive, observations):
    result = observations.copy()
    try:
        predictive = _get_group(prior_predictive, "prior_predictive")
    except ValueError:
        return result

    if "rate_observed" not in predictive:
        return result

    summary = _summarize_draws(np.asarray(predictive["rate_observed"]))
    if summary["mean"].ndim != 1 or len(summary["mean"]) != len(result):
        raise ValueError(
            "Prior-predictive variable 'rate_observed' does not align with observations."
        )

    for statistic in ("mean", "sd", "q025", "q50", "q975"):
        result[f"rate_observed_{statistic}"] = summary[statistic]
    return result


def summarize_prior_predictive(prior_predictive, model_data, parameter_names=None):
    return PriorPredictiveSummary(
        parameter_summary=summarize_prior_parameters(
            prior_predictive,
            parameter_names=parameter_names,
        ),
        model_point_summary=summarize_prior_model_points(
            prior_predictive,
            model_data.model_points,
        ),
        observation_summary=summarize_prior_observations(
            prior_predictive,
            model_data.observations,
        ),
    )


def summarize_prior_linear_observable(prior_predictive, observable_map):
    prior = _get_group(prior_predictive, "prior")
    if "ln_rate_model" not in prior:
        raise ValueError("Prior group does not contain 'ln_rate_model'.")

    observable_draws = evaluate_linear_observable_map_draws(
        np.asarray(prior["ln_rate_model"]),
        observable_map,
    )
    summary = _summarize_draws(observable_draws)
    result = observable_map.outputs.copy()

    if summary["mean"].ndim != 1 or len(summary["mean"]) != len(result):
        raise ValueError("Derived-observable draws do not align with observable-map outputs.")

    for statistic in ("mean", "sd", "q025", "q50", "q975"):
        result[statistic] = summary[statistic]
    return result
