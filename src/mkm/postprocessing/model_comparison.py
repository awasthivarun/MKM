from dataclasses import dataclass
from itertools import combinations

import arviz_stats as azs
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOOModelComparison:
    loo_results: dict
    loo_summary: pd.DataFrame
    compare_table: pd.DataFrame
    pareto_k: pd.DataFrame
    pointwise_elpd: pd.DataFrame
    pointwise_differences: pd.DataFrame
    difference_summary: pd.DataFrame


def compute_loo_results(inference_data_by_model, var_name="ln_rate_observed"):
    results = {}

    for model_name, inference_data in inference_data_by_model.items():
        results[model_name] = azs.loo(
            inference_data,
            var_name=var_name,
            pointwise=True,
        )

    n_points = {name: result.n_data_points for name, result in results.items()}
    if len(set(n_points.values())) != 1:
        raise ValueError(f"LOO results do not contain the same number of observations: {n_points}")

    return results


def build_loo_summary(loo_results):
    records = []

    for model_name, result in loo_results.items():
        pareto_k = np.asarray(result.pareto_k, dtype=float).reshape(-1)

        records.append(
            {
                "model": model_name,
                "elpd": float(result.elpd),
                "se": float(result.se),
                "p": float(result.p),
                "n_samples": int(result.n_samples),
                "n_data_points": int(result.n_data_points),
                "good_k": float(result.good_k),
                "max_pareto_k": float(np.max(pareto_k)),
                "n_pareto_k_above_good_k": int(np.sum(pareto_k > float(result.good_k))),
                "warning": bool(result.warning),
            }
        )

    return pd.DataFrame(records).sort_values("elpd", ascending=False).reset_index(drop=True)


def build_pareto_k_table(loo_results, observations):
    records = []

    for model_name, result in loo_results.items():
        values = np.asarray(result.pareto_k, dtype=float).reshape(-1)

        if len(values) != len(observations):
            raise ValueError(
                f"Pareto-k values for '{model_name}' have length {len(values)}, "
                f"but there are {len(observations)} observations."
            )

        frame = observations.copy()
        frame.insert(0, "model", model_name)
        frame["pareto_k"] = values
        frame["good_k"] = float(result.good_k)
        frame["pareto_k_above_good_k"] = frame["pareto_k"] > frame["good_k"]
        records.append(frame)

    return pd.concat(records, ignore_index=True)


def build_pointwise_elpd_table(loo_results, observations):
    result = observations.copy()

    for model_name, loo_result in loo_results.items():
        values = np.asarray(loo_result.elpd_i, dtype=float).reshape(-1)

        if len(values) != len(observations):
            raise ValueError(
                f"Pointwise ELPD values for '{model_name}' have length {len(values)}, "
                f"but there are {len(observations)} observations."
            )

        result[f"elpd_{model_name}"] = values

    return result


def build_pointwise_elpd_differences(pointwise_elpd, model_names):
    records = []

    for numerator, denominator in combinations(model_names, 2):
        frame = pointwise_elpd.copy()
        frame["numerator_model"] = numerator
        frame["denominator_model"] = denominator
        frame["comparison"] = f"{numerator}_minus_{denominator}"
        frame["elpd_difference"] = frame[f"elpd_{numerator}"] - frame[f"elpd_{denominator}"]
        records.append(frame)

    return pd.concat(records, ignore_index=True)


def summarize_pointwise_elpd_differences(pointwise_differences):
    records = []

    for comparison, frame in pointwise_differences.groupby("comparison", sort=False):
        values = frame["elpd_difference"].to_numpy(dtype=float)
        n = len(values)

        records.append(
            {
                "comparison": comparison,
                "numerator_model": frame["numerator_model"].iloc[0],
                "denominator_model": frame["denominator_model"].iloc[0],
                "elpd_difference": float(np.sum(values)),
                "dse": float(np.sqrt(n * np.var(values, ddof=1))),
                "mean_pointwise_difference": float(np.mean(values)),
                "median_pointwise_difference": float(np.median(values)),
                "fraction_points_favoring_numerator": float(np.mean(values > 0.0)),
                "max_pointwise_gain": float(np.max(values)),
                "max_pointwise_loss": float(np.min(values)),
            }
        )

    return pd.DataFrame(records)


def build_loo_model_comparison(inference_data_by_model, observations, var_name="ln_rate_observed"):
    model_names = list(inference_data_by_model)

    loo_results = compute_loo_results(
        inference_data_by_model=inference_data_by_model,
        var_name=var_name,
    )

    loo_summary = build_loo_summary(loo_results)

    compare_table = azs.compare(
        loo_results,
        method="stacking",
        round_to="none",
    )
    compare_table = compare_table.rename_axis("model").reset_index()

    pareto_k = build_pareto_k_table(
        loo_results=loo_results,
        observations=observations,
    )

    pointwise_elpd = build_pointwise_elpd_table(
        loo_results=loo_results,
        observations=observations,
    )

    pointwise_differences = build_pointwise_elpd_differences(
        pointwise_elpd=pointwise_elpd,
        model_names=model_names,
    )

    difference_summary = summarize_pointwise_elpd_differences(
        pointwise_differences
    )

    return LOOModelComparison(
        loo_results=loo_results,
        loo_summary=loo_summary,
        compare_table=compare_table,
        pareto_k=pareto_k,
        pointwise_elpd=pointwise_elpd,
        pointwise_differences=pointwise_differences,
        difference_summary=difference_summary,
    )