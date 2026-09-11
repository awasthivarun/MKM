"""Potential second-order kinetic observables for AgPd posterior diagnostics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mkm.observable_maps import (
    LinearObservableMap,
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
    build_potential_derivative_map,
    evaluate_linear_observable_map_draws,
)
from mkm.postprocessing.diagnostics import summarize_samples


@dataclass(frozen=True)
class SecondOrderOutputs:
    points: pd.DataFrame
    summary: pd.DataFrame


def _get_posterior(inference_data):
    if hasattr(inference_data, "posterior"):
        return inference_data.posterior

    try:
        return inference_data["posterior"]
    except Exception as error:
        raise ValueError("Inference data does not contain a posterior group.") from error


def _central_potential_difference(frame, value_column, group_columns, potential_step_V):
    potential_step_V = float(potential_step_V)
    if not np.isfinite(potential_step_V) or potential_step_V <= 0:
        raise ValueError("Potential derivative step must be finite and positive.")

    required = [*group_columns, "analysis_grid_index", "E_V_SHE", value_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Second-order experimental table is missing columns: {missing}")

    records = []
    for _, group in frame[required].groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("E_V_SHE").reset_index(drop=True)
        potential = group["E_V_SHE"].to_numpy(dtype=float)
        values = group[value_column].to_numpy(dtype=float)

        if len(np.unique(potential)) != len(potential):
            raise ValueError("Second-order experimental derivative requires unique potential points per curve.")
        if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(values)):
            raise ValueError("Second-order experimental derivative received non-finite values.")

        for i, target in group.iterrows():
            target_potential = float(target["E_V_SHE"])
            lower = np.flatnonzero(np.isclose(potential, target_potential - potential_step_V, rtol=0, atol=1e-12))
            upper = np.flatnonzero(np.isclose(potential, target_potential + potential_step_V, rtol=0, atol=1e-12))

            if len(lower) == 0 or len(upper) == 0:
                continue
            if len(lower) != 1 or len(upper) != 1:
                raise ValueError("Expected exactly one symmetric source point on each side of target potential.")

            record = {column: target[column] for column in group_columns}
            record.update(
                {
                    "analysis_grid_index": int(target["analysis_grid_index"]),
                    "E_V_SHE": target_potential,
                    "potential_step_V": potential_step_V,
                    "experimental": float((values[upper[0]] - values[lower[0]]) / (2.0 * potential_step_V)),
                }
            )
            records.append(record)

    return pd.DataFrame(records)


def _posterior_summary(outputs, draws):
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3:
        raise ValueError("Second-order posterior draws must have shape (chain, draw, observable).")
    if draws.shape[-1] != len(outputs):
        raise ValueError("Second-order posterior draws do not align with output metadata.")
    if not np.all(np.isfinite(draws)):
        raise ValueError("Second-order posterior draws contain non-finite values.")

    pooled = draws.reshape((-1, draws.shape[-1]))
    statistics = summarize_samples(pooled)
    result = outputs.copy().reset_index(drop=True)
    result = result.drop(columns=["observable_id"], errors="ignore")

    for statistic, values in statistics.items():
        result[statistic] = values

    chain_medians = np.median(draws, axis=1)
    result["chain_median_range"] = np.ptp(chain_medians, axis=0) if draws.shape[0] > 1 else 0.0
    return result


def _compare_second_order(posterior_frame, experimental_frame, key_columns, observable):
    posterior_frame = posterior_frame.copy()
    experimental_frame = experimental_frame.copy()

    merged = posterior_frame.merge(
        experimental_frame[[*key_columns, "experimental"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    merged.insert(0, "observable", observable)
    merged.insert(1, "units", "V^-1")
    merged["residual_median"] = merged["median"] - merged["experimental"]
    merged["posterior_95_hdi_contains_experimental"] = (
        (merged["experimental"] >= merged["hdi95_lower"])
        & (merged["experimental"] <= merged["hdi95_upper"])
    )
    return merged


def _summary_row(observable, frame, potential_step_V):
    abs_residual = np.abs(frame["residual_median"].to_numpy(dtype=float))
    chain_range = frame["chain_median_range"].to_numpy(dtype=float)
    return {
        "observable": observable,
        "units": "V^-1",
        "potential_step_V": float(potential_step_V),
        "n_points": int(len(frame)),
        "median_abs_residual": float(np.median(abs_residual)) if len(abs_residual) else np.nan,
        "posterior_95_hdi_contains_experimental": (
            float(frame["posterior_95_hdi_contains_experimental"].mean()) if len(frame) else np.nan
        ),
        "median_chain_median_range": float(np.median(chain_range)) if len(chain_range) else np.nan,
        "p95_chain_median_range": float(np.quantile(chain_range, 0.95)) if len(chain_range) else np.nan,
        "max_chain_median_range": float(np.max(chain_range)) if len(chain_range) else np.nan,
        "experimental_uncertainty_propagated": False,
    }


def _alpha_derivative_metadata(derivative_map, conditions):
    metadata = conditions[
        ["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]
    ].rename(columns={"electrolyte_concentration_M": "C_KOH_M"})
    outputs = derivative_map.outputs.merge(metadata, on="condition_id", how="left", validate="many_to_one")
    return LinearObservableMap(outputs=outputs, terms=derivative_map.terms)


def _align_delta2(dalpha_map, dalpha_draws, ddelta_oh_map, ddelta_oh_draws):
    left = dalpha_map.outputs.reset_index(names="left_index")
    right = ddelta_oh_map.outputs.reset_index(names="right_index")
    keys = ["material", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE", "potential_step_V"]

    matches = left.merge(
        right[["right_index", *keys]],
        on=keys,
        how="inner",
        validate="many_to_one",
    ).sort_values("left_index")

    left_index = matches["left_index"].to_numpy(dtype=np.int64)
    right_index = matches["right_index"].to_numpy(dtype=np.int64)
    outputs = dalpha_map.outputs.iloc[left_index].copy().reset_index(drop=True)
    outputs["observable_id"] = np.arange(len(outputs), dtype=np.int64)
    draws = dalpha_draws[..., left_index] - ddelta_oh_draws[..., right_index]
    return outputs, draws


def _experimental_second_order(observable_points, potential_step_V):
    alpha = observable_points.loc[observable_points["observable"] == "alpha"].copy()
    oh = observable_points.loc[observable_points["observable"] == "delta_OH"].copy()
    co = observable_points.loc[observable_points["observable"] == "delta_CO"].copy()

    dalpha = _central_potential_difference(
        alpha,
        value_column="alpha_mean",
        group_columns=["material", "C_KOH_M", "CO_mole_fraction"],
        potential_step_V=potential_step_V,
    )
    ddelta_oh = _central_potential_difference(
        oh,
        value_column="delta_OH",
        group_columns=["material", "CO_mole_fraction"],
        potential_step_V=potential_step_V,
    )
    ddelta_co = _central_potential_difference(
        co,
        value_column="delta_CO",
        group_columns=[
            "material",
            "C_KOH_M",
            "CO_lower_mole_fraction",
            "CO_upper_mole_fraction",
        ],
        potential_step_V=potential_step_V,
    )

    delta2_keys = ["material", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE", "potential_step_V"]
    delta2 = dalpha.merge(
        ddelta_oh[[*delta2_keys, "experimental"]],
        on=delta2_keys,
        how="inner",
        suffixes=("_alpha", "_oh"),
        validate="many_to_one",
    )
    delta2["experimental"] = delta2["experimental_alpha"] - delta2["experimental_oh"]
    delta2 = delta2.drop(columns=["experimental_alpha", "experimental_oh"])
    return dalpha, ddelta_oh, ddelta_co, delta2


def build_second_order_outputs(
    inference_data,
    model_points,
    conditions,
    observable_points,
    temperature_K,
    koh_values,
    co_values,
    potential_step_V=0.05,
):
    """Build posterior and experimental potential derivatives of the kinetic observables.

    Experimental derivatives use symmetric finite differences of the already-computed first-order
    observable point estimates. Their uncertainty is intentionally not propagated because adjacent
    first-order estimates are correlated and the current processed products do not carry that covariance.
    """
    potential_step_V = float(potential_step_V)
    if not np.isfinite(potential_step_V) or potential_step_V <= 0:
        raise ValueError("Potential derivative step must be finite and positive.")

    alpha_map = build_alpha_map(model_points, temperature_K)
    oh_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=koh_values,
        group_columns=["material", "CO_mole_fraction"],
    )
    co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=co_values,
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    dalpha_map = _alpha_derivative_metadata(
        build_potential_derivative_map(alpha_map, ["condition_id"], potential_step_V),
        conditions,
    )
    ddelta_oh_map = build_potential_derivative_map(
        oh_map,
        ["material", "CO_mole_fraction"],
        potential_step_V,
    )
    ddelta_co_map = build_potential_derivative_map(
        co_map,
        ["material", "electrolyte_concentration_M", "lower_CO_mole_fraction", "upper_CO_mole_fraction"],
        potential_step_V,
    )

    posterior = _get_posterior(inference_data)
    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior does not contain 'ln_rate_model'.")
    ln_rate = np.asarray(posterior["ln_rate_model"], dtype=float)

    dalpha_draws = evaluate_linear_observable_map_draws(ln_rate, dalpha_map)
    ddelta_oh_draws = evaluate_linear_observable_map_draws(ln_rate, ddelta_oh_map)
    ddelta_co_draws = evaluate_linear_observable_map_draws(ln_rate, ddelta_co_map)
    delta2_outputs, delta2_draws = _align_delta2(dalpha_map, dalpha_draws, ddelta_oh_map, ddelta_oh_draws)

    posterior_dalpha = _posterior_summary(dalpha_map.outputs, dalpha_draws)
    posterior_oh = _posterior_summary(ddelta_oh_map.outputs, ddelta_oh_draws)
    posterior_co = _posterior_summary(ddelta_co_map.outputs, ddelta_co_draws).rename(
        columns={
            "electrolyte_concentration_M": "C_KOH_M",
            "lower_CO_mole_fraction": "CO_lower_mole_fraction",
            "upper_CO_mole_fraction": "CO_upper_mole_fraction",
        }
    )
    posterior_delta2 = _posterior_summary(delta2_outputs, delta2_draws)

    experimental_dalpha, experimental_oh, experimental_co, experimental_delta2 = _experimental_second_order(
        observable_points,
        potential_step_V,
    )

    frames = {
        "dalpha_dE": _compare_second_order(
            posterior_dalpha,
            experimental_dalpha,
            ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE", "potential_step_V"],
            "dalpha_dE",
        ),
        "ddelta_OH_dE": _compare_second_order(
            posterior_oh,
            experimental_oh,
            ["material", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE", "potential_step_V"],
            "ddelta_OH_dE",
        ),
        "ddelta_CO_dE": _compare_second_order(
            posterior_co,
            experimental_co,
            [
                "material",
                "C_KOH_M",
                "CO_lower_mole_fraction",
                "CO_upper_mole_fraction",
                "analysis_grid_index",
                "E_V_SHE",
                "potential_step_V",
            ],
            "ddelta_CO_dE",
        ),
        "delta2": _compare_second_order(
            posterior_delta2,
            experimental_delta2,
            ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE", "potential_step_V"],
            "delta2",
        ),
    }

    points = pd.concat(frames.values(), ignore_index=True, sort=False)
    summary = pd.DataFrame([_summary_row(name, frame, potential_step_V) for name, frame in frames.items()])
    return SecondOrderOutputs(points=points, summary=summary)
