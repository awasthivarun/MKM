"""Material-resolved summaries for multi-material posterior fits."""

import numpy as np
import pandas as pd


def summarize_observation_diagnostics_by_material(observations):
    required = {
        "material",
        "residual",
        "standardized_residual",
        "observed_inside_predictive_95_hdi",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Observation diagnostics are missing columns: {sorted(missing)}.")

    records = []
    for material, frame in observations.groupby("material", sort=False):
        residual = frame["residual"].to_numpy(dtype=float)
        standardized = frame["standardized_residual"].to_numpy(dtype=float)
        records.append(
            {
                "material": material,
                "n_observations": len(frame),
                "residual_mean": float(np.mean(residual)),
                "residual_rms": float(np.sqrt(np.mean(residual**2))),
                "residual_mae": float(np.mean(np.abs(residual))),
                "median_abs_standardized_residual": float(
                    np.median(np.abs(standardized))
                ),
                "predictive_95_hdi_coverage": float(
                    frame["observed_inside_predictive_95_hdi"].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_residual_structure_by_material(curve_residuals, shared_residuals):
    materials = []
    if not curve_residuals.empty and "material" in curve_residuals:
        materials.extend(curve_residuals["material"].drop_duplicates().tolist())
    if not shared_residuals.empty and "material" in shared_residuals:
        materials.extend(shared_residuals["material"].drop_duplicates().tolist())

    records = []
    for material in dict.fromkeys(materials):
        record = {"material": material}

        curve = curve_residuals.loc[curve_residuals["material"] == material]
        if not curve.empty:
            record["median_lag1_residual_correlation"] = float(
                curve["lag1_residual_correlation"].median()
            )
            record["median_abs_residual_slope_per_V"] = float(
                curve["residual_slope_per_V"].abs().median()
            )

        shared = shared_residuals.loc[shared_residuals["material"] == material]
        if not shared.empty:
            record["median_shared_squared_residual_fraction"] = float(
                shared["shared_fraction_squared_residual"].median()
            )

        records.append(record)

    return pd.DataFrame(records)


def summarize_pointwise_loo_by_material(pointwise):
    required = {"material", "elpd_loo", "pareto_k"}
    missing = required - set(pointwise.columns)
    if missing:
        raise ValueError(f"Pointwise LOO table is missing columns: {sorted(missing)}.")

    records = []
    for material, frame in pointwise.groupby("material", sort=False):
        elpd = frame["elpd_loo"].to_numpy(dtype=float)
        pareto_k = frame["pareto_k"].to_numpy(dtype=float)
        records.append(
            {
                "material": material,
                "n_observations": len(frame),
                "elpd_loo_contribution": float(np.sum(elpd)),
                "mean_elpd_loo": float(np.mean(elpd)),
                "max_pareto_k": float(np.max(pareto_k)),
                "n_pareto_k_above_0p7": int(np.sum(pareto_k > 0.7)),
            }
        )

    return pd.DataFrame(records)
