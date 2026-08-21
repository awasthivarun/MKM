import numpy as np
import pandas as pd

from mkm.model_data import ModelDataTables


def summarize_point_dispersion(model_data: ModelDataTables):
    observations = model_data.observations.copy()

    summary = (
        observations.groupby(
            [
                "model_point_id",
                "condition_id",
                "material",
                "electrolyte_concentration_M",
                "CO_mole_fraction",
                "analysis_grid_index",
                "E_V_SHE",
            ],
            sort=False,
        )
        .agg(
            n_replicates=("ln_rate", "size"),
            ln_rate_mean=("ln_rate", "mean"),
            ln_rate_sd=("ln_rate", "std"),
            rate_mean_s_inv=("rate_s_inv", "mean"),
            rate_sd_s_inv=("rate_s_inv", "std"),
        )
        .reset_index()
    )

    return summary


def calculate_centered_log_residuals(model_data: ModelDataTables):
    observations = model_data.observations.copy()

    point_means = observations.groupby("model_point_id")["ln_rate"].mean().rename("ln_rate_point_mean")
    residuals = observations.join(point_means, on="model_point_id")
    residuals["ln_rate_centered_residual"] = residuals["ln_rate"] - residuals["ln_rate_point_mean"]

    return residuals


def calculate_pooled_log_rate_sd(model_data: ModelDataTables):
    residuals = calculate_centered_log_residuals(model_data)
    grouped = residuals.groupby("model_point_id", sort=False)

    numerator = 0.0
    degrees_of_freedom = 0

    for _, group in grouped:
        n = len(group)

        if n < 2:
            continue

        values = group["ln_rate_centered_residual"].to_numpy(dtype=float)

        numerator += np.sum(values**2)
        degrees_of_freedom += n - 1

    if degrees_of_freedom <= 0:
        raise ValueError("At least one model point must have two or more replicates.")

    pooled_variance = numerator / degrees_of_freedom

    return np.sqrt(pooled_variance)


def summarize_material_dispersion(model_data: ModelDataTables):
    residuals = calculate_centered_log_residuals(model_data)
    records = []

    for material, material_data in residuals.groupby("material", sort=False):
        numerator = 0.0
        degrees_of_freedom = 0

        for _, group in material_data.groupby("model_point_id", sort=False):
            n = len(group)

            if n < 2:
                continue

            values = group["ln_rate_centered_residual"].to_numpy(dtype=float)

            numerator += np.sum(values**2)
            degrees_of_freedom += n - 1

        if degrees_of_freedom > 0:
            pooled_sd = np.sqrt(numerator / degrees_of_freedom)
        else:
            pooled_sd = np.nan

        records.append(
            {
                "material": material,
                "n_model_points": material_data["model_point_id"].nunique(),
                "n_observations": len(material_data),
                "pooled_ln_rate_sd": pooled_sd,
            }
        )

    return pd.DataFrame(records)