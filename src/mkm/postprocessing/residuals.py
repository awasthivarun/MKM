import numpy as np
import pandas as pd


CURVE_GROUP_COLUMNS = (
    "material",
    "electrolyte_concentration_M",
    "CO_mole_fraction",
    "replicate",
)

SHARED_RESIDUAL_GROUP_COLUMNS = (
    "material",
    "electrolyte_concentration_M",
    "CO_mole_fraction",
)


def _curve_statistics(potential, residual):
    residual = np.asarray(residual, dtype=float)

    if len(residual) >= 3 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
        lag1 = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    else:
        lag1 = np.nan

    potential_slope = np.polyfit(potential, residual, 1)[0] if len(residual) >= 2 else np.nan
    return {
        "mean": float(np.mean(residual)),
        "mean_abs": float(np.mean(np.abs(residual))),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "lag1": float(lag1),
        "slope_per_V": float(potential_slope),
    }


def summarize_residual_curves(observation_diagnostics):
    required_columns = {*CURVE_GROUP_COLUMNS, "E_V_SHE", "residual"}
    missing = required_columns.difference(observation_diagnostics.columns)
    if missing:
        raise ValueError(
            f"Observation diagnostics are missing required columns: {sorted(missing)}"
        )

    has_standardized = "standardized_residual" in observation_diagnostics.columns
    records = []

    for group_values, curve in observation_diagnostics.groupby(
        list(CURVE_GROUP_COLUMNS), sort=False
    ):
        curve = curve.sort_values("E_V_SHE")
        potential = curve["E_V_SHE"].to_numpy(dtype=float)
        statistics = _curve_statistics(
            potential,
            curve["residual"].to_numpy(dtype=float),
        )
        standardized_lag1 = np.nan
        if has_standardized:
            standardized_lag1 = _curve_statistics(
                potential,
                curve["standardized_residual"].to_numpy(dtype=float),
            )["lag1"]

        records.append(
            {
                **{
                    column: value
                    for column, value in zip(CURVE_GROUP_COLUMNS, group_values)
                },
                "n_points": len(curve),
                "mean_residual": statistics["mean"],
                "mean_abs_residual": statistics["mean_abs"],
                "rms_residual": statistics["rms"],
                "lag1_residual_correlation": statistics["lag1"],
                "residual_slope_per_V": statistics["slope_per_V"],
                "standardized_lag1_residual_correlation": standardized_lag1,
            }
        )

    return pd.DataFrame(records)


def summarize_shared_replicate_residuals(observation_diagnostics):
    required_columns = {
        "material",
        "electrolyte_concentration_M",
        "CO_mole_fraction",
        "replicate",
        "E_V_SHE",
        "residual",
    }
    missing = required_columns.difference(observation_diagnostics.columns)
    if missing:
        raise ValueError(
            f"Observation diagnostics are missing required columns: {sorted(missing)}"
        )

    records = []
    for group_values, group in observation_diagnostics.groupby(
        list(SHARED_RESIDUAL_GROUP_COLUMNS), sort=False
    ):
        wide = group.pivot(
            index="E_V_SHE",
            columns="replicate",
            values="residual",
        ).dropna()
        if wide.empty:
            continue

        residual = wide.to_numpy(dtype=float)
        shared = residual.mean(axis=1, keepdims=True)
        replicate_specific = residual - shared
        shared_ss = np.sum(np.broadcast_to(shared, residual.shape) ** 2)
        replicate_ss = np.sum(replicate_specific**2)
        total_ss = np.sum(residual**2)

        records.append(
            {
                **{
                    column: value
                    for column, value in zip(
                        SHARED_RESIDUAL_GROUP_COLUMNS, group_values
                    )
                },
                "n_potentials": len(wide),
                "n_replicates": wide.shape[1],
                "shared_rms": float(np.sqrt(np.mean(shared**2))),
                "replicate_specific_rms": float(
                    np.sqrt(np.mean(replicate_specific**2))
                ),
                "shared_fraction_squared_residual": (
                    float(shared_ss / total_ss) if total_ss > 0 else np.nan
                ),
                "replicate_fraction_squared_residual": (
                    float(replicate_ss / total_ss) if total_ss > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(records)
