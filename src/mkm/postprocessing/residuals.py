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

    if (
        len(residual) >= 3
        and np.std(residual[:-1]) > 0
        and np.std(residual[1:]) > 0
    ):
        lag1 = np.corrcoef(
            residual[:-1],
            residual[1:],
        )[0, 1]
    else:
        lag1 = np.nan

    if len(residual) >= 2:
        potential_slope = np.polyfit(
            potential,
            residual,
            1,
        )[0]
    else:
        potential_slope = np.nan

    return {
        "mean": float(np.mean(residual)),
        "mean_abs": float(np.mean(np.abs(residual))),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "lag1": float(lag1),
        "slope_per_V": float(potential_slope),
    }


def summarize_residual_curves(observation_diagnostics):
    required_columns = {
        *CURVE_GROUP_COLUMNS,
        "E_V_SHE",
        "residual_conditional",
    }
    missing = required_columns.difference(observation_diagnostics.columns)
    if missing:
        raise ValueError(
            "Observation diagnostics are missing required "
            f"columns: {sorted(missing)}"
        )

    has_mechanism = "residual_mechanism" in observation_diagnostics.columns
    has_standardized = "standardized_residual_conditional" in observation_diagnostics.columns

    records = []

    for group_values, curve in observation_diagnostics.groupby(
        list(CURVE_GROUP_COLUMNS),
        sort=False,
    ):
        curve = curve.sort_values("E_V_SHE")

        potential = curve["E_V_SHE"].to_numpy(dtype=float)
        conditional_residual = curve["residual_conditional"].to_numpy(dtype=float)

        if has_mechanism:
            mechanism_residual = curve["residual_mechanism"].to_numpy(dtype=float)
        else:
            mechanism_residual = conditional_residual

        mechanism = _curve_statistics(potential, mechanism_residual)
        conditional = _curve_statistics(potential, conditional_residual)

        if has_standardized:
            standardized_conditional = curve[
                "standardized_residual_conditional"
            ].to_numpy(dtype=float)
            standardized = _curve_statistics(potential, standardized_conditional)
            standardized_lag1 = standardized["lag1"]
        else:
            standardized_lag1 = np.nan

        records.append(
            {
                **{
                    column: value
                    for column, value in zip(
                        CURVE_GROUP_COLUMNS,
                        group_values,
                    )
                },
                "n_points": len(curve),
                "mean_residual": mechanism["mean"],
                "mean_abs_residual": mechanism["mean_abs"],
                "rms_residual": mechanism["rms"],
                "lag1_residual_correlation": mechanism["lag1"],
                "residual_slope_per_V": mechanism["slope_per_V"],
                "mechanism_mean_residual": mechanism["mean"],
                "mechanism_mean_abs_residual": mechanism["mean_abs"],
                "mechanism_rms_residual": mechanism["rms"],
                "mechanism_lag1_residual_correlation": mechanism["lag1"],
                "mechanism_residual_slope_per_V": mechanism["slope_per_V"],
                "conditional_mean_residual": conditional["mean"],
                "conditional_mean_abs_residual": conditional["mean_abs"],
                "conditional_rms_residual": conditional["rms"],
                "conditional_lag1_residual_correlation": conditional["lag1"],
                "conditional_residual_slope_per_V": conditional["slope_per_V"],
                "standardized_conditional_lag1_residual_correlation": standardized_lag1,
            }
        )

    return pd.DataFrame(records)


def summarize_shared_replicate_residuals(
    observation_diagnostics,
):
    required_columns = {
        "material",
        "electrolyte_concentration_M",
        "CO_mole_fraction",
        "replicate",
        "E_V_SHE",
        "residual_conditional",
    }

    missing = required_columns.difference(
        observation_diagnostics.columns
    )

    if missing:
        raise ValueError(
            "Observation diagnostics are missing required "
            f"columns: {sorted(missing)}"
        )

    residual_column = (
        "residual_mechanism"
        if "residual_mechanism" in observation_diagnostics.columns
        else "residual_conditional"
    )

    records = []

    for group_values, group in observation_diagnostics.groupby(
        list(SHARED_RESIDUAL_GROUP_COLUMNS),
        sort=False,
    ):
        wide = group.pivot(
            index="E_V_SHE",
            columns="replicate",
            values=residual_column,
        ).dropna()

        if wide.empty:
            continue

        residual = wide.to_numpy(dtype=float)
        shared = residual.mean(axis=1, keepdims=True)
        replicate_specific = residual - shared

        shared_ss = np.sum(
            np.broadcast_to(shared, residual.shape) ** 2
        )
        replicate_ss = np.sum(replicate_specific**2)
        total_ss = np.sum(residual**2)

        if total_ss > 0:
            shared_fraction = shared_ss / total_ss
            replicate_fraction = replicate_ss / total_ss
        else:
            shared_fraction = np.nan
            replicate_fraction = np.nan

        records.append(
            {
                **{
                    column: value
                    for column, value in zip(
                        SHARED_RESIDUAL_GROUP_COLUMNS,
                        group_values,
                    )
                },
                "n_potentials": len(wide),
                "n_replicates": wide.shape[1],
                "shared_rms": float(np.sqrt(np.mean(shared**2))),
                "replicate_specific_rms": float(
                    np.sqrt(np.mean(replicate_specific**2))
                ),
                "shared_fraction_squared_residual": float(shared_fraction),
                "replicate_fraction_squared_residual": float(replicate_fraction),
            }
        )

    return pd.DataFrame(records)
