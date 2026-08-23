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


def summarize_residual_curves(observation_diagnostics):
    records = []

    for group_values, curve in observation_diagnostics.groupby(
        list(CURVE_GROUP_COLUMNS),
        sort=False,
    ):
        curve = curve.sort_values("E_V_SHE")

        potential = curve["E_V_SHE"].to_numpy(dtype=float)
        residual = curve[
            "residual_conditional"
        ].to_numpy(dtype=float)

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
                "mean_residual": np.mean(residual),
                "mean_abs_residual": np.mean(
                    np.abs(residual)
                ),
                "rms_residual": np.sqrt(
                    np.mean(residual**2)
                ),
                "lag1_residual_correlation": lag1,
                "residual_slope_per_V": potential_slope,
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

    records = []

    for group_values, group in observation_diagnostics.groupby(
        list(SHARED_RESIDUAL_GROUP_COLUMNS),
        sort=False,
    ):
        wide = group.pivot(
            index="E_V_SHE",
            columns="replicate",
            values="residual_conditional",
        ).dropna()

        if wide.empty:
            continue

        residual = wide.to_numpy(dtype=float)

        shared = residual.mean(
            axis=1,
            keepdims=True,
        )

        replicate_specific = residual - shared

        shared_ss = np.sum(
            np.broadcast_to(
                shared,
                residual.shape,
            )
            ** 2
        )

        replicate_ss = np.sum(
            replicate_specific**2
        )

        total_ss = np.sum(
            residual**2
        )

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
                "shared_rms": float(
                    np.sqrt(np.mean(shared**2))
                ),
                "replicate_specific_rms": float(
                    np.sqrt(
                        np.mean(
                            replicate_specific**2
                        )
                    )
                ),
                "shared_fraction_squared_residual": float(
                    shared_fraction
                ),
                "replicate_fraction_squared_residual": float(
                    replicate_fraction
                ),
            }
        )

    return pd.DataFrame(records)