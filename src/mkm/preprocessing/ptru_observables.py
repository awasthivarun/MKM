
"""Experimental kinetic observables for Pt/PtRu CO oxidation in acidic and basic media."""

import numpy as np
import pandas as pd

from mkm.preprocessing.observables import (
    add_transfer_coefficients,
    calculate_adjacent_log_orders,
    calculate_log_slope_order,
    summarize_transfer_coefficients,
)


def calculate_alpha_products(analysis, temperature_K):
    """Calculate alpha per replicate, then summarize available replicates at each potential."""
    curve_keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "replicate"]
    point_keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index",
                  "E_V_SHE"]

    if "sample_id" in analysis.columns:
        curve_keys.insert(2, "sample_id")
        point_keys.insert(2, "sample_id")

    replicates = add_transfer_coefficients(
        data=analysis,
        group_columns=curve_keys,
        temperature_K=temperature_K,
        potential_column="E_V_SHE",
        log_rate_column="ln_rate",
        output_column="alpha",
    )

    summary = summarize_transfer_coefficients(
        data=replicates,
        group_columns=point_keys,
        replicate_column="replicate",
        alpha_column="alpha",
    )

    return replicates, summary


def calculate_co_orders(analysis):
    """Estimate orders from mean rates, with independent replicate-variation propagation."""
    keys = ["material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index"]

    if analysis.duplicated([*keys, "replicate"]).any():
        raise ValueError("Duplicate replicate observations")
    if (analysis["rate_s_inv"] <= 0).any() or not np.isfinite(analysis["ln_rate"]).all():
        raise ValueError("CO orders require positive, finite rates")
    if "sample_id" in analysis.columns and analysis["sample_id"].nunique() != 1:
        raise ValueError("Process Em sample preparations separately")

    summary = (
        analysis.groupby(keys, as_index=False)
        .agg(
            E_V_SHE=("E_V_SHE", "first"),
            rate_mean_s_inv=("rate_s_inv", "mean"),
            rate_sd_s_inv=("rate_s_inv", "std"),
            ln_rate_sd=("ln_rate", "std"),
            n_replicates=("replicate", "nunique"),
        )
    )
    summary["ln_rate_mean"] = np.log(summary["rate_mean_s_inv"])

    orders = calculate_adjacent_log_orders(
        data=summary,
        varying_column="CO_mole_fraction",
        varying_values=sorted(analysis["CO_mole_fraction"].unique()),
        group_columns=["material", "C_electrolyte_M"],
        output_column="delta_CO",
        output_sd_column="delta_CO_sd",
        lower_value_column="CO_lower_mole_fraction",
        upper_value_column="CO_upper_mole_fraction",
    )
    orders.insert(0, "medium", analysis["medium"].iloc[0])

    # Retain the original per-condition sample counts and rate summaries.
    for side in ("lower", "upper"):
        co_col = f"CO_{side}_mole_fraction"
        stats = summary.rename(columns={
            "CO_mole_fraction": co_col,
            "n_replicates": f"n_{side}",
            "rate_mean_s_inv": f"rate_{side}_mean_s_inv",
            "rate_sd_s_inv": f"rate_{side}_sd_s_inv",
        })
        join_keys = ["material", "C_electrolyte_M", "analysis_grid_index", co_col]
        columns = [*join_keys, f"n_{side}", f"rate_{side}_mean_s_inv", f"rate_{side}_sd_s_inv"]
        orders = orders.merge(stats[columns], on=join_keys, validate="one_to_one")

    if "sample_id" in analysis.columns:
        orders["sample_id"] = analysis["sample_id"].iloc[0]

    if orders.empty or not np.isfinite(orders["delta_CO"]).all():
        raise ValueError("No valid CO orders calculated")
    return orders


def calculate_ion_orders(analysis, concentrations):
    """Fit one log-rate slope over all four concentrations at each common SHE potential."""
    keys = ["material", "CO_mole_fraction", "C_electrolyte_M", "analysis_grid_index"]

    if analysis.duplicated([*keys, "replicate"]).any():
        raise ValueError("Duplicate replicate observations")
    if (analysis["rate_s_inv"] <= 0).any() or not np.isfinite(analysis["ln_rate"]).all():
        raise ValueError("Electrolyte orders require positive, finite rates")

    summary = (
        analysis.groupby(keys, as_index=False)
        .agg(
            E_V_SHE=("E_V_SHE", "first"),
            rate_mean_s_inv=("rate_s_inv", "mean"),
            ln_rate_sd=("ln_rate", "std"),
            n_replicates=("replicate", "nunique"),
        )
    )
    summary["ln_rate_mean"] = np.log(summary["rate_mean_s_inv"])

    medium = analysis["medium"].iloc[0]
    order_name = "delta_Hplus" if medium == "acidic" else "delta_OH"

    orders = calculate_log_slope_order(
        data=summary,
        varying_column="C_electrolyte_M",
        varying_values=sorted(concentrations),
        group_columns=["material", "CO_mole_fraction"],
        output_column=order_name,
        output_sd_column=f"{order_name}_sd",
    )
    orders.insert(0, "medium", medium)

    if orders.empty or not np.isfinite(orders[order_name]).all():
        raise ValueError("No valid electrolyte reaction orders calculated")
    return orders
