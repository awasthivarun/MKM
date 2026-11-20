"""PtRu observable loading and posterior/experimental comparisons."""

import numpy as np
import pandas as pd
from mkm.observable_maps import build_adjacent_log_order_map, build_alpha_map, build_log_slope_order_map
from mkm.postprocessing.observable_comparison import (
    build_experimental_observable_comparison, summarize_experimental_observable,
)
from mkm.postprocessing.observables import summarize_posterior_linear_observable


def ptru_ion_name(medium):
    if medium == "acidic":
        return "delta_Hplus"
    if medium == "basic":
        return "delta_OH"
    raise ValueError(f"Unsupported PtRu medium: {medium}")


def ptru_electrolyte_label(medium):
    return "HClO4" if medium == "acidic" else "KOH"


def load_ptru_experimental_observables(paths, medium, material="Pt100", materials=None):
    """Load selected Todd observables for one Pt or PtRu material."""
    materials = (material,) if materials is None else tuple(materials)
    ion_name = ptru_ion_name(medium)
    names = {
        "alpha": "alpha_summary",
        ion_name: ion_name,
        "delta_CO": "delta_CO",
    }
    frames = {}

    for observable, product in names.items():
        path = paths.ptru_observable_path(medium, product, selected=True)
        data = pd.read_parquet(path)
        data = data.loc[data["material"].isin(materials)].copy()

        if "medium" in data.columns and not data["medium"].eq(medium).all():
            raise ValueError(f"Experimental {observable} table contains another medium.")

        if data.empty:
            raise ValueError(f"No selected {materials} {observable} observations found: {path}")

        frames[observable] = data

    return frames


def build_ptru_observable_comparisons(run, paths, temperature_K):
    """Build PtRu posterior/experimental comparisons using common observable maps."""
    medium = run.specification.medium
    ion_name = ptru_ion_name(medium)
    model_points = run.model_data.model_points
    conditions = run.model_data.conditions

    concentrations = sorted(model_points["electrolyte_concentration_M"].unique())
    co_fractions = sorted(model_points["CO_mole_fraction"].unique())

    alpha_map = build_alpha_map(model_points, temperature_K)
    ion_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=concentrations,
        group_columns=["material", "CO_mole_fraction"],
    )
    co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=co_fractions,
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    alpha = summarize_posterior_linear_observable(run.inference_data, alpha_map)
    ion = summarize_posterior_linear_observable(run.inference_data, ion_map)
    co = summarize_posterior_linear_observable(run.inference_data, co_map)

    condition_metadata = conditions[
        ["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]
    ].rename(columns={"electrolyte_concentration_M": "C_electrolyte_M"})

    alpha_pooled = alpha.pooled.merge(
        condition_metadata, on="condition_id", how="left", validate="many_to_one",
    )
    alpha_chain = alpha.by_chain.merge(
        condition_metadata, on="condition_id", how="left", validate="many_to_one",
    )

    rename_co = {
        "electrolyte_concentration_M": "C_electrolyte_M",
        "lower_CO_mole_fraction": "CO_lower_mole_fraction",
        "upper_CO_mole_fraction": "CO_upper_mole_fraction",
    }
    experimental = load_ptru_experimental_observables(
        paths, medium, materials=run.inputs.materials,
    )

    comparison_specs = {
        "alpha": (
            alpha_pooled, alpha_chain,
            ["material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index"],
            "alpha_mean", "alpha_sd",
        ),
        ion_name: (
            ion.pooled, ion.by_chain,
            ["material", "CO_mole_fraction", "analysis_grid_index"],
            ion_name, f"{ion_name}_sd",
        ),
        "delta_CO": (
            co.pooled.rename(columns=rename_co),
            co.by_chain.rename(columns=rename_co),
            [
                "material", "C_electrolyte_M", "CO_lower_mole_fraction",
                "CO_upper_mole_fraction", "analysis_grid_index",
            ],
            "delta_CO", "delta_CO_sd",
        ),
    }

    point_frames = []
    summary_rows = []

    for name, (pooled, by_chain, keys, value_column, sd_column) in comparison_specs.items():
        measured = experimental[name].copy()

        if sd_column not in measured:
            measured[sd_column] = np.nan

        comparison = build_experimental_observable_comparison(
            pooled=pooled,
            by_chain=by_chain,
            experimental=measured,
            key_columns=keys,
            observed_column=value_column,
            observed_sd_column=sd_column,
        )

        frame = comparison.pooled.copy()
        frame.insert(0, "observable", name)
        point_frames.append(frame)
        summary_rows.append(summarize_experimental_observable(name, comparison))

    return (
        pd.concat(point_frames, ignore_index=True, sort=False),
        pd.DataFrame(summary_rows),
    )
