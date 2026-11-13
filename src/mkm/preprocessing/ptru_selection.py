
"""Apply configurable low-potential selection rules to Todd's Pt/PtRu analysis data."""

import numpy as np
import pandas as pd


def _matching_rule(rules, value):
    """Find a concentration or CO-fraction rule using numeric comparison."""
    for key, rule in rules.items():
        if np.isclose(float(key), float(value), rtol=0, atol=1e-12):
            return rule
    return None


def select_ptru_analysis(analysis, config):
    """Return selected Todd observations and condition-specific truncation metadata."""
    condition_columns = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction"]
    truncation = config.get("truncation", {}).get("todd", {})
    selected_frames = []
    cutoff_records = []

    for (medium, material, concentration, co_fraction), condition in analysis.groupby(
        condition_columns, sort=False
    ):
        condition = condition.sort_values(["analysis_grid_index", "replicate"]).copy()

        grid = (
            condition.groupby("analysis_grid_index", as_index=False)
            .agg(
                E_V_SHE=("E_V_SHE", "first"),
                rate_mean_s_inv=("rate_s_inv", "mean"),
            )
            .sort_values("analysis_grid_index")
        )

        potential = grid["E_V_SHE"].to_numpy(float)
        rate = grid["rate_mean_s_inv"].to_numpy(float)
        indices = grid["analysis_grid_index"].to_numpy(int)

        if not np.isfinite(rate).all() or (rate <= 0).any():
            raise ValueError(f"Invalid rates for {medium}, {material}, {concentration:g} M, {co_fraction:g} CO")

        max_rate = float(rate.max())
        material_rules = truncation.get(medium, {}).get(material, {})
        rule = _matching_rule(material_rules, co_fraction)

        method = "none" if rule is None else rule["method"]
        fraction = np.nan
        threshold_rate = np.nan
        requested_cutoff = np.nan

        if method == "none":
            cutoff_position = 0

        elif method == "fixed_potential":
            requested_cutoff = _matching_rule(
                rule["E_cut_V_SHE_by_electrolyte_M"], concentration
            )
            if requested_cutoff is None:
                raise ValueError(f"Missing fixed-potential cutoff for {concentration:g} M")

            requested_cutoff = float(requested_cutoff)
            crossings = np.flatnonzero(potential >= requested_cutoff - 1e-10)

            if len(crossings) == 0:
                raise ValueError(f"Fixed cutoff removes all observations: {medium}, {material}")

            cutoff_position = int(crossings[0])

        elif method == "fraction_of_max_rate":
            fraction = float(rule["fraction"])
            if not 0 < fraction <= 1:
                raise ValueError(f"Invalid relative TOF threshold: {fraction}")

            threshold_rate = fraction * max_rate
            cutoff_position = int(np.flatnonzero(rate >= threshold_rate)[0])

        else:
            raise ValueError(f"Unsupported PtRu truncation method: {method}")

        cutoff_index = int(indices[cutoff_position])
        cutoff_potential = float(potential[cutoff_position])
        rate_at_cutoff = float(rate[cutoff_position])

        retained = condition.loc[
            condition["analysis_grid_index"] >= cutoff_index
        ].copy()

        n_retained_points = int(np.count_nonzero(indices >= cutoff_index))
        if n_retained_points < 3:
            raise ValueError(
                f"Fewer than three retained potential points: "
                f"{medium}, {material}, {concentration:g} M, {co_fraction:g} CO"
            )

        selected_frames.append(retained)

        cutoff_records.append({
            "medium": medium,
            "material": material,
            "C_electrolyte_M": concentration,
            "CO_mole_fraction": co_fraction,
            "low_potential_method": method,
            "threshold_fraction": fraction,
            "threshold_rate_s_inv": threshold_rate,
            "requested_E_cut_V_SHE": requested_cutoff,
            "low_cut_analysis_grid_index": cutoff_index,
            "E_cut_V_SHE": cutoff_potential,
            "original_min_E_V_SHE": float(potential[0]),
            "original_max_E_V_SHE": float(potential[-1]),
            "retained_min_E_V_SHE": cutoff_potential,
            "retained_max_E_V_SHE": float(potential[-1]),
            "peak_rate_mean_s_inv": max_rate,
            "rate_at_cut_fraction_of_max": rate_at_cutoff / max_rate,
            "n_points_original": len(grid),
            "n_points_retained": n_retained_points,
            "n_observations_original": len(condition),
            "n_observations_retained": len(retained),
        })

    selected = pd.concat(selected_frames, ignore_index=True)
    cutoffs = pd.DataFrame(cutoff_records)

    return selected, cutoffs
