
"""Preview Todd acidic Pt100 truncation at the second prominent alpha maximum."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

from mkm.preprocessing import plotting as style
from mkm.preprocessing.ptru_plotting import plot_alpha_grid, plot_rate_grid

ROOT = Path(__file__).resolve().parents[1]
DATASET = "PtRu_COOx_acidic"
ANALYSIS_DIR = ROOT / "data" / "processed" / DATASET / "analysis"
FIGURE_DIR = ROOT / "figures" / "preprocessing" / DATASET / "Pt100"
CONFIG_PATH = ROOT / "config" / "preprocessing" / "ptru_todd.yaml"

# Diagnostic peak-identification settings, not finalized selection parameters.
MIN_PROMINENCE = 0.20
MIN_PEAK_DISTANCE_POINTS = 5

def find_onset_cutoffs(analysis, config):
    """Determine CO-dependent low-potential cutoffs for Todd acidic Pt100."""
    thresholds = {0.01: 0.10, 0.10: 0.05, 1.00: 0.05}
    cutoffs = {}
    records = []

    for concentration in config["electrolyte_concentrations_M"]:
        for co_fraction in config["CO_mole_fractions"]:
            condition = analysis.loc[
                np.isclose(analysis["C_electrolyte_M"], concentration)
                & np.isclose(analysis["CO_mole_fraction"], co_fraction)
            ]

            summary = (
                condition.groupby("analysis_grid_index", as_index=False)
                .agg(E_V_SHE=("E_V_SHE", "first"), rate_s_inv=("rate_s_inv", "mean"))
                .sort_values("E_V_SHE")
            )

            if summary.empty:
                raise ValueError(f"Missing data: {concentration:g} M, {100 * co_fraction:g}% CO")

            potential = summary["E_V_SHE"].to_numpy(float)
            rate = summary["rate_s_inv"].to_numpy(float)
            max_rate = rate.max()

            if np.isclose(co_fraction, 0.001):
                cutoff = 0.75 if np.isclose(concentration, [0.25, 1.00]).any() else 0.70
                criterion = "fixed_potential"
                fraction = np.nan
            else:
                fraction = thresholds[float(co_fraction)]
                cutoff = float(potential[np.flatnonzero(rate >= fraction * max_rate)[0]])
                criterion = "relative_max_rate"

            cutoffs[(float(concentration), float(co_fraction))] = cutoff

            rate_at_cutoff = np.interp(cutoff, potential, rate)
            records.append({
                "C_HClO4_M": concentration,
                "CO_mole_fraction": co_fraction,
                "criterion": criterion,
                "threshold_fraction": fraction,
                "E_cut_V_SHE": cutoff,
                "rate_at_cut_fraction_of_max": rate_at_cutoff / max_rate,
                "n_retained": int(np.count_nonzero(potential >= cutoff)),
            })

    return cutoffs, pd.DataFrame(records)

def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    analysis = pd.read_parquet(ANALYSIS_DIR / f"{DATASET}_full.parquet")
    alpha_replicates = pd.read_parquet(ANALYSIS_DIR / f"{DATASET}_alpha_replicates.parquet")
    alpha_summary = pd.read_parquet(ANALYSIS_DIR / f"{DATASET}_alpha_summary.parquet")

    analysis = analysis.loc[analysis["material"] == "Pt100"].copy()
    alpha_replicates = alpha_replicates.loc[alpha_replicates["material"] == "Pt100"].copy()
    alpha_summary = alpha_summary.loc[alpha_summary["material"] == "Pt100"].copy()

    cutoffs, table = find_onset_cutoffs(analysis, config)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    table_path = FIGURE_DIR / "Pt100_second_alpha_peak_preview.csv"
    table.to_csv(table_path, index=False)

    print("\nTodd acidic Pt100: second-alpha-peak truncation preview")
    print(table.to_string(index=False))
    print(f"\nIdentified cutoffs: {len(cutoffs)} of {len(table)} conditions")
    print(f"Saved: {table_path}")

    figures = {
        "rate": plot_rate_grid(
            analysis, "Pt100", "acidic", False, config, cutoffs=cutoffs,
        ),
        "rate_logscale": plot_rate_grid(
            analysis, "Pt100", "acidic", True, config, cutoffs=cutoffs,
        ),
        "alpha": plot_alpha_grid(
            alpha_replicates, alpha_summary, "Pt100", "acidic", config, cutoffs=cutoffs,
        ),
    }

    for name, fig in figures.items():
        output_path = FIGURE_DIR / f"Pt100_{name}_second_alpha_peak_preview.png"
        style.save_agpd_figure(fig, output_path)
        plt.close(fig)
        print(f"Saved: {output_path} (+ SVG)")


if __name__ == "__main__":
    main()
