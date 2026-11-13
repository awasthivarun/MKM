
"""Preview the configured acidic Pt100 selection rules on full Todd curves."""

import matplotlib.pyplot as plt
import pandas as pd

from mkm.preprocessing import plotting as style
from mkm.preprocessing.ptru_plotting import plot_alpha_grid, plot_rate_grid
from mkm.preprocessing.ptru_selection import select_ptru_analysis
from mkm.project_paths import ProjectPaths
from mkm.workflows.configuration import load_yaml


def find_onset_cutoffs(analysis, config):
    """Return the same cutoffs and threshold metadata used in production preprocessing."""
    _, truncation = select_ptru_analysis(analysis, config)

    cutoffs = {
        (float(row.C_electrolyte_M), float(row.CO_mole_fraction)): float(row.E_cut_V_SHE)
        for row in truncation.itertuples(index=False)
        if row.low_potential_method != "none"
    }

    table = truncation.rename(columns={
        "C_electrolyte_M": "C_HClO4_M",
        "low_potential_method": "criterion",
        "n_points_retained": "n_retained",
    }).copy()
    table["criterion"] = table["criterion"].replace({
        "fraction_of_max_rate": "relative_max_rate",
    })
    table = table[[
        "C_HClO4_M",
        "CO_mole_fraction",
        "criterion",
        "threshold_fraction",
        "E_cut_V_SHE",
        "rate_at_cut_fraction_of_max",
        "n_retained",
    ]]
    return cutoffs, table


def main():
    paths = ProjectPaths.discover(__file__)
    config = load_yaml(paths.ptru_preprocessing_config_path)

    medium = "acidic"
    dataset = paths.ptru_dataset_name(medium)
    analysis_dir = paths.ptru_analysis_dir(medium)
    figure_dir = paths.ptru_preprocessing_figure_dir(medium) / "Pt100"

    analysis = pd.read_parquet(paths.ptru_full_path(medium))
    alpha_replicates = pd.read_parquet(analysis_dir / f"{dataset}_alpha_replicates.parquet")
    alpha_summary = pd.read_parquet(analysis_dir / f"{dataset}_alpha_summary.parquet")

    analysis = analysis.loc[analysis["material"] == "Pt100"].copy()
    alpha_replicates = alpha_replicates.loc[alpha_replicates["material"] == "Pt100"].copy()
    alpha_summary = alpha_summary.loc[alpha_summary["material"] == "Pt100"].copy()

    cutoffs, table = find_onset_cutoffs(analysis, config)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table_path = figure_dir / "Pt100_second_alpha_peak_preview.csv"
    table.to_csv(table_path, index=False)

    print("\nTodd acidic Pt100: configured truncation preview")
    print(table.to_string(index=False))
    print(f"\nConfigured cutoffs: {len(cutoffs)} of {len(table)} conditions")
    print(f"Saved: {table_path}")

    figures = {
        "rate": plot_rate_grid(
            analysis, "Pt100", medium, False, config, cutoffs=cutoffs or None,
        ),
        "rate_logscale": plot_rate_grid(
            analysis, "Pt100", medium, True, config, cutoffs=cutoffs or None,
        ),
        "alpha": plot_alpha_grid(
            alpha_replicates, alpha_summary, "Pt100", medium, config, cutoffs=cutoffs or None,
        ),
    }

    for name, fig in figures.items():
        output_path = figure_dir / f"Pt100_{name}_second_alpha_peak_preview.png"
        style.save_agpd_figure(fig, output_path)
        plt.close(fig)
        print(f"Saved: {output_path} (+ SVG)")


if __name__ == "__main__":
    main()
