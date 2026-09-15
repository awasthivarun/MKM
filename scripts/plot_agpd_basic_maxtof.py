"""Plot the AgPd basic-media maximum-TOF-truncated preprocessing products."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from mkm.preprocessing.plotting import (
    plot_agpd_alpha_overlay_koh,
    plot_agpd_alpha_overlay_pco,
    plot_agpd_co_order,
    plot_agpd_co_order_overlay_koh,
    plot_agpd_co_order_overlay_pco,
    plot_agpd_experimental_summary,
    plot_agpd_oh_order,
    plot_agpd_oh_order_overlay_pco,
    plot_agpd_rate_overlay_koh,
    plot_agpd_rate_overlay_pco,
    plot_agpd_second_order_composition,
    plot_agpd_second_order_difference,
    plot_agpd_second_order_overlay_koh,
    plot_agpd_second_order_overlay_pco,
    save_agpd_figure,
)
from mkm.project_paths import ProjectPaths


_UPPER_CUTOFF_FIGURES = ("rate", "rate_logscale", "alpha")


def _save_figure(root, name, figure):
    output_path = root / f"{name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_agpd_figure(figure, output_path)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.svg')}")
    plt.close(figure)


def _add_upper_truncation_lines(figures, truncation, material, config):
    """Mark condition-specific maximum-TOF cutoffs on the same grids that show the low cutoff."""

    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    expected_axes = len(koh_values) * len(co_values)

    for figure_name in _UPPER_CUTOFF_FIGURES:
        figure = figures[figure_name]
        if len(figure.axes) != expected_axes:
            raise ValueError(
                f"Expected {expected_axes} axes in {figure_name}, found {len(figure.axes)}."
            )

        for row, co_fraction in enumerate(co_values):
            for col, koh_m in enumerate(koh_values):
                rows = truncation[
                    (truncation["material"] == material)
                    & (truncation["C_KOH_M"] == koh_m)
                    & (truncation["CO_mole_fraction"] == co_fraction)
                ]
                if len(rows) != 1:
                    raise ValueError(
                        f"Expected one truncation row for {material}, {koh_m:g} M KOH, "
                        f"{100 * co_fraction:g}% CO; found {len(rows)}."
                    )

                cutoff = rows.iloc[0]
                retained_max = float(cutoff["retained_max_E_V_SHE"])
                original_max = float(cutoff["original_max_E_V_SHE"])
                if retained_max >= original_max - 1.0e-12:
                    continue

                axis = figure.axes[row * len(koh_values) + col]
                axis.axvline(retained_max, linestyle="--", linewidth=1.0, alpha=0.6)


def main():
    paths = ProjectPaths.discover(__file__).with_agpd_data_variant("maxtof")
    with open(paths.agpd_preprocessing_config_path, "r") as file:
        config = yaml.safe_load(file)

    full_replicates = pd.read_parquet(paths.agpd_full_path)
    selected_replicates = pd.read_parquet(paths.agpd_selected_path)
    selected_summary = pd.read_parquet(paths.agpd_summary_path)
    truncation = pd.read_parquet(paths.agpd_truncation_path)
    delta_oh = pd.read_parquet(paths.agpd_delta_oh_path)
    delta_co = pd.read_parquet(paths.agpd_delta_co_path)
    delta_co_replicates = pd.read_parquet(paths.agpd_delta_co_replicates_path)

    figure_root = paths.agpd_preprocessing_figure_dir
    figure_root.mkdir(parents=True, exist_ok=True)

    for material in config["materials"]:
        figures = plot_agpd_experimental_summary(
            full_replicates=full_replicates,
            selected_replicates=selected_replicates,
            selected_summary=selected_summary,
            truncation=truncation,
            material=material,
            config=config,
        )
        _add_upper_truncation_lines(
            figures=figures,
            truncation=truncation,
            material=material,
            config=config,
        )
        figures["delta_OH"] = plot_agpd_oh_order(delta_OH=delta_oh, material=material, config=config)
        figures["second_order_difference"] = plot_agpd_second_order_difference(
            selected_summary=selected_summary,
            delta_OH=delta_oh,
            material=material,
            config=config,
        )
        figures["delta_CO"] = plot_agpd_co_order(
            delta_CO_replicates=delta_co_replicates,
            delta_CO=delta_co,
            material=material,
            config=config,
        )

        for figure_name, figure in figures.items():
            _save_figure(figure_root, Path(material) / figure_name, figure)

    overlay_figures = {
        "alpha_overlay_PCO": plot_agpd_alpha_overlay_pco(selected_summary, config),
        "alpha_overlay_KOH": plot_agpd_alpha_overlay_koh(selected_summary, config),
        "delta_CO_overlay_PCO": plot_agpd_co_order_overlay_pco(delta_co, config),
        "delta_CO_overlay_KOH": plot_agpd_co_order_overlay_koh(delta_co, config),
        "delta_OH_overlay_PCO": plot_agpd_oh_order_overlay_pco(delta_oh, config),
        "rate_overlay_PCO": plot_agpd_rate_overlay_pco(selected_summary, config),
        "rate_overlay_KOH": plot_agpd_rate_overlay_koh(selected_summary, config),
        "second_order_overlay_PCO": plot_agpd_second_order_overlay_pco(selected_summary, delta_oh, config),
        "second_order_overlay_KOH": plot_agpd_second_order_overlay_koh(selected_summary, delta_oh, config),
        "second_order_composition": plot_agpd_second_order_composition(selected_summary, delta_oh, config),
    }
    for figure_name, figure in overlay_figures.items():
        _save_figure(figure_root, figure_name, figure)


if __name__ == "__main__":
    main()
