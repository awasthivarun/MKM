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


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "preprocessing" / "agpd_basic.yaml"
ANALYSIS_DIR = REPO_ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
FULL_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_full.parquet"
SELECTED_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_selected.parquet"
SUMMARY_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_summary.parquet"
TRUNCATION_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_truncation.parquet"
DELTA_OH_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_delta_OH.parquet"
DELTA_CO_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_delta_CO.parquet"
DELTA_CO_REPLICATES_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_delta_CO_replicates.parquet"
FIGURE_DIR = REPO_ROOT / "figures" / "preprocessing" / "AgPd_COOx_basic"


def _save_figure(name, figure):
    output_path = FIGURE_DIR / f"{name}.png"
    save_agpd_figure(figure, output_path)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.svg')}")
    plt.close(figure)


def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    full_replicates = pd.read_parquet(FULL_PATH)
    selected_replicates = pd.read_parquet(SELECTED_PATH)
    selected_summary = pd.read_parquet(SUMMARY_PATH)
    truncation = pd.read_parquet(TRUNCATION_PATH)
    delta_oh = pd.read_parquet(DELTA_OH_PATH)
    delta_co = pd.read_parquet(DELTA_CO_PATH)
    delta_co_replicates = pd.read_parquet(DELTA_CO_REPLICATES_PATH)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for material in config["materials"]:
        figures = plot_agpd_experimental_summary(
            full_replicates=full_replicates,
            selected_replicates=selected_replicates,
            selected_summary=selected_summary,
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
            _save_figure(Path(material) / figure_name, figure)

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
        _save_figure(figure_name, figure)


if __name__ == "__main__":
    main()
