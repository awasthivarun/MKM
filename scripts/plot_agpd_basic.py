from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from mkm.preprocessing.plotting import (
    plot_agpd_co_order,
    plot_agpd_experimental_summary,
    plot_agpd_oh_order,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "preprocessing"
    / "agpd_basic.yaml"
)

ANALYSIS_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "AgPd_COOx_basic"
    / "analysis"
)

FULL_PATH = (
    ANALYSIS_DIR
    / "AgPd_COOx_basic_full.parquet"
)

SELECTED_PATH = (
    ANALYSIS_DIR
    / "AgPd_COOx_basic_selected.parquet"
)

SUMMARY_PATH = (
    ANALYSIS_DIR
    / "AgPd_COOx_basic_summary.parquet"
)

TRUNCATION_PATH = (
    ANALYSIS_DIR
    / "AgPd_COOx_basic_truncation.parquet"
)

DELTA_OH_PATH = (
    ANALYSIS_DIR
    / "AgPd_COOx_basic_delta_OH.parquet"
)

DELTA_CO_PATH = (
    ANALYSIS_DIR
    / "AgPd_COOx_basic_delta_CO.parquet"
)

FIGURE_DIR = (
    REPO_ROOT
    / "figures"
    / "preprocessing"
    / "AgPd_COOx_basic"
)


def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    full_replicates = pd.read_parquet(
        FULL_PATH
    )

    selected_replicates = pd.read_parquet(
        SELECTED_PATH
    )

    selected_summary = pd.read_parquet(
        SUMMARY_PATH
    )

    truncation = pd.read_parquet(
        TRUNCATION_PATH
    )

    delta_OH = pd.read_parquet(
        DELTA_OH_PATH
    )

    delta_CO = pd.read_parquet(
        DELTA_CO_PATH
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for material in config["materials"]:
        figures = plot_agpd_experimental_summary(
            full_replicates=full_replicates,
            selected_replicates=selected_replicates,
            selected_summary=selected_summary,
            truncation=truncation,
            material=material,
            config=config,
        )

        figures["delta_OH"] = (
            plot_agpd_oh_order(
                delta_OH=delta_OH,
                material=material,
                config=config,
            )
        )

        figures["delta_CO"] = (
            plot_agpd_co_order(
                delta_CO=delta_CO,
                material=material,
                config=config,
            )
        )
        
        for figure_name, figure in figures.items():
            output_path = (
                FIGURE_DIR
                / f"{material}_{figure_name}.png"
            )

            figure.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )

            print(
                f"Saved: {output_path}"
            )

            plt.close(figure)


if __name__ == "__main__":
    main()