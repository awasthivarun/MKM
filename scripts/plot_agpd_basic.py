import matplotlib.pyplot as plt
import pandas as pd
import yaml

from mkm.preprocessing.plotting import (
    plot_agpd_co_order,
    plot_agpd_experimental_summary,
    plot_agpd_oh_order,
)
from mkm.project_paths import ProjectPaths


def main():
    paths = ProjectPaths.discover(__file__)
    with open(paths.agpd_preprocessing_config_path, "r") as file:
        config = yaml.safe_load(file)

    full_replicates = pd.read_parquet(paths.agpd_full_path)
    selected_replicates = pd.read_parquet(paths.agpd_selected_path)
    selected_summary = pd.read_parquet(paths.agpd_summary_path)
    truncation = pd.read_parquet(paths.agpd_truncation_path)
    delta_OH = pd.read_parquet(paths.agpd_delta_oh_path)
    delta_CO = pd.read_parquet(paths.agpd_delta_co_path)
    delta_CO_replicates = pd.read_parquet(paths.agpd_delta_co_replicates_path)

    paths.agpd_preprocessing_figure_dir.mkdir(parents=True, exist_ok=True)

    for material in config["materials"]:
        figures = plot_agpd_experimental_summary(
            full_replicates=full_replicates,
            selected_replicates=selected_replicates,
            selected_summary=selected_summary,
            truncation=truncation,
            material=material,
            config=config,
        )
        figures["delta_OH"] = plot_agpd_oh_order(
            delta_OH=delta_OH,
            material=material,
            config=config,
        )
        figures["delta_CO"] = plot_agpd_co_order(
            delta_CO_replicates=delta_CO_replicates,
            delta_CO=delta_CO,
            material=material,
            config=config,
        )

        for figure_name, figure in figures.items():
            output_path = (
                paths.agpd_preprocessing_figure_dir
                / f"{material}_{figure_name}.png"
            )
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_path}")
            plt.close(figure)


if __name__ == "__main__":
    main()
