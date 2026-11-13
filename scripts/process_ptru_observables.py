
"""Calculate and plot Pt/PtRu experimental kinetic observables for Todd and Em."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from mkm.preprocessing import plotting as style
from mkm.preprocessing.ptru_observables import (
    calculate_alpha_products,
    calculate_co_orders,
    calculate_ion_orders,
)
from mkm.preprocessing.ptru_plotting import (
    plot_alpha_grid,
    plot_co_order_grid,
    plot_ion_order_grid,
)

from argparse import ArgumentParser

from mkm.project_paths import ProjectPaths
from mkm.workflows.configuration import load_yaml


def save_figure(fig, output_path):
    """Save a figure using the existing AgPd PNG/SVG convention."""
    style.save_agpd_figure(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path} (+ SVG)")



def process_source(
    analysis_path, figure_dir, config, medium, source_label=None, generate_figures=True,
):
    """Calculate full and selected observables, preserving full-curve alpha derivatives."""
    analysis = pd.read_parquet(analysis_path)
    prefix = analysis_path.name.removesuffix("_full.parquet")
    is_todd = source_label is None

    selected = analysis
    if is_todd:
        selected = pd.read_parquet(analysis_path.with_name(f"{prefix}_selected.parquet"))

    print(f"\n{prefix}")

    # Transfer coefficients: differentiate full curves before selection
    alpha_replicates, alpha_summary = calculate_alpha_products(analysis, config["temperature_K"])

    alpha_replicates.to_parquet(
        analysis_path.with_name(f"{prefix}_alpha_replicates.parquet"), index=False,
    )
    alpha_summary.to_parquet(
        analysis_path.with_name(f"{prefix}_alpha_summary.parquet"), index=False,
    )

    alpha_replicates_selected = alpha_replicates
    alpha_summary_selected = alpha_summary

    if is_todd:
        keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index"]
        selected_grid = selected[keys].drop_duplicates()

        alpha_replicates_selected = alpha_replicates.merge(selected_grid, on=keys, how="inner")
        alpha_summary_selected = alpha_summary.merge(selected_grid, on=keys, how="inner")

        alpha_replicates_selected.to_parquet(
            analysis_path.with_name(f"{prefix}_alpha_replicates_selected.parquet"), index=False,
        )
        alpha_summary_selected.to_parquet(
            analysis_path.with_name(f"{prefix}_alpha_summary_selected.parquet"), index=False,
        )

    print(f"  Full alpha observations: {len(alpha_replicates):,}")
    print(f"  Selected alpha observations: {len(alpha_replicates_selected):,}")

    # CO reaction orders: calculate from full and selected rates
    co_orders_full = calculate_co_orders(analysis)
    co_orders_full.to_parquet(
        analysis_path.with_name(f"{prefix}_delta_CO.parquet"), index=False,
    )

    co_orders_selected = co_orders_full
    if is_todd:
        co_orders_selected = calculate_co_orders(selected)
        co_orders_selected.to_parquet(
            analysis_path.with_name(f"{prefix}_delta_CO_selected.parquet"), index=False,
        )

    print(f"  Full CO orders: {len(co_orders_full):,}")
    print(f"  Selected CO orders: {len(co_orders_selected):,}")

    # Electrolyte orders: Todd only
    ion_orders_selected = None

    if is_todd:
        ion_name = "delta_Hplus" if medium == "acidic" else "delta_OH"
        concentrations = config["electrolyte_concentrations_M"]

        ion_orders_full = calculate_ion_orders(analysis, concentrations)
        ion_orders_full.to_parquet(
            analysis_path.with_name(f"{prefix}_{ion_name}.parquet"), index=False,
        )

        ion_orders_selected = calculate_ion_orders(selected, concentrations)
        ion_orders_selected.to_parquet(
            analysis_path.with_name(f"{prefix}_{ion_name}_selected.parquet"), index=False,
        )

        print(f"  Full {ion_name} orders: {len(ion_orders_full):,}")
        print(f"  Selected {ion_name} orders: {len(ion_orders_selected):,}")

    
    if generate_figures:
        plot_source(
            analysis_path=analysis_path,
            figure_dir=figure_dir,
            config=config,
            medium=medium,
            source_label=source_label,
            alpha_replicates=alpha_replicates_selected,
            alpha_summary=alpha_summary_selected,
            co_orders=co_orders_selected,
            ion_orders=ion_orders_selected,
        )


def plot_source(
    analysis_path,
    figure_dir,
    config,
    medium,
    source_label=None,
    alpha_replicates=None,
    alpha_summary=None,
    co_orders=None,
    ion_orders=None,
):
    """Render saved or in-memory observable products without recalculating them."""
    prefix = analysis_path.name.removesuffix("_full.parquet")
    is_todd = source_label is None
    suffix = "_selected" if is_todd else ""

    if alpha_replicates is None:
        alpha_replicates = pd.read_parquet(
            analysis_path.with_name(f"{prefix}_alpha_replicates{suffix}.parquet")
        )
    if alpha_summary is None:
        alpha_summary = pd.read_parquet(
            analysis_path.with_name(f"{prefix}_alpha_summary{suffix}.parquet")
        )
    if co_orders is None:
        co_orders = pd.read_parquet(
            analysis_path.with_name(f"{prefix}_delta_CO{suffix}.parquet")
        )

    ion_name = "delta_Hplus" if medium == "acidic" else "delta_OH"
    if is_todd and ion_orders is None:
        ion_orders = pd.read_parquet(
            analysis_path.with_name(f"{prefix}_{ion_name}_selected.parquet")
        )

    # Use selected observables only; excluded regions remain absent.
    for material in sorted(alpha_replicates["material"].unique()):
        output_dir = figure_dir / material if is_todd else figure_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        material_replicates = alpha_replicates.loc[
            alpha_replicates["material"] == material
        ]
        material_summary = alpha_summary.loc[
            alpha_summary["material"] == material
        ]

        plot_config = {
            **config,
            "electrolyte_concentrations_M": sorted(material_replicates["C_electrolyte_M"].unique()),
            "CO_mole_fractions": sorted(material_replicates["CO_mole_fraction"].unique()),
        }

        fig = plot_alpha_grid(
            material_replicates, material_summary, material, medium, plot_config,
            source_label=source_label,
        )
        save_figure(fig, output_dir / f"{material}_alpha.png")

        fig = plot_co_order_grid(
            co_orders, material, medium, source_label=source_label,
        )
        save_figure(fig, output_dir / f"{material}_delta_CO.png")

        if is_todd and ion_orders is not None:
            fig = plot_ion_order_grid(ion_orders, material, medium)
            save_figure(fig, output_dir / f"{material}_{ion_name}.png")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Generate observable tables only; plot them separately afterward.",
    )
    args = parser.parse_args()

    paths = ProjectPaths.discover(__file__)
    config = load_yaml(paths.ptru_preprocessing_config_path)

    for medium in ("acidic", "basic"):
        dataset = paths.ptru_dataset_name(medium)
        analysis_dir = paths.ptru_analysis_dir(medium)
        figure_dir = paths.ptru_preprocessing_figure_dir(medium)

        process_source(
            analysis_path=paths.ptru_full_path(medium),
            figure_dir=figure_dir,
            config=config,
            medium=medium,
            generate_figures=not args.skip_plots,
        )

        for analysis_path in sorted(analysis_dir.glob(f"{dataset}_Em_*_full.parquet")):
            prefix = analysis_path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")

            process_source(
                analysis_path=analysis_path,
                figure_dir=figure_dir / f"Pt100_Em_{sample_id}",
                config=config,
                medium=medium,
                source_label=f"Em - {sample_id}",
                generate_figures=not args.skip_plots,
            )



if __name__ == "__main__":
    main()
