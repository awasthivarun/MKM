
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

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "preprocessing" / "ptru_todd.yaml"


def save_figure(fig, output_path):
    """Save a figure using the existing AgPd PNG/SVG convention."""
    style.save_agpd_figure(fig, output_path)
    plt.close(fig)
    print(f"  Saved: {output_path} (+ SVG)")


def process_source(analysis_path, figure_dir, config, medium, source_label=None):
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

    # Figures: selected observables only, without truncated regions
    for material in sorted(analysis["material"].unique()):
        output_dir = figure_dir / material if is_todd else figure_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        material_replicates = alpha_replicates_selected.loc[
            alpha_replicates_selected["material"] == material
        ]
        material_summary = alpha_summary_selected.loc[
            alpha_summary_selected["material"] == material
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
            co_orders_selected, material, medium, source_label=source_label,
        )
        save_figure(fig, output_dir / f"{material}_delta_CO.png")

        if ion_orders_selected is not None:
            fig = plot_ion_order_grid(ion_orders_selected, material, medium)
            save_figure(fig, output_dir / f"{material}_{ion_name}.png")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        analysis_dir = ROOT / "data" / "processed" / dataset / "analysis"
        figure_dir = ROOT / "figures" / "preprocessing" / dataset

        # Todd: Pt100, Pt66Ru33, Pt50Ru50.
        process_source(
            analysis_path=analysis_dir / f"{dataset}_full.parquet",
            figure_dir=figure_dir,
            config=config,
            medium=medium,
        )

        # Em: Pt/C preparations remain separate from Todd and each other.
        for analysis_path in sorted(analysis_dir.glob(f"{dataset}_Em_*_full.parquet")):
            prefix = analysis_path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")

            process_source(
                analysis_path=analysis_path,
                figure_dir=figure_dir / f"Pt100_Em_{sample_id}",
                config=config,
                medium=medium,
                source_label=f"Em - {sample_id}",
            )


if __name__ == "__main__":
    main()
