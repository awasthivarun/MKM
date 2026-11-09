
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
    """Process one source without combining Todd and Em or distinct Em preparations."""
    analysis = pd.read_parquet(analysis_path)
    prefix = analysis_path.name.removesuffix("_full.parquet")
    is_todd = source_label is None

    # -----------------------------------------------------------------
    # Apparent transfer coefficients
    # -----------------------------------------------------------------
    alpha_replicates, alpha_summary = calculate_alpha_products(
        analysis, config["temperature_K"]
    )

    alpha_replicates_path = analysis_path.with_name(f"{prefix}_alpha_replicates.parquet")
    alpha_summary_path = analysis_path.with_name(f"{prefix}_alpha_summary.parquet")

    alpha_replicates.to_parquet(alpha_replicates_path, index=False)
    alpha_summary.to_parquet(alpha_summary_path, index=False)

    print(f"\n{prefix}")
    print(f"  Alpha replicate observations: {len(alpha_replicates):,}")
    print(f"  Alpha summary observations: {len(alpha_summary):,}")
    print(f"  Saved: {alpha_replicates_path}")
    print(f"  Saved: {alpha_summary_path}")

    # -----------------------------------------------------------------
    # Adjacent-pressure CO reaction orders
    # -----------------------------------------------------------------
    co_orders = calculate_co_orders(analysis)
    co_orders_path = analysis_path.with_name(f"{prefix}_delta_CO.parquet")
    co_orders.to_parquet(co_orders_path, index=False)

    print(f"  CO-order observations: {len(co_orders):,}")
    print(f"  CO-order points with SD: {co_orders['delta_CO_sd'].notna().sum():,}")
    print(f"  Saved: {co_orders_path}")

    # -----------------------------------------------------------------
    # Full-range electrolyte reaction orders: Todd only
    # -----------------------------------------------------------------
    ion_orders = None
    if is_todd:
        ion_orders = calculate_ion_orders(
            analysis, config["electrolyte_concentrations_M"]
        )
        ion_name = "delta_Hplus" if medium == "acidic" else "delta_OH"
        ion_orders_path = analysis_path.with_name(f"{prefix}_{ion_name}.parquet")
        ion_orders.to_parquet(ion_orders_path, index=False)

        print(f"  {ion_name} observations: {len(ion_orders):,}")
        print(f"  {ion_name} points with SD: {ion_orders[f'{ion_name}_sd'].notna().sum():,}")
        print(f"  Saved: {ion_orders_path}")

    # -----------------------------------------------------------------
    # Figures: retain the existing Todd and Em directory structures
    # -----------------------------------------------------------------
    for material in sorted(analysis["material"].unique()):
        output_dir = figure_dir / material if is_todd else figure_dir

        material_replicates = alpha_replicates.loc[alpha_replicates["material"] == material]
        material_summary = alpha_summary.loc[alpha_summary["material"] == material]

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

        if ion_orders is not None:
            fig = plot_ion_order_grid(ion_orders, material, medium)
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
