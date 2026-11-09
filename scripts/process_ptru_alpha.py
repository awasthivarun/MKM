
"""Calculate and plot experimental Pt/PtRu transfer coefficients for Todd and Em."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from mkm.preprocessing import plotting as style
from mkm.preprocessing.observables import add_transfer_coefficients, summarize_transfer_coefficients

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "preprocessing" / "ptru_todd.yaml"

PT_COLORS = {
    "Pt100": "#303030",
    "Pt66Ru33": "#832C2C",
    "Pt50Ru50": "#D62728",
}


def calculate_alpha_products(analysis, temperature_K):
    """Calculate alpha per replicate, then summarize available replicates at each potential."""
    curve_keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "replicate"]
    point_keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index",
                  "E_V_SHE"]

    if "sample_id" in analysis.columns:
        curve_keys.insert(2, "sample_id")
        point_keys.insert(2, "sample_id")

    replicates = add_transfer_coefficients(
        data=analysis,
        group_columns=curve_keys,
        temperature_K=temperature_K,
        potential_column="E_V_SHE",
        log_rate_column="ln_rate",
        output_column="alpha",
    )

    summary = summarize_transfer_coefficients(
        data=replicates,
        group_columns=point_keys,
        replicate_column="replicate",
        alpha_column="alpha",
    )

    return replicates, summary


def plot_alpha_grid(replicates, summary, material, medium, config, source_label=None):
    """Plot experimental alpha using the same condition-grid presentation as PtRu rates."""
    concentrations = config["electrolyte_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    electrolyte = "HClO4" if medium == "acidic" else "KOH"
    color = style._darken(PT_COLORS[material], 0.82)

    is_em = len(concentrations) == 1
    ncols = 2 if is_em else len(concentrations)
    nrows = (len(co_values) + 1) // 2 if is_em else len(co_values)

    panels = (
        [(i // 2, i % 2, co, concentrations[0]) for i, co in enumerate(co_values)]
        if is_em
        else [
            (row, col, co, concentration)
            for row, co in enumerate(co_values)
            for col, concentration in enumerate(concentrations)
        ]
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=style._grid_figsize(
            ncols, nrows,
            panel_width=3.50 if is_em else style.GRID_PANEL_WIDTH_IN,
            panel_height=3.10 if is_em else 2.40,
            extra_width=1.00 if is_em else style.GRID_EXTRA_WIDTH_IN,
            extra_height=1.00 if is_em else style.GRID_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, col, co, concentration in panels:
        ax = axes[row, col]

        replicate_condition = replicates.loc[
            np.isclose(replicates["C_electrolyte_M"], concentration)
            & np.isclose(replicates["CO_mole_fraction"], co)
        ]
        summary_condition = summary.loc[
            np.isclose(summary["C_electrolyte_M"], concentration)
            & np.isclose(summary["CO_mole_fraction"], co)
        ].sort_values("E_V_SHE")

        if replicate_condition.empty or summary_condition.empty:
            raise ValueError(
                f"Missing alpha data: {material}, {medium}, {concentration:g} M, {100 * co:g}% CO"
            )

        # Display individual traces only when multiple replicates exist.
        if replicate_condition["replicate"].nunique() > 1:
            for _, curve in replicate_condition.groupby("replicate"):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["alpha"],
                    color=style.RETAINED_REPLICATE_COLOR,
                    linewidth=style.RETAINED_REPLICATE_LINEWIDTH,
                    alpha=0.78,
                    zorder=1,
                )

        potential = summary_condition["E_V_SHE"].to_numpy(float)
        mean = summary_condition["alpha_mean"].to_numpy(float)
        sd = summary_condition["alpha_sd"].to_numpy(float)
        replicated = summary_condition["n_replicates"].to_numpy() >= 2

        if replicated.any():
            ax.fill_between(
                potential, mean - sd, mean + sd,
                where=replicated,
                color=color,
                alpha=style.SUMMARY_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )

        ax.plot(
            potential, mean,
            color=color,
            linewidth=style.SUMMARY_LINEWIDTH,
            zorder=3,
        )

        style._style_axis(ax, potential_x=True)

        if is_em:
            ax.set_title(
                f"{100 * co:g}% CO",
                fontsize=style.FONT_SIZE_SUBPLOT_TITLE,
                fontweight=style.FONT_WEIGHT,
            )
        else:
            if row == 0:
                ax.set_title(
                    f"{concentration:g} M {electrolyte}",
                    fontsize=style.FONT_SIZE_SUBPLOT_TITLE,
                    fontweight=style.FONT_WEIGHT,
                )

            if col == ncols - 1:
                ax.text(
                    1.0, 0.5, f"{100 * co:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    fontsize=style.FONT_SIZE_SUBPLOT_TITLE,
                    fontweight=style.FONT_WEIGHT,
                )

    if is_em:
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)

    # Preserve the x-axis formatting established for the Todd basic rate figures.
    if not is_em and medium == "basic":
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        axes[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    title = f"{material}: Transfer coefficient"
    if source_label:
        title += f" - {source_label}"

    style._finish_condition_grid(
        fig, axes,
        title=title,
        ylabel="Transfer coefficient",
        left=0.11 if is_em else 0.085,
        right=0.95 if is_em else 0.92,
        bottom=0.065,
        top=0.925,
        hspace=0.16 if is_em else 0.08,
        wspace=0.28,
    )

    return fig


def process_source(analysis_path, output_prefix, figure_dir, config, medium, source_label=None):
    """Calculate, save, and plot alpha for one experimentally distinct source."""
    analysis = pd.read_parquet(analysis_path)
    replicates, summary = calculate_alpha_products(analysis, config["temperature_K"])

    replicates_path = output_prefix.with_name(output_prefix.name + "_alpha_replicates.parquet")
    summary_path = output_prefix.with_name(output_prefix.name + "_alpha_summary.parquet")

    replicates.to_parquet(replicates_path, index=False)
    summary.to_parquet(summary_path, index=False)

    print(f"\n{output_prefix.name}")
    print(f"  Alpha replicate observations: {len(replicates):,}")
    print(f"  Alpha summary observations: {len(summary):,}")
    print(f"  Points with multiple replicates: {(summary['n_replicates'] >= 2).sum():,}")
    print(f"  Saved: {replicates_path}")
    print(f"  Saved: {summary_path}")

    for material in sorted(analysis["material"].unique()):
        material_replicates = replicates.loc[replicates["material"] == material]
        material_summary = summary.loc[summary["material"] == material]

        plot_config = {
            **config,
            "electrolyte_concentrations_M": sorted(material_replicates["C_electrolyte_M"].unique()),
            "CO_mole_fractions": sorted(material_replicates["CO_mole_fraction"].unique()),
        }

        fig = plot_alpha_grid(
            material_replicates, material_summary,
            material, medium, plot_config,
            source_label=source_label,
        )

        output_dir = figure_dir if source_label is not None else figure_dir / material
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{material}_alpha.png"
        style.save_agpd_figure(fig, output_path)
        plt.close(fig)

        print(f"  Saved: {output_path}")
        print(f"  Saved: {output_path.with_suffix('.svg')}")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        analysis_dir = ROOT / "data" / "processed" / dataset / "analysis"
        figure_dir = ROOT / "figures" / "preprocessing" / dataset

        # Todd: all three catalyst compositions, retained as separate materials.
        process_source(
            analysis_path=analysis_dir / f"{dataset}_full.parquet",
            output_prefix=analysis_dir / dataset,
            figure_dir=figure_dir,
            config=config,
            medium=medium,
        )

        # Em: distinct Pt/C preparations are never combined with Todd or one another.
        for analysis_path in sorted(analysis_dir.glob(f"{dataset}_Em_*_full.parquet")):
            prefix = analysis_path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")

            process_source(
                analysis_path=analysis_path,
                output_prefix=analysis_dir / prefix,
                figure_dir=figure_dir / f"Pt100_Em_{sample_id}",
                config=config,
                medium=medium,
                source_label=f"Em - {sample_id}",
            )


if __name__ == "__main__":
    main()
