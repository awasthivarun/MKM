
"""Calculate and plot unpaired adjacent-pressure CO orders for Todd and Em."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator

from mkm.preprocessing import plotting as style
from mkm.preprocessing.observables import calculate_adjacent_log_orders

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Pt100": "#303030", "Pt66Ru33": "#832C2C", "Pt50Ru50": "#D62728"}


def calculate_co_orders(analysis):
    """Estimate orders from mean rates, with independent replicate-variation propagation."""
    keys = ["material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index"]

    if analysis.duplicated([*keys, "replicate"]).any():
        raise ValueError("Duplicate replicate observations")
    if (analysis["rate_s_inv"] <= 0).any() or not np.isfinite(analysis["ln_rate"]).all():
        raise ValueError("CO orders require positive, finite rates")
    if "sample_id" in analysis.columns and analysis["sample_id"].nunique() != 1:
        raise ValueError("Process Em sample preparations separately")

    summary = (
        analysis.groupby(keys, as_index=False)
        .agg(
            E_V_SHE=("E_V_SHE", "first"),
            rate_mean_s_inv=("rate_s_inv", "mean"),
            rate_sd_s_inv=("rate_s_inv", "std"),
            ln_rate_sd=("ln_rate", "std"),
            n_replicates=("replicate", "nunique"),
        )
    )
    summary["ln_rate_mean"] = np.log(summary["rate_mean_s_inv"])

    orders = calculate_adjacent_log_orders(
        data=summary,
        varying_column="CO_mole_fraction",
        varying_values=sorted(analysis["CO_mole_fraction"].unique()),
        group_columns=["material", "C_electrolyte_M"],
        output_column="delta_CO",
        output_sd_column="delta_CO_sd",
        lower_value_column="CO_lower_mole_fraction",
        upper_value_column="CO_upper_mole_fraction",
    )
    orders.insert(0, "medium", analysis["medium"].iloc[0])

    # Retain the original per-condition sample counts and rate summaries.
    for side in ("lower", "upper"):
        co_col = f"CO_{side}_mole_fraction"
        stats = summary.rename(columns={
            "CO_mole_fraction": co_col,
            "n_replicates": f"n_{side}",
            "rate_mean_s_inv": f"rate_{side}_mean_s_inv",
            "rate_sd_s_inv": f"rate_{side}_sd_s_inv",
        })
        join_keys = ["material", "C_electrolyte_M", "analysis_grid_index", co_col]
        columns = [*join_keys, f"n_{side}", f"rate_{side}_mean_s_inv", f"rate_{side}_sd_s_inv"]
        orders = orders.merge(stats[columns], on=join_keys, validate="one_to_one")

    if "sample_id" in analysis.columns:
        orders["sample_id"] = analysis["sample_id"].iloc[0]

    if orders.empty or not np.isfinite(orders["delta_CO"]).all():
        raise ValueError("No valid CO orders calculated")
    return orders


def plot_co_order_grid(orders, material, medium, source_label=None):
    data = orders.loc[orders["material"] == material]
    concentrations = sorted(data["C_electrolyte_M"].unique())
    pairs = list(
        data[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]]
        .drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"])
        .itertuples(index=False, name=None)
    )
    color = style._darken(COLORS[material], 0.82)
    electrolyte = "HClO4" if medium == "acidic" else "KOH"
    is_em = len(concentrations) == 1

    ncols = 2 if is_em else len(concentrations)
    nrows = (len(pairs) + 1) // 2 if is_em else len(pairs)
    panels = (
        [(i // 2, i % 2, pair, concentrations[0]) for i, pair in enumerate(pairs)]
        if is_em
        else [(i, j, pair, conc) for i, pair in enumerate(pairs) for j, conc in enumerate(concentrations)]
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=style._grid_figsize(
            ncols, nrows,
            panel_width=3.5 if is_em else style.GRID_PANEL_WIDTH_IN,
            panel_height=2.4,
            extra_width=1.0 if is_em else style.GRID_EXTRA_WIDTH_IN,
            extra_height=style.GRID_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=False, squeeze=False,
    )

    for row, col, (lower_co, upper_co), concentration in panels:
        ax = axes[row, col]
        condition = data.loc[
            np.isclose(data["C_electrolyte_M"], concentration)
            & np.isclose(data["CO_lower_mole_fraction"], lower_co)
            & np.isclose(data["CO_upper_mole_fraction"], upper_co)
        ].sort_values("analysis_grid_index")

        if condition.empty:
            raise ValueError(f"Missing CO-order panel for {material}, {concentration:g} M")

        e = condition["E_V_SHE"].to_numpy(float)
        mean = condition["delta_CO"].to_numpy(float)
        sd = condition["delta_CO_sd"].to_numpy(float)
        indices = condition["analysis_grid_index"].to_numpy()

        # Do not connect disjoint potential ranges.
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        for segment in np.split(np.arange(len(e)), breaks):
            valid_sd = np.isfinite(sd[segment])
            if valid_sd.any():
                ax.fill_between(
                    e[segment], mean[segment] - sd[segment], mean[segment] + sd[segment],
                    where=valid_sd, color=color, alpha=style.SUMMARY_BAND_ALPHA,
                    linewidth=0, zorder=2,
                )
            ax.plot(
                e[segment], mean[segment],
                color=color, linewidth=style.SUMMARY_LINEWIDTH, zorder=3,
            )

        ax.axhline(
            0, color=style.REFERENCE_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.65, zorder=1,
        )
        style._style_axis(ax, potential_x=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        style._bold_axis_text(ax)

        pair_label = f"{100 * lower_co:g}% to {100 * upper_co:g}% CO"
        if is_em:
            ax.set_title(pair_label, fontsize=style.FONT_SIZE_SUBPLOT_TITLE, fontweight=style.FONT_WEIGHT)
        else:
            if row == 0:
                ax.set_title(
                    f"{concentration:g} M {electrolyte}",
                    fontsize=style.FONT_SIZE_SUBPLOT_TITLE, fontweight=style.FONT_WEIGHT,
                )
            if col == ncols - 1:
                ax.text(
                    1.0, 0.5, pair_label, transform=ax.transAxes,
                    rotation=-90, va="center", fontsize=style.FONT_SIZE_SUBPLOT_TITLE,
                    fontweight=style.FONT_WEIGHT,
                )

    if is_em:
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)

    if not is_em and medium == "basic":
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        axes[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    title = f"{material}: CO reaction order"
    if source_label:
        title += f" - {source_label}"

    style._finish_condition_grid(
        fig, axes, title=title, ylabel="CO reaction order",
        left=0.11 if is_em else 0.085,
        right=0.95 if is_em else 0.92,
        bottom=0.065, top=0.925,
        hspace=0.16 if is_em else 0.08, wspace=0.28,
    )
    return fig


def process_source(analysis_path, figure_dir, source_label=None):
    analysis = pd.read_parquet(analysis_path)
    medium = analysis["medium"].iloc[0]
    orders = calculate_co_orders(analysis)

    prefix = analysis_path.name.removesuffix("_full.parquet")
    parquet_path = analysis_path.with_name(f"{prefix}_delta_CO.parquet")
    orders.to_parquet(parquet_path, index=False)

    n_bands = int(orders["delta_CO_sd"].notna().sum())
    print(f"\n{prefix}: {len(orders):,} CO orders, {n_bands:,} with replicate uncertainty")
    print(f"Saved: {parquet_path}")

    for material in sorted(orders["material"].unique()):
        fig = plot_co_order_grid(orders, material, medium, source_label=source_label)
        output_dir = figure_dir if source_label is not None else figure_dir / material
        output_path = output_dir / f"{material}_delta_CO.png"

        style.save_agpd_figure(fig, output_path)
        plt.close(fig)
        print(f"Saved: {output_path} (+ SVG)")


def main():
    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        base = ROOT / "data" / "processed" / dataset / "analysis"
        figures = ROOT / "figures" / "preprocessing" / dataset

        process_source(base / f"{dataset}_full.parquet", figures)

        for path in sorted(base.glob(f"{dataset}_Em_*_full.parquet")):
            prefix = path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")
            process_source(
                path, figures / f"Pt100_Em_{sample_id}",
                source_label=f"Em - {sample_id}",
            )


if __name__ == "__main__":
    main()
