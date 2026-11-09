
"""Calculate full-range H+/OH- orders with propagated unpaired replicate variability."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator

from mkm.preprocessing import plotting as style
from mkm.preprocessing.observables import calculate_log_slope_order

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "preprocessing" / "ptru_todd.yaml"
COLORS = {"Pt100": "#303030", "Pt66Ru33": "#832C2C", "Pt50Ru50": "#D62728"}


def calculate_ion_orders(analysis, concentrations):
    """Fit one log-rate slope over all four concentrations at each common SHE potential."""
    keys = ["material", "CO_mole_fraction", "C_electrolyte_M", "analysis_grid_index"]

    if analysis.duplicated([*keys, "replicate"]).any():
        raise ValueError("Duplicate replicate observations")
    if (analysis["rate_s_inv"] <= 0).any() or not np.isfinite(analysis["ln_rate"]).all():
        raise ValueError("Electrolyte orders require positive, finite rates")

    summary = (
        analysis.groupby(keys, as_index=False)
        .agg(
            E_V_SHE=("E_V_SHE", "first"),
            rate_mean_s_inv=("rate_s_inv", "mean"),
            ln_rate_sd=("ln_rate", "std"),
            n_replicates=("replicate", "nunique"),
        )
    )
    summary["ln_rate_mean"] = np.log(summary["rate_mean_s_inv"])

    medium = analysis["medium"].iloc[0]
    order_name = "delta_Hplus" if medium == "acidic" else "delta_OH"

    orders = calculate_log_slope_order(
        data=summary,
        varying_column="C_electrolyte_M",
        varying_values=sorted(concentrations),
        group_columns=["material", "CO_mole_fraction"],
        output_column=order_name,
        output_sd_column=f"{order_name}_sd",
    )
    orders.insert(0, "medium", medium)

    if orders.empty or not np.isfinite(orders[order_name]).all():
        raise ValueError("No valid electrolyte reaction orders calculated")
    return orders


def plot_ion_order_grid(orders, material, medium):
    """Use AgPd's four vertically stacked CO-condition panels."""
    data = orders.loc[orders["material"] == material]
    co_values = sorted(data["CO_mole_fraction"].unique())
    order_name = "delta_Hplus" if medium == "acidic" else "delta_OH"
    ion_label = "H+" if medium == "acidic" else "OH-"
    color = style._darken(COLORS[material], 0.82)

    fig, axes = plt.subplots(
        nrows=len(co_values), ncols=1,
        figsize=(
            style.OH_ORDER_FIGURE_WIDTH_IN,
            style.GRID_PANEL_HEIGHT_IN * len(co_values) + style.OH_ORDER_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=False, squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        ax = axes[row, 0]
        condition = data.loc[
            np.isclose(data["CO_mole_fraction"], co_fraction)
        ].sort_values("analysis_grid_index")

        if condition.empty:
            raise ValueError(f"No electrolyte order for {material}, {100 * co_fraction:g}% CO")

        e = condition["E_V_SHE"].to_numpy(float)
        mean = condition[order_name].to_numpy(float)
        sd = condition[f"{order_name}_sd"].to_numpy(float)
        indices = condition["analysis_grid_index"].to_numpy()

        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        for segment in np.split(np.arange(len(e)), breaks):
            valid_sd = np.isfinite(sd[segment])
            if valid_sd.any():
                ax.fill_between(
                    e[segment], mean[segment] - sd[segment], mean[segment] + sd[segment],
                    where=valid_sd, color=color, alpha=style.SUMMARY_BAND_ALPHA,
                    linewidth=0, zorder=1,
                )
            ax.plot(
                e[segment], mean[segment],
                color=color, linewidth=style.SUMMARY_LINEWIDTH, zorder=2,
            )

        ax.axhline(
            0, color=style.REFERENCE_COLOR,
            linestyle="--", linewidth=1.0, alpha=0.65, zorder=0,
        )
        style._style_axis(ax, potential_x=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        style._bold_axis_text(ax)

        ax.text(
            1.0, 0.5, f"{100 * co_fraction:g}% CO",
            transform=ax.transAxes, rotation=-90, va="center",
            fontsize=style.FONT_SIZE_SUBPLOT_TITLE,
            fontweight=style.FONT_WEIGHT,
        )

    if medium == "basic":
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        axes[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    style._finish_condition_grid(
        fig, axes,
        title=f"{material}: {ion_label} reaction order",
        ylabel=f"{ion_label} reaction order",
        right=0.90, hspace=0.08, wspace=0.0,
    )
    return fig


def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        analysis_dir = ROOT / "data" / "processed" / dataset / "analysis"
        figure_dir = ROOT / "figures" / "preprocessing" / dataset

        analysis = pd.read_parquet(analysis_dir / f"{dataset}_full.parquet")
        orders = calculate_ion_orders(analysis, config["electrolyte_concentrations_M"])
        order_name = "delta_Hplus" if medium == "acidic" else "delta_OH"

        parquet_path = analysis_dir / f"{dataset}_{order_name}.parquet"
        orders.to_parquet(parquet_path, index=False)

        n_bands = int(orders[f"{order_name}_sd"].notna().sum())
        print(f"\n{dataset}: {len(orders):,} {order_name} orders, {n_bands:,} with replicate uncertainty")
        print(f"Saved: {parquet_path}")

        for material in config["materials"]:
            fig = plot_ion_order_grid(orders, material, medium)
            output_path = figure_dir / material / f"{material}_{order_name}.png"

            style.save_agpd_figure(fig, output_path)
            plt.close(fig)
            print(f"Saved: {output_path} (+ SVG)")


if __name__ == "__main__":
    main()
