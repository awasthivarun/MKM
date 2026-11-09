
"""Plot Todd and Em Pt/PtRu rates with AgPd styling and compact axis labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, FormatStrFormatter

from mkm.preprocessing import plotting as style

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "preprocessing" / "ptru_todd.yaml"

PT_COLORS = {
    "Pt100": "#303030",
    "Pt66Ru33": "#832C2C",
    "Pt50Ru50": "#D62728",
}


def _linear_tick_label(value, _):
    """Show no more than three decimal places."""
    if value != 0 and abs(value) < 0.0005:
        return f"{value:.0e}"
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _log_tick_label(value, _):
    """Use integer mantissas: 10^n, 2*10^n, or 5*10^n."""
    if value <= 0:
        return ""

    exponent = int(np.floor(np.log10(value) + 1e-10))
    mantissa = int(round(value / 10.0 ** exponent))

    if mantissa == 1:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa}\times10^{{{exponent}}}$"


def _format_rate_ticks(ax, log_y):
    """Limit linear precision and use only conventional 1-2-5 logarithmic ticks."""
    if not log_y:
        ax.yaxis.set_major_formatter(FuncFormatter(_linear_tick_label))
        style._bold_axis_text(ax)
        return

    lower, upper = ax.get_ylim()
    exponents = range(int(np.floor(np.log10(lower))) - 1, int(np.ceil(np.log10(upper))) + 2)
    candidates = np.array(sorted(
        factor * 10.0 ** exponent
        for exponent in exponents
        for factor in (1, 2, 5)
    ))

    ticks = candidates[(candidates >= lower) & (candidates <= upper)]

    # Ensure at least one labeled tick in narrow log-scale panels.
    if len(ticks) == 0:
        center = np.sqrt(lower * upper)
        nearest = candidates[np.argmin(np.abs(np.log(candidates / center)))]

        ax.set_ylim(
            min(lower, nearest * 0.98),
            max(upper, nearest * 1.02),
        )
        ticks = np.array([nearest])

    # At most three labels; all mantissas remain 1, 2, or 5.
    if len(ticks) > 3:
        indices = np.linspace(0, len(ticks) - 1, 3).round().astype(int)
        ticks = ticks[indices]

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style._bold_axis_text(ax)


def plot_rate_grid(analysis, material, medium, log_y, config, source_label=None):
    """Plot untruncated rates using individual interpolated trials and their summaries."""
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

        condition = analysis.loc[
            np.isclose(analysis["C_electrolyte_M"], concentration)
            & np.isclose(analysis["CO_mole_fraction"], co)
        ]

        if condition.empty:
            raise ValueError(
                f"Missing data: {material}, {medium}, {concentration:g} M, {100 * co:g}% CO"
            )

        n_trials = condition["replicate"].nunique()

        # Gray curves are useful only when multiple experimental trials exist.
        if n_trials > 1:
            for _, curve in condition.groupby("replicate"):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["rate_s_inv"],
                    color=style.RETAINED_REPLICATE_COLOR,
                    linewidth=style.RETAINED_REPLICATE_LINEWIDTH,
                    alpha=0.85,
                    zorder=1,
                )

        summary = (
            condition.groupby("analysis_grid_index", as_index=False)
            .agg(
                E_V_SHE=("E_V_SHE", "first"),
                mean=("rate_s_inv", "mean"),
                sd=("rate_s_inv", "std"),
                n=("rate_s_inv", "size"),
            )
            .sort_values("E_V_SHE")
        )

        potential = summary["E_V_SHE"].to_numpy(float)
        mean = summary["mean"].to_numpy(float)
        sd = summary["sd"].fillna(0).to_numpy(float)
        replicated = summary["n"].to_numpy() >= 2

        if log_y:
            ax.set_yscale("log")

        if replicated.any():
            lower = mean - sd
            if log_y:
                lower = np.where(lower > 0, lower, np.nan)

            ax.fill_between(
                potential, lower, mean + sd,
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

        style._style_axis(ax, potential_x=True, log_y=log_y)
        _format_rate_ticks(ax, log_y)

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

    # Todd basic: at most four evenly spaced x-ticks, formatted to one decimal place.
    if not is_em and medium == "basic":
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        axes[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if is_em:
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)

    title = f"{material}: Experimental TOF"
    if log_y:
        title += " (log scale)"
    if source_label:
        title += f" - {source_label}"

    style._finish_condition_grid(
        fig, axes,
        title=title,
        ylabel=r"TOF / s$^{-1}$",
        left=0.11 if is_em else 0.085,
        right=0.95 if is_em else 0.92,
        bottom=0.065,
        top=0.925,
        hspace=0.16 if is_em else 0.08,
        wspace=0.28,
    )

    return fig


def save_rate_figures(analysis, material, medium, config, output_dir, source_label=None):
    """Save linear and logarithmic rate figures as PNG and SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for figure_name, log_y in (("rate", False), ("rate_logscale", True)):
        fig = plot_rate_grid(
            analysis, material, medium, log_y, config,
            source_label=source_label,
        )

        output_path = output_dir / f"{material}_{figure_name}.png"
        style.save_agpd_figure(fig, output_path)
        plt.close(fig)

        print(f"Saved: {output_path}")
        print(f"Saved: {output_path.with_suffix('.svg')}")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        base = ROOT / "data" / "processed" / dataset
        figure_dir = ROOT / "figures" / "preprocessing" / dataset

        # Todd: Pt100, Pt66Ru33, Pt50Ru50.
        analysis = pd.read_parquet(base / "analysis" / f"{dataset}_full.parquet")

        for material in config["materials"]:
            material_analysis = analysis.loc[analysis["material"] == material]

            if material_analysis.empty:
                raise ValueError(f"Missing Todd data for {material} in {medium}")

            save_rate_figures(
                material_analysis, material, medium, config,
                figure_dir / material,
            )

        # Em: Pt/C only, with distinct sample preparations.
        em_paths = sorted((base / "analysis").glob(f"{dataset}_Em_*_full.parquet"))

        if not em_paths:
            print(f"No processed Em data found for {medium}; skipping Em plots.")

        for analysis_path in em_paths:
            prefix = analysis_path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")
            em_analysis = pd.read_parquet(analysis_path)

            em_config = {
                **config,
                "electrolyte_concentrations_M": sorted(em_analysis["C_electrolyte_M"].unique()),
                "CO_mole_fractions": sorted(em_analysis["CO_mole_fraction"].unique()),
            }

            save_rate_figures(
                em_analysis, "Pt100", medium, em_config,
                figure_dir / f"Pt100_Em_{sample_id}",
                source_label=f"Em - {sample_id}",
            )


if __name__ == "__main__":
    main()
