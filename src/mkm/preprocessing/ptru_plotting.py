
"""Plot experimental Pt/PtRu CO oxidation rates and kinetic observables."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, MaxNLocator, NullFormatter

from mkm.preprocessing import plotting as style

PT_COLORS = {
    "Pt100": "#303030",
    "Pt66Ru33": "#832C2C",
    "Pt50Ru50": "#D62728",
}


# ---------------------------------------------------------------------
# Rate tick formatting
# ---------------------------------------------------------------------


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

    if len(ticks) == 0:
        center = np.sqrt(lower * upper)
        nearest = candidates[np.argmin(np.abs(np.log(candidates / center)))]

        ax.set_ylim(
            min(lower, nearest * 0.98),
            max(upper, nearest * 1.02),
        )
        ticks = np.array([nearest])

    if len(ticks) > 3:
        indices = np.linspace(0, len(ticks) - 1, 3).round().astype(int)
        ticks = ticks[indices]

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    style._bold_axis_text(ax)


# ---------------------------------------------------------------------
# Rate plots
# ---------------------------------------------------------------------


def plot_rate_grid(analysis, material, medium, log_y, config, source_label=None, cutoffs=None):
    """Plot experimental rates with optional low-potential truncation previews."""
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
        cutoff = None if cutoffs is None else cutoffs.get((float(concentration), float(co)))

        if log_y:
            ax.set_yscale("log")

        if replicated.any():
            lower = mean - sd
            if log_y:
                lower = np.where(lower > 0, lower, np.nan)

            ax.fill_between(
                potential, lower, mean + sd,
                where=replicated if cutoff is None else replicated & (potential >= cutoff),
                color=color,
                alpha=style.SUMMARY_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )

        if cutoff is None:
            ax.plot(
                potential, mean,
                color=color,
                linewidth=style.SUMMARY_LINEWIDTH,
                zorder=3,
            )
        else:
            ax.axvspan(
                potential.min(), cutoff,
                color=style.REFERENCE_COLOR,
                alpha=style.TRUNCATED_REGION_ALPHA,
                zorder=0,
            )
            ax.axvline(
                cutoff, color=style.REFERENCE_COLOR,
                linestyle="--", linewidth=1.0, alpha=0.75, zorder=2,
            )
            ax.plot(
                potential, mean,
                color=style.FULL_REPLICATE_COLOR,
                linewidth=style.FULL_REPLICATE_LINEWIDTH,
                zorder=1,
            )
            retained = potential >= cutoff
            ax.plot(
                potential[retained], mean[retained],
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


def save_rate_figures(analysis, material, medium, config, output_dir, source_label=None, cutoffs=None):
    """Save linear and logarithmic rate figures with optional truncation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for figure_name, log_y in (("rate", False), ("rate_logscale", True)):
        fig = plot_rate_grid(
            analysis, material, medium, log_y, config,
            source_label=source_label, cutoffs=cutoffs,
        )

        output_path = output_dir / f"{material}_{figure_name}.png"
        style.save_agpd_figure(fig, output_path)
        plt.close(fig)

        print(f"Saved: {output_path}")
        print(f"Saved: {output_path.with_suffix('.svg')}")


# ---------------------------------------------------------------------
# Transfer coefficient plots
# ---------------------------------------------------------------------


def plot_alpha_grid(replicates, summary, material, medium, config, source_label=None, cutoffs=None):
    """Plot experimental alpha with optional low-potential truncation previews."""
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
        cutoff = None if cutoffs is None else cutoffs.get((float(concentration), float(co)))

        if replicated.any():
            ax.fill_between(
                potential, mean - sd, mean + sd,
                where=replicated if cutoff is None else replicated & (potential >= cutoff),
                color=color,
                alpha=style.SUMMARY_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )

        if cutoff is None:
            ax.plot(
                potential, mean,
                color=color,
                linewidth=style.SUMMARY_LINEWIDTH,
                zorder=3,
            )
        else:
            ax.axvspan(
                potential.min(), cutoff,
                color=style.REFERENCE_COLOR,
                alpha=style.TRUNCATED_REGION_ALPHA,
                zorder=0,
            )
            ax.axvline(
                cutoff, color=style.REFERENCE_COLOR,
                linestyle="--", linewidth=1.0, alpha=0.75, zorder=2,
            )
            ax.plot(
                potential, mean,
                color=style.FULL_REPLICATE_COLOR,
                linewidth=style.FULL_REPLICATE_LINEWIDTH,
                zorder=1,
            )
            retained = potential >= cutoff
            ax.plot(
                potential[retained], mean[retained],
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


# ---------------------------------------------------------------------
# CO reaction order plots
# ---------------------------------------------------------------------


def plot_co_order_grid(orders, material, medium, source_label=None):
    """Plot adjacent-pressure CO orders and their available uncertainty bands."""
    data = orders.loc[orders["material"] == material]
    concentrations = sorted(data["C_electrolyte_M"].unique())
    pairs = list(
        data[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]]
        .drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"])
        .itertuples(index=False, name=None)
    )
    color = style._darken(PT_COLORS[material], 0.82)
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
            ax.set_title(
                pair_label,
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
                    1.0, 0.5, pair_label,
                    transform=ax.transAxes,
                    rotation=-90, va="center",
                    fontsize=style.FONT_SIZE_SUBPLOT_TITLE,
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
        fig, axes,
        title=title,
        ylabel="CO reaction order",
        left=0.11 if is_em else 0.085,
        right=0.95 if is_em else 0.92,
        bottom=0.065,
        top=0.925,
        hspace=0.16 if is_em else 0.08,
        wspace=0.28,
    )

    return fig


# ---------------------------------------------------------------------
# Electrolyte reaction order plots
# ---------------------------------------------------------------------


def plot_ion_order_grid(orders, material, medium):
    """Plot H+ or OH- orders in four vertically stacked CO-condition panels."""
    data = orders.loc[orders["material"] == material]
    co_values = sorted(data["CO_mole_fraction"].unique())
    order_name = "delta_Hplus" if medium == "acidic" else "delta_OH"
    ion_label = "H+" if medium == "acidic" else "OH-"
    color = style._darken(PT_COLORS[material], 0.82)

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
