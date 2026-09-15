from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator, NullFormatter, ScalarFormatter


MATERIAL_COLORS = {
    "Pd100": "#303030",
    "Ag10Pd90": "#6A3D9A",
    "Ag25Pd75": "#B23A7A",
    "Ag50Pd50": "#304FC7",
    "Ag75Pd25": "#1976D2",
    "Ag90Pd10": "#20A7D8",
}
MATERIAL_ORDER = ("Pd100", "Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10")
REFERENCE_COLOR = "#6F6F6F"
FULL_REPLICATE_COLOR = "#D0D0D0"
RETAINED_REPLICATE_COLOR = "#969696"
FULL_REPLICATE_LINEWIDTH = 0.95
RETAINED_REPLICATE_LINEWIDTH = 1.15
SUMMARY_LINEWIDTH = 2.0
SUMMARY_BAND_ALPHA = 0.18
TRUNCATED_REGION_ALPHA = 0.07
PRESENTATION_DPI = 320
PANEL_BOX_ASPECT = 0.88

FONT_FAMILY = "DejaVu Sans"
FONT_WEIGHT = "bold"
FONT_SIZE_BASE = 11.5
FONT_SIZE_FIGURE_TITLE = 12.0
FONT_SIZE_SUBPLOT_TITLE = 11.5
FONT_SIZE_AXIS_LABEL = 11.5
FONT_SIZE_TICK = 9.5
FONT_SIZE_LEGEND = 10.0
TEXT_GAP_PT = 8.0
GLOBAL_TITLE_GAP_PT = 12.0
SECOND_ORDER_STEP_V = 0.05

GRID_PANEL_WIDTH_IN = 3.30
GRID_PANEL_HEIGHT_IN = 2.55
GRID_EXTRA_WIDTH_IN = 0.90
GRID_EXTRA_HEIGHT_IN = 1.00
OH_ORDER_FIGURE_WIDTH_IN = 4.55
OH_ORDER_EXTRA_HEIGHT_IN = 1.00
OVERLAY_PANEL_WIDTH_IN = 2.65
OVERLAY_PANEL_HEIGHT_IN = 2.35
OVERLAY_EXTRA_WIDTH_IN = 0.80
OVERLAY_EXTRA_HEIGHT_IN = 1.25
OVERLAY_SINGLE_ROW_HEIGHT_IN = 3.70
SECOND_ORDER_COMPOSITION_FIGSIZE_IN = (6.2, 4.4)
SECOND_ORDER_COMPOSITION_X_TICKS = (0.00, 0.25, 0.50, 0.75)
SECOND_ORDER_COMPOSITION_X_PAD = 0.025

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "bf",
        "font.size": FONT_SIZE_BASE,
        "font.weight": "normal",
        "axes.titlesize": FONT_SIZE_SUBPLOT_TITLE,
        "axes.titleweight": FONT_WEIGHT,
        "axes.labelsize": FONT_SIZE_AXIS_LABEL,
        "axes.labelweight": FONT_WEIGHT,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.frameon": False,
        "figure.titlesize": FONT_SIZE_FIGURE_TITLE,
        "figure.titleweight": FONT_WEIGHT,
        "axes.labelpad": TEXT_GAP_PT,
        "axes.titlepad": TEXT_GAP_PT,
        "axes.grid": False,
    }
)


def _grid_figsize(ncols, nrows, *, panel_width, panel_height, extra_width=0.0, extra_height=0.0):
    return (
        float(panel_width) * int(ncols) + float(extra_width),
        float(panel_height) * int(nrows) + float(extra_height),
    )


def _mix_with_white(color, amount):
    rgb = np.asarray(to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - float(amount)) + float(amount)
    return to_hex(np.clip(mixed, 0.0, 1.0))


def _darken(color, factor=0.82):
    rgb = np.asarray(to_rgb(color), dtype=float) * float(factor)
    return to_hex(np.clip(rgb, 0.0, 1.0))


def _material_color(material):
    return _darken(MATERIAL_COLORS.get(material, "#0072B2"), 0.82)


def _ordered_materials(values):
    values = [str(value) for value in values]
    known = [material for material in MATERIAL_ORDER if material in values]
    return known + sorted(material for material in values if material not in known)


def _condition_colors(material, n_colors):
    base = MATERIAL_COLORS.get(material, "#0072B2")
    if n_colors <= 1:
        return [_darken(base, 0.82)]
    lightness = np.linspace(0.56, 0.0, int(n_colors))
    return [_mix_with_white(base, amount) for amount in lightness]


def save_agpd_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PRESENTATION_DPI, bbox_inches="tight", pad_inches=0.03)
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)


def _set_presentation_axes(axes):
    for ax in np.asarray(axes, dtype=object).flat:
        if ax.get_visible():
            ax.set_box_aspect(PANEL_BOX_ASPECT)


def _bold_axis_text(ax):
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight(FONT_WEIGHT)
        label.set_fontsize(FONT_SIZE_TICK)
    for offset_text in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
        offset_text.set_fontweight(FONT_WEIGHT)
        offset_text.set_fontsize(FONT_SIZE_TICK)


def _nice_linear_ticks(axis):
    axis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    axis.set_major_formatter(formatter)


def _set_log_y_ticks(ax):
    lower, upper = ax.get_ylim()
    if lower <= 0 or upper <= 0:
        return
    exponent_min = int(np.ceil(np.log10(lower)))
    exponent_max = int(np.floor(np.log10(upper)))
    exponents = np.arange(exponent_min, exponent_max + 1, dtype=int)
    if exponents.size > 4:
        chosen = np.linspace(0, exponents.size - 1, 4).round().astype(int)
        exponents = np.unique(exponents[chosen])
    if exponents.size:
        ax.set_yticks(10.0 ** exponents)
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _style_axis(ax, *, potential_x=False, log_y=False):
    ax.grid(False)
    if potential_x:
        lower, upper = ax.get_xlim()
        standard = np.array([-0.2, -0.1, 0.0, 0.1])
        visible = standard[(standard >= lower - 1e-12) & (standard <= upper + 1e-12)]
        if len(visible) >= 3:
            ax.set_xticks(visible)
        else:
            _nice_linear_ticks(ax.xaxis)
    else:
        _nice_linear_ticks(ax.xaxis)

    if log_y:
        _set_log_y_ticks(ax)
    else:
        _nice_linear_ticks(ax.yaxis)
    _bold_axis_text(ax)


def _style_legend(legend):
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight(FONT_WEIGHT)
        text.set_fontsize(FONT_SIZE_LEGEND)
    title = legend.get_title()
    if title is not None:
        title.set_fontweight(FONT_WEIGHT)
        title.set_fontsize(FONT_SIZE_LEGEND)


def _add_fixed_gap_global_title(fig, axes, title, *, legend=None, gap_pt=GLOBAL_TITLE_GAP_PT):
    title_artist = fig.text(
        0.5,
        0.0,
        title,
        ha="center",
        va="bottom",
        fontweight=FONT_WEIGHT,
        fontsize=FONT_SIZE_FIGURE_TITLE,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = float(gap_pt) * fig.dpi / 72.0

    reference_boxes = []
    if legend is not None and legend.get_visible():
        reference_boxes.append(legend.get_window_extent(renderer))
    else:
        visible_axes = [ax for ax in np.asarray(axes, dtype=object).flat if ax.get_visible()]
        title_boxes = [
            ax.title.get_window_extent(renderer)
            for ax in visible_axes
            if ax.title.get_visible() and ax.title.get_text()
        ]
        reference_boxes.extend(title_boxes or [ax.get_window_extent(renderer) for ax in visible_axes])

    if reference_boxes:
        reference_top = max(box.y1 for box in reference_boxes)
        title_box = title_artist.get_window_extent(renderer)
        delta_y = reference_top + gap_px - title_box.y0
        current_x, current_y = title_artist.get_position()
        title_artist.set_position((current_x, current_y + delta_y / fig.bbox.height))
    return title_artist


def _add_fixed_gap_global_labels(fig, axes, *, xlabel, ylabel, gap_pt=TEXT_GAP_PT):
    xlabel_artist = fig.supxlabel(xlabel, y=0.0, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_AXIS_LABEL)
    ylabel_artist = fig.supylabel(ylabel, x=0.0, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_AXIS_LABEL)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_axes = [ax for ax in np.asarray(axes, dtype=object).flat if ax.get_visible()]
    gap_px = float(gap_pt) * fig.dpi / 72.0

    x_tick_boxes = [
        label.get_window_extent(renderer)
        for ax in visible_axes
        for label in ax.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    if x_tick_boxes:
        bottom = min(box.y0 for box in x_tick_boxes)
        label_box = xlabel_artist.get_window_extent(renderer)
        delta_y = bottom - gap_px - label_box.y1
        current_x, current_y = xlabel_artist.get_position()
        xlabel_artist.set_position((current_x, current_y + delta_y / fig.bbox.height))

    y_tick_boxes = [
        label.get_window_extent(renderer)
        for ax in visible_axes
        for label in ax.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    if y_tick_boxes:
        left = min(box.x0 for box in y_tick_boxes)
        label_box = ylabel_artist.get_window_extent(renderer)
        delta_x = left - gap_px - label_box.x1
        current_x, current_y = ylabel_artist.get_position()
        ylabel_artist.set_position((current_x + delta_x / fig.bbox.width, current_y))


def _position_row_labels(fig, axes, *, gap_pt=TEXT_GAP_PT):
    axes_array = np.asarray(axes, dtype=object)
    right_axes = [axes_array[-1]] if axes_array.ndim == 1 else list(axes_array[:, -1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = float(gap_pt) * fig.dpi / 72.0

    for ax in right_axes:
        if not ax.get_visible():
            continue
        row_labels = [
            text
            for text in ax.texts
            if text.get_visible() and np.isclose(float(text.get_rotation()) % 360.0, 270.0)
        ]
        if not row_labels:
            continue
        reference_right = ax.transAxes.transform((1.0, 0.5))[0]
        for text in row_labels:
            text_box = text.get_window_extent(renderer)
            delta_x = reference_right + gap_px - text_box.x0
            current_x, current_y = text.get_position()
            text.set_position((current_x + delta_x / ax.bbox.width, current_y))


def _position_legend_above_column_titles(fig, legend, axes, *, gap_pt=TEXT_GAP_PT):
    if legend is None:
        return
    axes_array = np.asarray(axes, dtype=object)
    top_axes = axes_array if axes_array.ndim == 1 else axes_array[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_boxes = [
        ax.title.get_window_extent(renderer)
        for ax in np.asarray(top_axes, dtype=object).flat
        if ax.get_visible() and ax.title.get_text()
    ]
    if not title_boxes:
        return
    title_top = max(box.y1 for box in title_boxes)
    legend_box = legend.get_window_extent(renderer)
    gap_px = float(gap_pt) * fig.dpi / 72.0
    delta_y = title_top + gap_px - legend_box.y0
    anchor_box = legend.get_bbox_to_anchor()
    anchor_y = anchor_box.y0 + delta_y
    anchor_y_figure = fig.transFigure.inverted().transform((0.0, anchor_y))[1]
    legend.set_bbox_to_anchor((0.5, anchor_y_figure), transform=fig.transFigure)


def _finish_condition_grid(
    fig,
    axes,
    *,
    title,
    ylabel,
    xlabel="Potential (V vs SHE)",
    left=0.075,
    right=0.92,
    bottom=0.065,
    top=0.925,
    hspace=0.08,
    wspace=0.28,
):
    _set_presentation_axes(axes)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    _add_fixed_gap_global_labels(fig, axes, xlabel=xlabel, ylabel=ylabel)
    _position_row_labels(fig, axes)
    _add_fixed_gap_global_title(fig, axes, title)


def _add_axis_legend(ax, handles, labels, *, loc="best"):
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=1,
        prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
    )
    _style_legend(legend)
    return legend


def _get_condition_data(data, material, C_KOH_M, CO_fraction):
    return data[
        (data["material"] == material)
        & (data["C_KOH_M"] == C_KOH_M)
        & (data["CO_mole_fraction"] == CO_fraction)
    ]


def _get_truncation_row(truncation, material, C_KOH_M, CO_fraction):
    rows = truncation[
        (truncation["material"] == material)
        & (truncation["C_KOH_M"] == C_KOH_M)
        & (truncation["CO_mole_fraction"] == CO_fraction)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one truncation record for {material}, {C_KOH_M} M KOH, "
            f"{CO_fraction} CO mole fraction; found {len(rows)}."
        )
    return rows.iloc[0]


def _add_condition_headers(axes, koh_values, row_labels):
    for col, c_koh in enumerate(koh_values):
        axes[0, col].set_title(f"{c_koh:g} M KOH", fontweight=FONT_WEIGHT)
    for row, label in enumerate(row_labels):
        axes[row, -1].text(
            1.0,
            0.5,
            label,
            transform=axes[row, -1].transAxes,
            rotation=-90,
            va="center",
            fontweight=FONT_WEIGHT,
            fontsize=FONT_SIZE_SUBPLOT_TITLE,
        )


def _plot_rate_like_grid(
    full_replicates,
    selected_replicates,
    selected_summary,
    truncation,
    material,
    config,
    replicate_column,
    mean_column,
    sd_column,
    ylabel,
    title,
    log_y=False,
):
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    color = _material_color(material)
    fig, axes = plt.subplots(
        nrows=len(co_values),
        ncols=len(koh_values),
        figsize=_grid_figsize(
            len(koh_values),
            len(co_values),
            panel_width=GRID_PANEL_WIDTH_IN,
            panel_height=GRID_PANEL_HEIGHT_IN,
            extra_width=GRID_EXTRA_WIDTH_IN,
            extra_height=GRID_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            full_condition = _get_condition_data(full_replicates, material, c_koh, co_fraction)
            selected_condition = _get_condition_data(selected_replicates, material, c_koh, co_fraction)
            summary_condition = _get_condition_data(
                selected_summary, material, c_koh, co_fraction
            ).sort_values("E_V_SHE")
            truncation_row = _get_truncation_row(truncation, material, c_koh, co_fraction)
            cutoff_e = float(truncation_row["retained_min_E_V_SHE"])
            original_min_e = float(truncation_row["original_min_E_V_SHE"])

            if cutoff_e > original_min_e:
                ax.axvspan(original_min_e, cutoff_e, color=REFERENCE_COLOR, alpha=TRUNCATED_REGION_ALPHA, zorder=0)
            ax.axvline(cutoff_e, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)

            for replicate in config["replicates"]:
                full_curve = full_condition[full_condition["replicate"] == replicate].sort_values("E_V_SHE")
                retained_curve = selected_condition[selected_condition["replicate"] == replicate].sort_values("E_V_SHE")
                if not full_curve.empty:
                    ax.plot(
                        full_curve["E_V_SHE"],
                        full_curve[replicate_column],
                        color=FULL_REPLICATE_COLOR,
                        linewidth=FULL_REPLICATE_LINEWIDTH,
                        alpha=0.85,
                        zorder=1,
                    )
                if not retained_curve.empty:
                    ax.plot(
                        retained_curve["E_V_SHE"],
                        retained_curve[replicate_column],
                        color=RETAINED_REPLICATE_COLOR,
                        linewidth=RETAINED_REPLICATE_LINEWIDTH,
                        alpha=0.85,
                        zorder=2,
                    )

            potential = summary_condition["E_V_SHE"].to_numpy(dtype=float)
            mean = summary_condition[mean_column].to_numpy(dtype=float)
            sd = summary_condition[sd_column].to_numpy(dtype=float)
            lower = mean - sd
            upper = mean + sd
            if log_y:
                lower = np.where(lower > 0.0, lower, np.nan)
            ax.fill_between(potential, lower, upper, color=color, alpha=SUMMARY_BAND_ALPHA, linewidth=0, zorder=2.5)
            ax.plot(potential, mean, color=color, linewidth=SUMMARY_LINEWIDTH, zorder=3)
            if log_y:
                ax.set_yscale("log")
            _style_axis(ax, potential_x=True, log_y=log_y)

    _add_condition_headers(axes, koh_values, [f"{100 * value:g}% CO" for value in co_values])
    _finish_condition_grid(fig, axes, title=f"{material}: {title}", ylabel=ylabel)
    return fig


def _plot_alpha_grid(selected_replicates, selected_summary, truncation, material, config):
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    color = _material_color(material)
    fig, axes = plt.subplots(
        nrows=len(co_values),
        ncols=len(koh_values),
        figsize=_grid_figsize(
            len(koh_values),
            len(co_values),
            panel_width=GRID_PANEL_WIDTH_IN,
            panel_height=GRID_PANEL_HEIGHT_IN,
            extra_width=GRID_EXTRA_WIDTH_IN,
            extra_height=GRID_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            replicate_condition = _get_condition_data(selected_replicates, material, c_koh, co_fraction)
            summary_condition = _get_condition_data(
                selected_summary, material, c_koh, co_fraction
            ).sort_values("E_V_SHE")
            cutoff_e = float(
                _get_truncation_row(truncation, material, c_koh, co_fraction)["retained_min_E_V_SHE"]
            )
            ax.axvline(cutoff_e, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)

            for replicate in config["replicates"]:
                curve = replicate_condition[replicate_condition["replicate"] == replicate].sort_values("E_V_SHE")
                if curve.empty:
                    continue
                ax.plot(
                    curve["E_V_SHE"],
                    curve["alpha"],
                    color=RETAINED_REPLICATE_COLOR,
                    linewidth=RETAINED_REPLICATE_LINEWIDTH,
                    alpha=0.78,
                    zorder=1.5,
                )

            potential = summary_condition["E_V_SHE"].to_numpy(dtype=float)
            mean = summary_condition["alpha_mean"].to_numpy(dtype=float)
            sd = summary_condition["alpha_sd"].to_numpy(dtype=float)
            ax.fill_between(
                potential, mean - sd, mean + sd, color=color, alpha=SUMMARY_BAND_ALPHA, linewidth=0, zorder=2
            )
            ax.plot(potential, mean, color=color, linewidth=SUMMARY_LINEWIDTH, zorder=3)
            _style_axis(ax, potential_x=True)

    _add_condition_headers(axes, koh_values, [f"{100 * value:g}% CO" for value in co_values])
    _finish_condition_grid(fig, axes, title=f"{material}: Transfer coefficient", ylabel="Transfer coefficient")
    return fig


def plot_agpd_experimental_summary(
    full_replicates, selected_replicates, selected_summary, truncation, material, config
):
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")
    rate_figure = _plot_rate_like_grid(
        full_replicates=full_replicates,
        selected_replicates=selected_replicates,
        selected_summary=selected_summary,
        truncation=truncation,
        material=material,
        config=config,
        replicate_column="rate_s_inv",
        mean_column="rate_mean_s_inv",
        sd_column="rate_sd_s_inv",
        ylabel=r"TOF / s$^{-1}$",
        title="Experimental TOF",
        log_y=False,
    )
    rate_logscale_figure = _plot_rate_like_grid(
        full_replicates=full_replicates,
        selected_replicates=selected_replicates,
        selected_summary=selected_summary,
        truncation=truncation,
        material=material,
        config=config,
        replicate_column="rate_s_inv",
        mean_column="rate_mean_s_inv",
        sd_column="rate_sd_s_inv",
        ylabel=r"TOF / s$^{-1}$",
        title="Experimental TOF (log scale)",
        log_y=True,
    )
    alpha_figure = _plot_alpha_grid(
        selected_replicates=selected_replicates,
        selected_summary=selected_summary,
        truncation=truncation,
        material=material,
        config=config,
    )
    return {"rate": rate_figure, "rate_logscale": rate_logscale_figure, "alpha": alpha_figure}


def plot_agpd_oh_order(delta_OH, material, config):
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")
    co_values = config["CO_mole_fractions"]
    color = _material_color(material)
    fig, axes = plt.subplots(
        nrows=len(co_values),
        ncols=1,
        figsize=(OH_ORDER_FIGURE_WIDTH_IN, GRID_PANEL_HEIGHT_IN * len(co_values) + OH_ORDER_EXTRA_HEIGHT_IN),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        ax = axes[row, 0]
        condition = delta_OH[
            (delta_OH["material"] == material) & (delta_OH["CO_mole_fraction"] == co_fraction)
        ].sort_values("E_V_SHE")
        if condition.empty:
            raise ValueError(f"No OH-order data found for {material}, {100 * co_fraction:g}% CO.")
        potential = condition["E_V_SHE"].to_numpy(dtype=float)
        mean = condition["delta_OH"].to_numpy(dtype=float)
        sd = condition["delta_OH_sd"].to_numpy(dtype=float)
        ax.fill_between(potential, mean - sd, mean + sd, color=color, alpha=SUMMARY_BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(potential, mean, color=color, linewidth=SUMMARY_LINEWIDTH, zorder=2)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.65, color=REFERENCE_COLOR, zorder=0)
        ax.text(
            1.0,
            0.5,
            f"{100 * co_fraction:g}% CO",
            transform=ax.transAxes,
            rotation=-90,
            va="center",
            fontweight=FONT_WEIGHT,
            fontsize=FONT_SIZE_SUBPLOT_TITLE,
        )
        _style_axis(ax, potential_x=True)

    _finish_condition_grid(
        fig,
        axes,
        title=f"{material}: OH order",
        ylabel="OH order",
        right=0.90,
        hspace=0.08,
        wspace=0.0,
    )
    return fig


def plot_agpd_co_order(delta_CO_replicates, delta_CO, material, config):
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    co_pairs = list(zip(co_values[:-1], co_values[1:], strict=True))
    color = _material_color(material)
    fig, axes = plt.subplots(
        nrows=len(co_pairs),
        ncols=len(koh_values),
        figsize=_grid_figsize(
            len(koh_values),
            len(co_pairs),
            panel_width=GRID_PANEL_WIDTH_IN,
            panel_height=GRID_PANEL_HEIGHT_IN,
            extra_width=GRID_EXTRA_WIDTH_IN,
            extra_height=GRID_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, (lower_co, upper_co) in enumerate(co_pairs):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            replicate_condition = delta_CO_replicates[
                (delta_CO_replicates["material"] == material)
                & (delta_CO_replicates["C_KOH_M"] == c_koh)
                & (delta_CO_replicates["CO_lower_mole_fraction"] == lower_co)
                & (delta_CO_replicates["CO_upper_mole_fraction"] == upper_co)
            ]
            condition = delta_CO[
                (delta_CO["material"] == material)
                & (delta_CO["C_KOH_M"] == c_koh)
                & (delta_CO["CO_lower_mole_fraction"] == lower_co)
                & (delta_CO["CO_upper_mole_fraction"] == upper_co)
            ].sort_values("E_V_SHE")
            if condition.empty:
                raise ValueError(
                    f"No CO-order data found for {material}, {c_koh:g} M KOH, "
                    f"{100 * lower_co:g}% to {100 * upper_co:g}% CO."
                )

            potential = condition["E_V_SHE"].to_numpy(dtype=float)
            mean = condition["delta_CO"].to_numpy(dtype=float)
            sd = condition["delta_CO_sd"].to_numpy(dtype=float)
            for replicate in config["replicates"]:
                curve = replicate_condition[replicate_condition["replicate"] == replicate].sort_values("E_V_SHE")
                if curve.empty:
                    raise ValueError(
                        f"No paired CO-order data found for replicate {replicate}, {material}, "
                        f"{c_koh:g} M KOH, {100 * lower_co:g}% to {100 * upper_co:g}% CO."
                    )
                ax.plot(
                    curve["E_V_SHE"],
                    curve["delta_CO"],
                    color=RETAINED_REPLICATE_COLOR,
                    linewidth=RETAINED_REPLICATE_LINEWIDTH,
                    alpha=0.78,
                    zorder=1,
                )

            ax.fill_between(
                potential, mean - sd, mean + sd, color=color, alpha=SUMMARY_BAND_ALPHA, linewidth=0, zorder=2
            )
            ax.plot(potential, mean, color=color, linewidth=SUMMARY_LINEWIDTH, zorder=3)
            ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.65, color=REFERENCE_COLOR, zorder=0)
            _style_axis(ax, potential_x=True)

    _add_condition_headers(
        axes,
        koh_values,
        [f"{100 * lower:g}% → {100 * upper:g}% CO" for lower, upper in co_pairs],
    )
    _finish_condition_grid(fig, axes, title=f"{material}: CO order", ylabel="CO order")
    return fig



def _central_potential_derivative(frame, value_column, group_columns, step_v):
    """Central potential derivative on the existing analysis grid without interpolation."""
    step_v = float(step_v)
    if not np.isfinite(step_v) or step_v <= 0:
        raise ValueError("Potential derivative step must be finite and positive.")

    columns = [*group_columns, "E_V_SHE", value_column]
    data = frame[columns].dropna().copy()
    if data.empty:
        return pd.DataFrame(columns=[*group_columns, "E_V_SHE", "derivative"])

    data["_E_key"] = np.round(data["E_V_SHE"].to_numpy(dtype=float), 10)
    lower = data[[*group_columns, "_E_key", value_column]].rename(columns={value_column: "lower_value"})
    upper = data[[*group_columns, "_E_key", value_column]].rename(columns={value_column: "upper_value"})
    lower["_E_key"] = np.round(lower["_E_key"].to_numpy(dtype=float) + step_v, 10)
    upper["_E_key"] = np.round(upper["_E_key"].to_numpy(dtype=float) - step_v, 10)

    derivative = lower.merge(upper, on=[*group_columns, "_E_key"], how="inner", validate="one_to_one")
    derivative["E_V_SHE"] = derivative["_E_key"].astype(float)
    derivative["derivative"] = (derivative["upper_value"] - derivative["lower_value"]) / (2.0 * step_v)
    return derivative[[*group_columns, "E_V_SHE", "derivative"]]


def _experimental_second_order_difference(selected_summary, delta_OH, *, step_v=SECOND_ORDER_STEP_V):
    """Return d(alpha)/dE - d(delta_OH)/dE from experimental first-order point estimates."""
    alpha = _central_potential_derivative(
        selected_summary,
        "alpha_mean",
        ["material", "C_KOH_M", "CO_mole_fraction"],
        step_v,
    ).rename(columns={"derivative": "dalpha_dE"})
    oh = _central_potential_derivative(
        delta_OH,
        "delta_OH",
        ["material", "CO_mole_fraction"],
        step_v,
    ).rename(columns={"derivative": "ddelta_OH_dE"})

    result = alpha.merge(
        oh,
        on=["material", "CO_mole_fraction", "E_V_SHE"],
        how="inner",
        validate="many_to_one",
    )
    result["second_order_difference"] = result["dalpha_dE"] - result["ddelta_OH_dE"]
    return result


def plot_agpd_second_order_difference(
    selected_summary,
    delta_OH,
    material,
    config,
    *,
    potential_step_v=SECOND_ORDER_STEP_V,
):
    """Plot the experimental second-order kinetic difference for one material.

    No uncertainty bars are shown: neighboring first-order estimates are correlated and the
    preprocessing products do not contain the covariance needed for valid uncertainty propagation.
    """
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")

    data = _experimental_second_order_difference(selected_summary, delta_OH, step_v=potential_step_v)
    data = data.loc[data["material"] == material].copy()
    if data.empty:
        raise ValueError(f"No second-order kinetic-difference data found for {material}.")

    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    color = _material_color(material)
    fig, axes = plt.subplots(
        nrows=len(co_values),
        ncols=len(koh_values),
        figsize=_grid_figsize(
            len(koh_values),
            len(co_values),
            panel_width=GRID_PANEL_WIDTH_IN,
            panel_height=GRID_PANEL_HEIGHT_IN,
            extra_width=GRID_EXTRA_WIDTH_IN,
            extra_height=GRID_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = data[
                (data["C_KOH_M"] == c_koh) & (data["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            if condition.empty:
                raise ValueError(
                    f"No second-order kinetic-difference data found for {material}, "
                    f"{c_koh:g} M KOH, {100 * co_fraction:g}% CO."
                )
            ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.65, color=REFERENCE_COLOR, zorder=0)
            ax.plot(
                condition["E_V_SHE"],
                condition["second_order_difference"],
                linestyle="none",
                marker="o",
                markersize=4.2,
                color=color,
                zorder=2,
            )
            _style_axis(ax, potential_x=True)

    _add_condition_headers(axes, koh_values, [f"{100 * value:g}% CO" for value in co_values])
    _finish_condition_grid(
        fig,
        axes,
        title=f"{material}: second-order kinetic difference",
        ylabel=r"$d\alpha/dE - d\delta_{\mathbf{OH}}/dE$ / V$^{-1}$",
    )
    return fig


def _ag_fraction_from_material(material):
    """Return nominal Ag fraction encoded by the AgPd material name."""
    material = str(material)
    if material == "Pd100":
        return 0.0
    if material.startswith("Ag") and "Pd" in material:
        ag_percent = material[2:].split("Pd", 1)[0]
        try:
            return float(ag_percent) / 100.0
        except ValueError as error:
            raise ValueError(f"Could not parse Ag fraction from material '{material}'.") from error
    raise ValueError(f"Could not parse Ag fraction from material '{material}'.")


def _common_second_order_support(data, materials):
    """Restrict second-order points to the condition/potential support shared by all materials."""
    materials = [str(material) for material in materials]
    keys = ["C_KOH_M", "CO_mole_fraction", "E_V_SHE"]
    present = data.loc[data["material"].isin(materials), ["material", *keys]].drop_duplicates()
    common = present.groupby(keys, as_index=False)["material"].nunique()
    common = common.loc[common["material"] == len(materials), keys]
    if common.empty:
        raise ValueError("No common second-order condition/potential support exists across materials.")
    return data.merge(common, on=keys, how="inner", validate="many_to_one")


def plot_agpd_second_order_composition(
    selected_summary,
    delta_OH,
    config,
    *,
    potential_step_v=SECOND_ORDER_STEP_V,
):
    """Plot mean +/- SD second-order kinetic difference versus nominal Ag fraction.

    The mean and SD describe variation across the common KOH/CO/potential support, not
    propagated experimental uncertainty. Restricting to common support avoids composition
    trends caused only by material-dependent truncation of the potential window.
    """
    materials = _ordered_materials(config["materials"])
    data = _experimental_second_order_difference(selected_summary, delta_OH, step_v=potential_step_v)
    data = _common_second_order_support(data, materials)

    summary = (
        data.groupby("material", as_index=False)["second_order_difference"]
        .agg(mean="mean", sd="std", n_points="size")
    )
    summary["xAg"] = summary["material"].map(_ag_fraction_from_material)
    summary = summary.sort_values("xAg").reset_index(drop=True)
    if summary["sd"].isna().any():
        raise ValueError("At least two common second-order points are required per material for an SD.")

    fig, ax = plt.subplots(figsize=SECOND_ORDER_COMPOSITION_FIGSIZE_IN)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.65, color=REFERENCE_COLOR, zorder=0)
    for row in summary.itertuples(index=False):
        ax.errorbar(
            row.xAg,
            row.mean,
            yerr=row.sd,
            fmt="o",
            color=_material_color(row.material),
            markersize=6.0,
            linewidth=1.15,
            capsize=2.5,
            zorder=2,
        )

    x_min = float(summary["xAg"].min()) - SECOND_ORDER_COMPOSITION_X_PAD
    x_max = float(summary["xAg"].max()) + SECOND_ORDER_COMPOSITION_X_PAD
    ax.set_xlim(x_min, x_max)
    visible_ticks = [tick for tick in SECOND_ORDER_COMPOSITION_X_TICKS if x_min <= tick <= x_max]
    ax.set_xticks(visible_ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in visible_ticks])
    _nice_linear_ticks(ax.yaxis)
    _bold_axis_text(ax)
    ax.grid(False)
    ax.set_box_aspect(0.72)

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.88)
    _add_fixed_gap_global_labels(
        fig,
        np.asarray([ax], dtype=object),
        xlabel=r"Ag fraction, $x_{\mathbf{Ag}}$",
        ylabel=r"Mean $d\alpha/dE - d\delta_{\mathbf{OH}}/dE$ / V$^{-1}$",
    )
    _add_fixed_gap_global_title(
        fig,
        np.asarray([ax], dtype=object),
        "Second-order kinetic difference vs Ag composition",
    )
    return fig

def _overlay_condition_legend(fig, condition_labels, *, n_conditions, y=0.955):
    shades = [_mix_with_white("#4D4D4D", amount) for amount in np.linspace(0.58, 0.0, n_conditions)]
    handles = [Line2D([0], [0], color=color, linewidth=2.0) for color in shades]
    legend = fig.legend(
        handles,
        condition_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        columnspacing=0.8,
        handlelength=1.5,
        prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
    )
    _style_legend(legend)
    return legend


def _finish_overlay_grid(fig, axes, *, title, ylabel, condition_labels, row_labels=None, wspace=0.09):
    _set_presentation_axes(axes)
    for ax in np.asarray(axes, dtype=object).flat:
        _style_axis(ax, potential_x=True)
    if row_labels is not None:
        for row, label in enumerate(row_labels):
            axes[row, -1].text(
                1.0,
                0.5,
                label,
                transform=axes[row, -1].transAxes,
                rotation=-90,
                va="center",
                fontweight=FONT_WEIGHT,
                fontsize=FONT_SIZE_SUBPLOT_TITLE,
            )
    legend = _overlay_condition_legend(fig, condition_labels, n_conditions=len(condition_labels), y=0.955)
    fig.subplots_adjust(left=0.05, right=0.965, bottom=0.07, top=0.875, hspace=0.09, wspace=wspace)
    _add_fixed_gap_global_labels(fig, axes, xlabel="Potential (V vs SHE)", ylabel=ylabel)
    _position_row_labels(fig, axes)
    _position_legend_above_column_titles(fig, legend, axes)
    _add_fixed_gap_global_title(fig, axes, title, legend=legend)


def _plot_experimental_overlay_panel(ax, data, *, color, mean_column, sd_column):
    if data.empty:
        return
    data = data.sort_values("E_V_SHE")
    ax.errorbar(
        data["E_V_SHE"],
        data[mean_column],
        yerr=data[sd_column],
        fmt="o",
        color=color,
        markersize=2.8,
        linestyle="none",
        elinewidth=0.65,
        capsize=1.2,
        zorder=2,
    )


def plot_agpd_alpha_overlay_pco(selected_summary, config):
    materials = _ordered_materials(selected_summary["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        len(koh_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(koh_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, c_koh in enumerate(koh_values):
            for color, co_fraction in zip(colors, co_values, strict=True):
                data = _get_condition_data(selected_summary, material, c_koh, co_fraction)
                _plot_experimental_overlay_panel(
                    axes[row, col], data, color=color, mean_column="alpha_mean", sd_column="alpha_sd"
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental transfer coefficient across materials",
        ylabel="Transfer coefficient",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    return fig


def plot_agpd_alpha_overlay_koh(selected_summary, config):
    materials = _ordered_materials(selected_summary["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        len(co_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(co_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, co_fraction in enumerate(co_values):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = _get_condition_data(selected_summary, material, c_koh, co_fraction)
                _plot_experimental_overlay_panel(
                    axes[row, col], data, color=color, mean_column="alpha_mean", sd_column="alpha_sd"
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental transfer coefficient across materials",
        ylabel="Transfer coefficient",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    return fig


def _co_intervals(delta_co):
    return list(
        delta_co[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]]
        .drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"])
        .itertuples(index=False, name=None)
    )


def plot_agpd_co_order_overlay_pco(delta_CO, config):
    materials = _ordered_materials(delta_CO["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    intervals = _co_intervals(delta_CO)
    fig, axes = plt.subplots(
        len(koh_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(koh_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(intervals))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, c_koh in enumerate(koh_values):
            for color, (lower_co, upper_co) in zip(colors, intervals, strict=True):
                data = delta_CO[
                    (delta_CO["material"] == material)
                    & (delta_CO["C_KOH_M"] == c_koh)
                    & (delta_CO["CO_lower_mole_fraction"] == lower_co)
                    & (delta_CO["CO_upper_mole_fraction"] == upper_co)
                ]
                _plot_experimental_overlay_panel(
                    axes[row, col], data, color=color, mean_column="delta_CO", sd_column="delta_CO_sd"
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental adjacent CO order across materials",
        ylabel="CO order",
        condition_labels=[f"{100 * lo:g}% → {100 * hi:g}% CO" for lo, hi in intervals],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    return fig


def plot_agpd_co_order_overlay_koh(delta_CO, config):
    materials = _ordered_materials(delta_CO["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    intervals = _co_intervals(delta_CO)
    fig, axes = plt.subplots(
        len(intervals),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(intervals),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, (lower_co, upper_co) in enumerate(intervals):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = delta_CO[
                    (delta_CO["material"] == material)
                    & (delta_CO["C_KOH_M"] == c_koh)
                    & (delta_CO["CO_lower_mole_fraction"] == lower_co)
                    & (delta_CO["CO_upper_mole_fraction"] == upper_co)
                ]
                _plot_experimental_overlay_panel(
                    axes[row, col], data, color=color, mean_column="delta_CO", sd_column="delta_CO_sd"
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental adjacent CO order across materials",
        ylabel="CO order",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * lo:g}% → {100 * hi:g}% CO" for lo, hi in intervals],
    )
    return fig


def plot_agpd_oh_order_overlay_pco(delta_OH, config):
    materials = _ordered_materials(delta_OH["material"].unique())
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        1,
        len(materials),
        figsize=(OVERLAY_PANEL_WIDTH_IN * len(materials) + OVERLAY_EXTRA_WIDTH_IN, OVERLAY_SINGLE_ROW_HEIGHT_IN),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for color, co_fraction in zip(colors, co_values, strict=True):
            data = delta_OH[
                (delta_OH["material"] == material) & (delta_OH["CO_mole_fraction"] == co_fraction)
            ]
            _plot_experimental_overlay_panel(
                axes[0, col], data, color=color, mean_column="delta_OH", sd_column="delta_OH_sd"
            )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental OH order across materials",
        ylabel="OH order",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    return fig


def plot_agpd_rate_overlay_pco(selected_summary, config):
    materials = _ordered_materials(selected_summary["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        len(koh_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(koh_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, c_koh in enumerate(koh_values):
            for color, co_fraction in zip(colors, co_values, strict=True):
                data = _get_condition_data(selected_summary, material, c_koh, co_fraction)
                _plot_experimental_overlay_panel(
                    axes[row, col], data, color=color, mean_column="rate_mean_s_inv", sd_column="rate_sd_s_inv"
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental TOF across materials",
        ylabel=r"TOF / s$^{-1}$",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
        wspace=0.24,
    )
    return fig


def plot_agpd_rate_overlay_koh(selected_summary, config):
    materials = _ordered_materials(selected_summary["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        len(co_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(co_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, co_fraction in enumerate(co_values):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = _get_condition_data(selected_summary, material, c_koh, co_fraction)
                _plot_experimental_overlay_panel(
                    axes[row, col], data, color=color, mean_column="rate_mean_s_inv", sd_column="rate_sd_s_inv"
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental TOF across materials",
        ylabel=r"TOF / s$^{-1}$",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values],
        wspace=0.24,
    )
    return fig


def _plot_experimental_marker_overlay_panel(ax, data, *, color, value_column):
    """Plot experimental overlay points without connecting lines or uncertainty bars."""
    if data.empty:
        return
    data = data.sort_values("E_V_SHE")
    ax.plot(
        data["E_V_SHE"],
        data[value_column],
        linestyle="none",
        marker="o",
        markersize=3.2,
        color=color,
        zorder=2,
    )


def plot_agpd_second_order_overlay_pco(
    selected_summary,
    delta_OH,
    config,
    *,
    potential_step_v=SECOND_ORDER_STEP_V,
):
    """Plot experimental second-order kinetic difference across materials, overlaying CO fractions."""
    data = _experimental_second_order_difference(selected_summary, delta_OH, step_v=potential_step_v)
    materials = _ordered_materials(data["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        len(koh_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(koh_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, c_koh in enumerate(koh_values):
            axes[row, col].axhline(
                0.0,
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                color=REFERENCE_COLOR,
                zorder=0,
            )
            for color, co_fraction in zip(colors, co_values, strict=True):
                condition = data[
                    (data["material"] == material)
                    & (data["C_KOH_M"] == c_koh)
                    & (data["CO_mole_fraction"] == co_fraction)
                ]
                _plot_experimental_marker_overlay_panel(
                    axes[row, col],
                    condition,
                    color=color,
                    value_column="second_order_difference",
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental second-order kinetic difference across materials",
        ylabel=r"$d\alpha/dE - d\delta_{\mathbf{OH}}/dE$ / V$^{-1}$",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    return fig


def plot_agpd_second_order_overlay_koh(
    selected_summary,
    delta_OH,
    config,
    *,
    potential_step_v=SECOND_ORDER_STEP_V,
):
    """Plot experimental second-order kinetic difference across materials, overlaying KOH levels."""
    data = _experimental_second_order_difference(selected_summary, delta_OH, step_v=potential_step_v)
    materials = _ordered_materials(data["material"].unique())
    koh_values = config["KOH_concentrations_M"]
    co_values = config["CO_mole_fractions"]
    fig, axes = plt.subplots(
        len(co_values),
        len(materials),
        figsize=_grid_figsize(
            len(materials),
            len(co_values),
            panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN,
            extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, fontweight=FONT_WEIGHT)
        for row, co_fraction in enumerate(co_values):
            axes[row, col].axhline(
                0.0,
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                color=REFERENCE_COLOR,
                zorder=0,
            )
            for color, c_koh in zip(colors, koh_values, strict=True):
                condition = data[
                    (data["material"] == material)
                    & (data["C_KOH_M"] == c_koh)
                    & (data["CO_mole_fraction"] == co_fraction)
                ]
                _plot_experimental_marker_overlay_panel(
                    axes[row, col],
                    condition,
                    color=color,
                    value_column="second_order_difference",
                )
    _finish_overlay_grid(
        fig,
        axes,
        title="Experimental second-order kinetic difference across materials",
        ylabel=r"$d\alpha/dE - d\delta_{\mathbf{OH}}/dE$ / V$^{-1}$",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    return fig

