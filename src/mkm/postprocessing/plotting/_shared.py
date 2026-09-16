import math
from pathlib import Path

import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator, NullFormatter, ScalarFormatter
from scipy.stats import gaussian_kde
from arviz_stats.ecdf_utils import ecdf_pit

from mkm.inference.priors import prior_pdf as _prior_pdf, prior_support as _prior_support
from mkm.postprocessing.sampling import build_sampling_datatree


PATHWAY_COLORS = {
    "CO_adsorption": "#CC79A7",
    "BF": "#0072B2",
    "ER": "#D55E00",
    "LH": "#009E73",
}

COVERAGE_COLORS = {
    "theta_CO": "#0072B2",
    "theta_OH_Pd": "#D55E00",
    "theta_empty_Pd": "#777777",
    "theta_OH_Ag": "#009E73",
    "theta_empty_Ag": "#777777",
}

POINTWISE_LABELS = {
    "theta_CO": "CO",
    "theta_OH_Pd": "OH",
    "theta_empty_Pd": r"$[\ast]$",
    "theta_OH_Ag": "OH",
    "theta_empty_Ag": "[#]",
    "rate_fraction_BF": "BF",
    "rate_fraction_ER": "ER",
    "rate_fraction_LH": "LH",
}

# Material hue is the visual identity for experiment-vs-model plots. Chemistry-specific plots
# (coverages, pathway fractions, and DRC) keep their own semantic color maps.
MATERIAL_COLORS = {
    "Pd100": "#303030",
    "Ag10Pd90": "#6A3D9A",
    "Ag25Pd75": "#B23A7A",
    "Ag50Pd50": "#304FC7",
    "Ag75Pd25": "#1976D2",
    "Ag90Pd10": "#20A7D8",
}
MATERIAL_ORDER = ("Pd100", "Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10")
POSTERIOR_COLOR = "#0072B2"
EXPERIMENT_COLOR = "#B8B8B8"
REFERENCE_COLOR = "#6F6F6F"
ALERT_COLOR = "#B33C6A"
HDI95_ALPHA = 0.10
HDI80_ALPHA = 0.22
PRESENTATION_DPI = 320
PANEL_BOX_ASPECT = 0.88  # height / width; deliberately a little wider than geometric square

FONT_FAMILY = "DejaVu Sans"
FONT_WEIGHT = "bold"
FONT_SIZE_BASE = 11.5
FONT_SIZE_FIGURE_TITLE = 12.0
FONT_SIZE_SUBPLOT_TITLE = 11.5
FONT_SIZE_AXIS_LABEL = 11.5
FONT_SIZE_TICK = 9.5
FONT_SIZE_LEGEND = 10.0
FONT_SIZE_POSTERIOR_ANNOTATION = 8.0
TEXT_GAP_PT = 8.0
GLOBAL_TITLE_GAP_PT = 12.0

REPLICATE_LINE_COLOR = "#969696"
REPLICATE_LINEWIDTH = 1.15
REPLICATE_LINE_ALPHA = 0.78
RESIDUAL_REPLICATE_GREY_RANGE = (0.30, 0.58)

TRACE_ALPHA_GOOD = 0.55
TRACE_ALPHA_BAD = 1.00
TRACE_BFMI_BAD = 0.30
TRACE_BFMI_GOOD = 0.60

# Figure sizing profiles. These reproduce the existing figure dimensions exactly; centralizing them
# changes no layout by itself. Width/height are in inches.
POSTERIOR_PANEL_WIDTH_IN = 3.65
POSTERIOR_PANEL_HEIGHT_IN = 2.55
POSTERIOR_EXTRA_HEIGHT_IN = 0.35
POSTERIOR_BOX_ASPECT = 0.62
POSTERIOR_WSPACE = 0.10
OBSERVATION_PANEL_WIDTH_IN = 3.30
OBSERVATION_PANEL_HEIGHT_IN = 2.55
OBSERVATION_EXTRA_WIDTH_IN = 0.90
OBSERVATION_EXTRA_HEIGHT_IN = 1.00
MATERIAL_PANEL_WIDTH_IN = 2.95
MATERIAL_PANEL_HEIGHT_IN = 2.55
MATERIAL_EXTRA_WIDTH_IN = 0.70
MATERIAL_EXTRA_HEIGHT_IN = 1.00
POINTWISE_EXTRA_HEIGHT_IN = 1.20
OVERLAY_PANEL_WIDTH_IN = 2.65
OVERLAY_PANEL_HEIGHT_IN = 2.35
OVERLAY_EXTRA_WIDTH_IN = 0.80
OVERLAY_EXTRA_HEIGHT_IN = 1.25
OVERLAY_SINGLE_ROW_HEIGHT_IN = 3.70
OH_ORDER_FIGURE_WIDTH_IN = 4.55
OH_ORDER_EXTRA_HEIGHT_IN = 1.00
LOO_PANEL_WIDTH_IN = 3.15
LOO_PANEL_HEIGHT_IN = 2.55
LOO_EXTRA_HEIGHT_IN = 0.90
LOO_POINTWISE_PANEL_WIDTH_IN = 3.25
LOO_POINTWISE_PANEL_HEIGHT_IN = 2.65
LOO_POINTWISE_EXTRA_WIDTH_IN = 1.00
LOO_POINTWISE_EXTRA_HEIGHT_IN = 1.20
TRACE_FIGURE_WIDTH_IN = 13.20
TRACE_PANEL_HEIGHT_IN = 2.55
TRACE_EXTRA_HEIGHT_IN = 0.20
RANK_FIGURE_WIDTH_IN = 13.20
RANK_PANEL_HEIGHT_IN = 2.45
RANK_EXTRA_HEIGHT_IN = 0.20
ENERGY_FIGURE_SIZE_IN = (8.80, 3.15)
PAIR_MIN_SIZE_IN = 10.0
PAIR_SIZE_PER_PARAMETER_IN = 1.8
CORRELATION_MIN_SIZE_IN = 9.0
CORRELATION_SIZE_PER_PARAMETER_IN = 0.62
CORRELATION_ANNOTATION_SIZE = 8.0
LOO_PIT_SUMMARY_SIZE_IN = (7.20, 5.60)
LOO_DIAGNOSTICS_SIZE_IN = (7.40, 7.50)

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

DEFAULT_COLOR_CYCLE = plt.rcParamsDefault["axes.prop_cycle"]
DEFAULT_COLORS = tuple(item["color"] for item in DEFAULT_COLOR_CYCLE)


def _grid_figsize(ncols, nrows, *, panel_width, panel_height, extra_width=0.0, extra_height=0.0):
    return (
        float(panel_width) * int(ncols) + float(extra_width),
        float(panel_height) * int(nrows) + float(extra_height),
    )


def _chain_bfmi_values(inference_data):
    sample_stats = getattr(inference_data, "sample_stats", None)
    if sample_stats is None:
        try:
            sample_stats = inference_data["sample_stats"]
        except (KeyError, TypeError):
            return None
    if hasattr(sample_stats, "to_dataset") and not isinstance(sample_stats, xr.Dataset):
        sample_stats = sample_stats.to_dataset()
    try:
        energy = np.asarray(sample_stats["energy"], dtype=float)
    except (KeyError, TypeError):
        return None
    energy = np.squeeze(energy)
    if energy.ndim != 2 or energy.shape[1] < 2:
        return None
    numerator = np.square(np.diff(energy, axis=1)).mean(axis=1)
    denominator = np.var(energy, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _trace_alpha_for_bfmi(value):
    if not np.isfinite(value):
        return TRACE_ALPHA_GOOD
    if value <= TRACE_BFMI_BAD:
        return TRACE_ALPHA_BAD
    if value >= TRACE_BFMI_GOOD:
        return TRACE_ALPHA_GOOD
    fraction = (value - TRACE_BFMI_BAD) / (TRACE_BFMI_GOOD - TRACE_BFMI_BAD)
    return TRACE_ALPHA_BAD + fraction * (TRACE_ALPHA_GOOD - TRACE_ALPHA_BAD)


def _apply_trace_chain_alpha(fig, inference_data):
    bfmi = _chain_bfmi_values(inference_data)
    if bfmi is None:
        chain_alphas = [TRACE_ALPHA_GOOD] * len(DEFAULT_COLORS)
    else:
        chain_alphas = [_trace_alpha_for_bfmi(value) for value in np.ravel(bfmi)]

    color_alpha = {
        to_hex(color).lower(): chain_alphas[index]
        for index, color in enumerate(DEFAULT_COLORS[: len(chain_alphas)])
    }
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_marker() == "|" or line.get_linestyle() in {"None", "none", ""}:
                continue
            try:
                alpha = color_alpha.get(to_hex(line.get_color()).lower())
            except (TypeError, ValueError):
                alpha = None
            if alpha is not None:
                line.set_alpha(alpha)


def _mix_with_white(color, amount):
    rgb = np.asarray(to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - float(amount)) + float(amount)
    return to_hex(np.clip(mixed, 0.0, 1.0))


def _darken(color, factor=0.82):
    rgb = np.asarray(to_rgb(color), dtype=float) * float(factor)
    return to_hex(np.clip(rgb, 0.0, 1.0))


def _material_from_context(context_label):
    if not context_label:
        return None
    return str(context_label).split(" / ")[0]


def _material_plot_colors(material):
    base = MATERIAL_COLORS.get(material, POSTERIOR_COLOR)
    posterior = "#202020" if material == "Pd100" else _darken(base, 0.82)
    return posterior, EXPERIMENT_COLOR


def _ordered_materials(values):
    values = [str(value) for value in values]
    known = [material for material in MATERIAL_ORDER if material in values]
    return known + sorted(material for material in values if material not in known)


def _condition_colors(material, n_colors):
    base = MATERIAL_COLORS.get(material, POSTERIOR_COLOR)
    if n_colors <= 1:
        return [_darken(base, 0.82)]
    lightness = np.linspace(0.56, 0.0, int(n_colors))
    return [_mix_with_white(base, amount) for amount in lightness]


def _parameter_value_text(value, hdi_width):
    if not np.isfinite(hdi_width) or hdi_width <= 0.0:
        return f"{value:.4g}"
    exponent = int(np.floor(np.log10(hdi_width)))
    decimals = int(np.clip(2 - exponent, 0, 9))
    return f"{value:.{decimals}f}"


def _save_presentation_figure(fig, output_path, *, svg=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PRESENTATION_DPI, bbox_inches="tight", pad_inches=0.03)
    if svg and output_path.suffix.lower() == ".png":
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


def _style_axis(ax, *, potential_x=False, bounded01_y=False, log_y=False):
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

    if bounded01_y:
        ax.set_yticks([0.0, 0.5, 1.0])
    elif log_y:
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
        0.5, 0.0, title, ha="center", va="bottom",
        fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_FIGURE_TITLE,
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
        if title_boxes:
            reference_boxes.extend(title_boxes)
        else:
            reference_boxes.extend(ax.get_window_extent(renderer) for ax in visible_axes)

    if reference_boxes:
        reference_top = max(box.y1 for box in reference_boxes)
        title_box = title_artist.get_window_extent(renderer)
        delta_y = reference_top + gap_px - title_box.y0
        current_x, current_y = title_artist.get_position()
        title_artist.set_position((current_x, current_y + delta_y / fig.bbox.height))
    return title_artist


def _add_fixed_gap_global_labels(fig, axes, *, xlabel, ylabel, gap_pt=TEXT_GAP_PT):
    xlabel_artist = fig.supxlabel(
        xlabel, y=0.0, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_AXIS_LABEL
    )
    ylabel_artist = fig.supylabel(
        ylabel, x=0.0, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_AXIS_LABEL
    )
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
    if axes_array.ndim == 1:
        right_axes = [axes_array[-1]]
    else:
        right_axes = list(axes_array[:, -1])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = float(gap_pt) * fig.dpi / 72.0

    for ax in right_axes:
        if not ax.get_visible():
            continue
        row_labels = [
            text for text in ax.texts
            if text.get_visible() and np.isclose(float(text.get_rotation()) % 360.0, 270.0)
        ]
        if not row_labels:
            continue

        reference_right = ax.transAxes.transform((1.0, 0.5))[0]
        ax_position = np.asarray(ax.get_position().bounds, dtype=float)
        for other in fig.axes:
            if other is ax or not other.get_visible():
                continue
            if not np.allclose(np.asarray(other.get_position().bounds), ax_position, atol=1e-6, rtol=0.0):
                continue
            for tick_label in other.get_yticklabels():
                if not tick_label.get_visible() or not tick_label.get_text():
                    continue
                tick_box = tick_label.get_window_extent(renderer)
                if tick_box.x0 >= reference_right - 1.0:
                    reference_right = max(reference_right, tick_box.x1)

        for text in row_labels:
            text_box = text.get_window_extent(renderer)
            delta_x = reference_right + gap_px - text_box.x0
            current_x, current_y = text.get_position()
            text.set_position((current_x + delta_x / ax.bbox.width, current_y))



def _add_fixed_gap_right_ylabel(fig, right_axes, row_axes, *, ylabel, gap_pt=TEXT_GAP_PT):
    ylabel_artist = fig.text(
        1.0, 0.5, ylabel, rotation=-90, va="center", ha="center",
        fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_AXIS_LABEL,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = float(gap_pt) * fig.dpi / 72.0
    reference_boxes = []

    for ax in np.asarray(right_axes, dtype=object).flat:
        for label in ax.get_yticklabels():
            if label.get_visible() and label.get_text():
                reference_boxes.append(label.get_window_extent(renderer))

    for ax in np.asarray(row_axes, dtype=object).flat:
        for text in ax.texts:
            if text.get_visible() and np.isclose(float(text.get_rotation()) % 360.0, 270.0):
                reference_boxes.append(text.get_window_extent(renderer))

    if reference_boxes:
        reference_right = max(box.x1 for box in reference_boxes)
        label_box = ylabel_artist.get_window_extent(renderer)
        delta_x = reference_right + gap_px - label_box.x0
        current_x, current_y = ylabel_artist.get_position()
        ylabel_artist.set_position((current_x + delta_x / fig.bbox.width, current_y))

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


def _add_figure_legend(fig, axes, *, max_columns=4, y=0.955):
    handles = []
    labels = []
    for ax in np.asarray(axes, dtype=object).flat:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if not label or label.startswith("_") or label in labels:
                continue
            handles.append(handle)
            labels.append(label)
    if not handles:
        return None
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=min(len(handles), max_columns),
        columnspacing=1.0,
        handlelength=1.8,
        prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
    )
    _style_legend(legend)
    return legend


def _add_axis_legend(ax, *, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels, strict=True):
        if not label or label.startswith("_") or label in unique_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        legend = ax.legend(
            unique_handles, unique_labels, loc=loc, ncol=1,
            prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
        )
        _style_legend(legend)


def _finish_condition_grid(
    fig,
    axes,
    *,
    title,
    xlabel,
    ylabel,
    legend=True,
    left=0.075,
    right=0.92,
    bottom=0.065,
    top=None,
    hspace=0.08,
    wspace=0.08,
):
    if top is None:
        top = 0.885 if legend else 0.925
    figure_legend = _add_figure_legend(fig, axes, y=0.955) if legend else None
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    _add_fixed_gap_global_labels(fig, axes, xlabel=xlabel, ylabel=ylabel)
    _position_row_labels(fig, axes)
    if figure_legend is not None:
        _position_legend_above_column_titles(fig, figure_legend, axes)
    _add_fixed_gap_global_title(fig, axes, title, legend=figure_legend)


def _posterior_hdi_columns(frame, prefix=None, probability=95):
    base = "" if prefix is None else f"{prefix}_"
    columns = (f"{base}hdi{probability}_lower", f"{base}hdi{probability}_upper")
    if all(column in frame.columns for column in columns):
        return columns
    raise ValueError(f"Could not find posterior {probability}% HDI columns for prefix '{prefix}'.")


def _plot_summary_curve(ax, frame, x_column, *, prefix=None, color=None, label=None, linewidth=1.8):
    base = "" if prefix is None else f"{prefix}_"
    median_column = f"{base}median"
    lower95, upper95 = _posterior_hdi_columns(frame, prefix, 95)
    lower80, upper80 = _posterior_hdi_columns(frame, prefix, 80)
    line, = ax.plot(
        frame[x_column], frame[median_column], color=color, linewidth=linewidth, label=label, zorder=3
    )
    color = line.get_color()
    ax.fill_between(
        frame[x_column], frame[lower95], frame[upper95], color=color, alpha=HDI95_ALPHA, linewidth=0, zorder=1
    )
    ax.fill_between(
        frame[x_column], frame[lower80], frame[upper80], color=color, alpha=HDI80_ALPHA, linewidth=0, zorder=2
    )
    return line

def _posterior_components(posterior, parameter_specs):
    components = []

    for name, spec in parameter_specs.items():
        if name not in posterior:
            raise ValueError(f"Posterior is missing configured parameter '{name}'.")

        data = posterior[name].squeeze(drop=True)
        extra_dims = tuple(dim for dim in data.dims if dim not in {"chain", "draw"})
        if not extra_dims:
            components.append((name, np.asarray(data, dtype=float).reshape(-1), spec))
            continue

        shape = tuple(data.sizes[dim] for dim in extra_dims)
        for index in np.ndindex(shape):
            indexers = dict(zip(extra_dims, index, strict=True))
            labels = []
            for dim, dim_index in indexers.items():
                if dim in data.coords and data.coords[dim].ndim == 1:
                    value = data.coords[dim].values[dim_index]
                else:
                    value = dim_index
                labels.append(f"{dim}={value}")
            label = f"{name}[{','.join(labels)}]"
            values = np.asarray(data.isel(indexers, drop=True), dtype=float).reshape(-1)
            components.append((label, values, spec))

    return components



__all__ = [name for name in globals() if not name.startswith("__")]
