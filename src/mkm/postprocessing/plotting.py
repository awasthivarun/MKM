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
from scipy.stats import gaussian_kde, lognorm, norm, truncnorm
from arviz_stats.ecdf_utils import ecdf_pit

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


def _prior_pdf(x, spec):
    distribution = spec["distribution"]

    if distribution == "normal":
        return norm.pdf(x, loc=float(spec["mu"]), scale=float(spec["sigma"]))

    if distribution == "uniform":
        lower, upper = float(spec["lower"]), float(spec["upper"])
        return np.where((x >= lower) & (x <= upper), 1.0 / (upper - lower), 0.0)

    if distribution == "truncated_normal":
        mu, sigma = float(spec["mu"]), float(spec["sigma"])
        lower = float(spec.get("lower", -np.inf))
        upper = float(spec.get("upper", np.inf))
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm.pdf(x, a=a, b=b, loc=mu, scale=sigma)

    if distribution == "lognormal":
        return lognorm.pdf(x, s=float(spec["log_sd"]), scale=float(spec["median"]))

    raise ValueError(f"Unsupported prior distribution '{distribution}'.")


def _prior_support(spec):
    distribution = spec["distribution"]
    if distribution == "uniform":
        return float(spec["lower"]), float(spec["upper"])
    if distribution == "truncated_normal":
        return float(spec.get("lower", -np.inf)), float(spec.get("upper", np.inf))
    if distribution == "lognormal":
        return 0.0, np.inf
    return -np.inf, np.inf


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


def plot_parameter_posteriors(posterior, parameter_specs, output_path: str | Path):
    components = _posterior_components(posterior, parameter_specs)
    if not components:
        raise ValueError("No parameter specifications were provided.")
    ncols = min(3, len(components))
    nrows = math.ceil(len(components) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=_grid_figsize(
            ncols, nrows, panel_width=POSTERIOR_PANEL_WIDTH_IN,
            panel_height=POSTERIOR_PANEL_HEIGHT_IN, extra_height=POSTERIOR_EXTRA_HEIGHT_IN,
        ),
        squeeze=False,
    )

    for ax, (label, values, spec) in zip(axes.flat, components):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Posterior parameter '{label}' contains non-finite values.")
        hdi80 = np.asarray(azs.hdi(values, prob=0.80), dtype=float).reshape(-1)
        hdi95 = np.asarray(azs.hdi(values, prob=0.95), dtype=float).reshape(-1)
        mean = float(np.mean(values))
        lower, upper = np.quantile(values, [0.001, 0.999])
        if np.isclose(lower, upper):
            width = max(abs(float(lower)) * 0.05, 1e-9)
            lower -= width
            upper += width
        posterior_width = max(float(upper - lower), 1e-12)
        support_lower, support_upper = _prior_support(spec)
        pad = 0.08 * posterior_width
        if np.isfinite(support_lower) and support_lower >= lower - 2.0 * posterior_width:
            lower = min(lower, support_lower - pad)
        if np.isfinite(support_upper) and support_upper <= upper + 2.0 * posterior_width:
            upper = max(upper, support_upper + pad)
        x = np.linspace(lower, upper, 600)

        if len(np.unique(values)) > 1:
            posterior_density = gaussian_kde(values)(x)
            posterior_density = np.where((x >= support_lower) & (x <= support_upper), posterior_density, 0.0)
            density_max = float(np.max(posterior_density))
            ax.fill_between(
                x, 0.0, posterior_density, where=(x >= hdi95[0]) & (x <= hdi95[1]),
                color=POSTERIOR_COLOR, alpha=HDI95_ALPHA, interpolate=True,
            )
            ax.fill_between(
                x, 0.0, posterior_density, where=(x >= hdi80[0]) & (x <= hdi80[1]),
                color=POSTERIOR_COLOR, alpha=HDI80_ALPHA, interpolate=True,
            )
            ax.plot(x, posterior_density, color=POSTERIOR_COLOR, linewidth=1.7, label="posterior")
        else:
            density_max = 1.0
            ax.axvline(values[0], color=POSTERIOR_COLOR, linewidth=1.7, label="posterior")

        prior = _prior_pdf(x, spec)
        if np.any(np.isfinite(prior)) and np.nanmax(prior) > 0:
            prior_scaled = prior / np.nanmax(prior) * 0.35 * density_max
            ax.plot(x, prior_scaled, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, alpha=0.75, label="prior")

        ax.axvline(hdi95[0], color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, alpha=0.85)
        ax.axvline(hdi95[1], color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, alpha=0.85)
        ax.axvline(mean, color="#303030", linewidth=1.0, alpha=0.9, zorder=2)
        hdi_width = float(hdi95[1] - hdi95[0])
        text_y = 0.96 * max(density_max, 1e-12)
        for value, horizontal_alignment in ((hdi95[0], "right"), (hdi95[1], "left")):
            ax.annotate(
                _parameter_value_text(float(value), hdi_width),
                xy=(float(value), text_y),
                rotation=90,
                ha=horizontal_alignment,
                va="top",
                fontsize=FONT_SIZE_POSTERIOR_ANNOTATION,
                fontweight=FONT_WEIGHT,
                color="#303030",
            )
        ax.annotate(
            _parameter_value_text(mean, hdi_width),
            xy=(mean, 0.04 * max(density_max, 1e-12)),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_POSTERIOR_ANNOTATION,
            fontweight=FONT_WEIGHT,
            color="#202020",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.0},
            zorder=5,
        )
        ax.set_title(label, pad=TEXT_GAP_PT, fontweight="bold")
        ax.set_yticks([])
        ax.set_box_aspect(POSTERIOR_BOX_ASPECT)
        _nice_linear_ticks(ax.xaxis)
        _bold_axis_text(ax)
        ax.grid(False)

    for ax in axes.flat[len(components):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.025, top=0.938, hspace=0.25, wspace=POSTERIOR_WSPACE)
    _add_fixed_gap_global_title(fig, axes, "Posterior parameters")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def _posterior_interval_columns(frame, prefix=None):
    base = "" if prefix is None else f"{prefix}_"
    columns = (
        f"{base}median",
        f"{base}hdi95_lower",
        f"{base}hdi95_upper",
    )
    if all(column in frame.columns for column in columns):
        return columns
    raise ValueError(
        f"Could not find posterior median/95% interval columns for prefix '{prefix}'."
    )


def plot_observation_grid(
    observations,
    output_path: str | Path,
    residual=False,
    y_scale="log",
    distribution="predictive",
    context_label=None,
):
    if y_scale not in {"linear", "log"}:
        raise ValueError("y_scale must be 'linear' or 'log'.")
    if distribution not in {"model", "predictive"}:
        raise ValueError("distribution must be 'model' or 'predictive'.")
    KOH_values = sorted(observations["electrolyte_concentration_M"].unique())
    CO_values = sorted(observations["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(CO_values), len(KOH_values),
        figsize=_grid_figsize(
            len(KOH_values), len(CO_values), panel_width=OBSERVATION_PANEL_WIDTH_IN,
            panel_height=OBSERVATION_PANEL_HEIGHT_IN, extra_width=OBSERVATION_EXTRA_WIDTH_IN,
            extra_height=OBSERVATION_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    material = _material_from_context(context_label)
    posterior_color, experiment_color = _material_plot_colors(material)
    replicate_names = (
        sorted(str(value) for value in observations["replicate"].unique())
        if "replicate" in observations
        else []
    )
    grey_levels = np.linspace(*RESIDUAL_REPLICATE_GREY_RANGE, max(len(replicate_names), 1))
    replicate_color_map = {
        name: str(float(level)) for name, level in zip(replicate_names, grey_levels, strict=False)
    }

    if not residual:
        prefix = f"rate_{distribution}"
        _posterior_interval_columns(observations, prefix)
    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = observations[
                (observations["electrolyte_concentration_M"] == c_koh)
                & (observations["CO_mole_fraction"] == co_fraction)
            ]
            if residual:
                for replicate, curve in condition.groupby("replicate", sort=True):
                    curve = curve.sort_values("E_V_SHE")
                    ax.plot(
                        curve["E_V_SHE"], curve["residual"], linewidth=REPLICATE_LINEWIDTH, alpha=0.95,
                        color=replicate_color_map.get(str(replicate), "0.45"),
                    )
                ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, alpha=0.7)
            else:
                for _, curve in condition.groupby("replicate", sort=True):
                    curve = curve.sort_values("E_V_SHE")
                    ax.plot(
                        curve["E_V_SHE"], curve["rate"], color=REPLICATE_LINE_COLOR,
                        linewidth=REPLICATE_LINEWIDTH, alpha=REPLICATE_LINE_ALPHA, zorder=2.4,
                    )
                posterior_curve = condition.sort_values("E_V_SHE").drop_duplicates("model_point_id")
                _plot_summary_curve(
                    ax, posterior_curve, "E_V_SHE", prefix=f"rate_{distribution}", color=posterior_color,
                    linewidth=2.0,
                )
                if distribution == "predictive":
                    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8, alpha=0.45)
            if not residual and y_scale == "log":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True, log_y=(not residual and y_scale == "log"))
    _set_presentation_axes(axes)
    display_context = _material_from_context(context_label)
    prefix = f"{display_context}: " if display_context else ""
    if residual:
        title = f"{prefix}Rate residuals"
        ylabel = r"observed rate - posterior model median / s$^{-1}$"
    else:
        rate_title = "Model rate" if distribution == "model" else "Posterior predictive rate"
        title = f"{prefix}{rate_title}"
        ylabel = r"rate / s$^{-1}$"
    _finish_condition_grid(
        fig, axes, title=title, xlabel="Potential (V vs SHE)", ylabel=ylabel, legend=False,
        left=0.085, right=0.92, bottom=0.065, top=0.925, wspace=0.30,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_pointwise_variable(summary, variable_name, output_path: str | Path, context_label=None):
    KOH_values = sorted(summary["electrolyte_concentration_M"].unique())
    CO_values = sorted(summary["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(CO_values), len(KOH_values),
        figsize=_grid_figsize(
            len(KOH_values), len(CO_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=POINTWISE_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = summary[
                (summary["electrolyte_concentration_M"] == c_koh)
                & (summary["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            _plot_summary_curve(ax, condition, "E_V_SHE", color=POSTERIOR_COLOR, linewidth=1.9)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    display_context = _material_from_context(context_label)
    title = f"{display_context}: {variable_name}" if display_context else variable_name
    _finish_condition_grid(
        fig, axes, title=title, xlabel="Potential (V vs SHE)", ylabel=variable_name,
        legend=False, left=0.08, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_pointwise_variables(
    summary,
    variable_names,
    output_path: str | Path,
    *,
    title,
    ylabel,
    colors=None,
    labels=None,
    context_label=None,
    bounded=True,
    observed_rates=None,
):
    variable_names = [variable for variable in variable_names if f"{variable}_median" in summary.columns]
    if not variable_names:
        return False
    colors = {} if colors is None else colors
    labels = {} if labels is None else labels
    KOH_values = sorted(summary["electrolyte_concentration_M"].unique())
    CO_values = sorted(summary["CO_mole_fraction"].unique())
    if observed_rates is not None:
        required = {"electrolyte_concentration_M", "CO_mole_fraction", "E_V_SHE", "rate"}
        missing = sorted(required - set(observed_rates.columns))
        if missing:
            raise ValueError(f"Observed-rate overlay is missing required columns: {missing}.")
    fig, axes = plt.subplots(
        len(CO_values), len(KOH_values),
        figsize=_grid_figsize(
            len(KOH_values), len(CO_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = summary[
                (summary["electrolyte_concentration_M"] == c_koh)
                & (summary["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            for variable in variable_names:
                _plot_summary_curve(
                    ax, condition, "E_V_SHE", prefix=variable, color=colors.get(variable),
                    label=labels.get(variable, variable), linewidth=1.9,
                )
            if observed_rates is not None:
                observed_condition = observed_rates[
                    (observed_rates["electrolyte_concentration_M"] == c_koh)
                    & (observed_rates["CO_mole_fraction"] == co_fraction)
                ]
                if not observed_condition.empty:
                    mean_rate = (
                        observed_condition.groupby("E_V_SHE", as_index=False)["rate"]
                        .mean()
                        .sort_values("E_V_SHE")
                    )
                    rate_values = mean_rate["rate"].to_numpy(dtype=float)
                    if not np.all(np.isfinite(rate_values)):
                        raise ValueError("Observed-rate overlay contains non-finite mean TOF values.")
                    max_rate = float(np.max(rate_values))
                    if max_rate > 0.0:
                        ax.plot(
                            mean_rate["E_V_SHE"], rate_values / max_rate, color=REFERENCE_COLOR,
                            linestyle=":", linewidth=1.5, label="TOF",
                        )
            if bounded:
                ax.set_ylim(-0.02, 1.02)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True, bounded01_y=bounded)
    _set_presentation_axes(axes)
    _add_axis_legend(axes[0, 0], loc="best")
    display_context = _material_from_context(context_label)
    prefix = f"{display_context}: " if display_context else ""
    _finish_condition_grid(
        fig, axes, title=f"{prefix}{title}", xlabel="Potential (V vs SHE)", ylabel=ylabel,
        legend=False, left=0.08, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)
    return True

def plot_sampling_trace(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    nrows = math.ceil(len(plotted_names) / 3)
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        pc = azp.plot_trace(
            data,
            var_names=plotted_names,
            group="posterior",
            backend="matplotlib",
            aes_by_visuals={"divergence": ["color"]},
            visuals={"divergence": {"marker": "|", "size": 30}},
            col_wrap=3,
            figure_kwargs={
                "figsize": (TRACE_FIGURE_WIDTH_IN, TRACE_PANEL_HEIGHT_IN * nrows + TRACE_EXTRA_HEIGHT_IN),
                "layout": "none",
            },
        )
        fig = pc.get_target(plotted_names[0], {}).figure
        for ax in fig.axes:
            ax.grid(False)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
            ax.set_title(ax.get_title(), fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        _apply_trace_chain_alpha(fig, inference_data)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.035, top=0.965, hspace=0.30, wspace=0.20)
        _add_fixed_gap_global_title(fig, fig.axes, "MCMC traces")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)

def _rank_ecdf_outside_envelope(data, variable, *, envelope_prob=0.99, n_simulations=1000):
    """Return rank-ECDF curves and a mask for excursions outside the simultaneous envelope."""
    posterior = data.posterior.to_dataset()[[variable]]
    thinned = posterior.azstats.thin(sample_dims=["draw"])
    sample_size = int(thinned.sizes["draw"])
    n_chains = int(thinned.sizes["chain"])

    ranks = thinned.azstats.compute_ranks(dim=["chain", "draw"])
    ecdf = ranks.azstats.ecdf(dim=["draw"], pit=True, npoints=sample_size)[variable]
    x = np.asarray(ecdf.sel(plot_axis="x"), dtype=float)
    y = np.asarray(ecdf.sel(plot_axis="y"), dtype=float)

    x_ci, _, lower_ci, upper_ci = ecdf_pit(
        np.linspace(0.0, 1.0, sample_size),
        envelope_prob,
        n_simulations=n_simulations,
        n_chains=n_chains,
    )
    lower_ci = np.asarray(lower_ci, dtype=float) - np.asarray(x_ci, dtype=float)
    upper_ci = np.asarray(upper_ci, dtype=float) - np.asarray(x_ci, dtype=float)

    lower_at_x = np.vstack([np.interp(chain_x, x_ci, lower_ci) for chain_x in x])
    upper_at_x = np.vstack([np.interp(chain_x, x_ci, upper_ci) for chain_x in x])
    outside = (y < lower_at_x) | (y > upper_at_x)
    return x, y, outside


def _overlay_rank_envelope_violations(ax, data, variable):
    x, y, outside = _rank_ecdf_outside_envelope(data, variable)
    for chain_index in range(x.shape[0]):
        flagged = np.where(outside[chain_index], y[chain_index], np.nan)
        if np.any(np.isfinite(flagged)):
            ax.step(
                x[chain_index], flagged, where="pre", color="black", linewidth=1.8,
                solid_capstyle="round", zorder=8,
            )


def plot_sampling_rank(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    nrows = math.ceil(len(plotted_names) / 3)
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        pc = azp.plot_rank(
            data,
            var_names=plotted_names,
            group="posterior",
            backend="matplotlib",
            envelope_prob=0.99,
            stats={"ecdf_pit": {"n_simulations": 1000}},
            col_wrap=3,
            visuals={
                "xlabel": False,
                "credible_interval": {"color": "#D9D9D9", "alpha": 0.42},
            },
            figure_kwargs={
                "figsize": (RANK_FIGURE_WIDTH_IN, RANK_PANEL_HEIGHT_IN * nrows + RANK_EXTRA_HEIGHT_IN),
                "layout": "none",
            },
        )
        fig = pc.get_target(plotted_names[0], {}).figure
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        for ax, variable in zip(visible_axes, plotted_names, strict=False):
            _overlay_rank_envelope_violations(ax, data, variable)
        for ax in fig.axes:
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
            ax.set_title(ax.get_title(), fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.03, top=0.965, hspace=0.27, wspace=0.20)
        _add_fixed_gap_global_title(fig, fig.axes, "Chain ranks")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)

def plot_sampling_energy(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        azp.plot_energy(
            data,
            backend="matplotlib",
            show_bfmi=True,
            threshold=0.3,
            figure_kwargs={"figsize": ENERGY_FIGURE_SIZE_IN, "layout": "none"},
        )
        fig = plt.gcf()
        for ax in fig.axes:
            ax.grid(False)
            _nice_linear_ticks(ax.xaxis)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
            ax.set_title(ax.get_title(), fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
            ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
            ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
            legend = ax.get_legend()
            if legend is not None:
                _style_legend(legend)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.20, top=0.78, wspace=0.32)
        _add_fixed_gap_global_title(fig, fig.axes, "Sampling energy and BFMI")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)

def _sampling_parameter_matrix(inference_data, parameter_names):
    data = build_sampling_datatree(inference_data, parameter_names)
    names = list(data.posterior.data_vars)
    if not names:
        raise ValueError("No sampling parameters were available for correlation diagnostics.")
    columns = []
    for name in names:
        values = np.asarray(data.posterior[name], dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Sampling parameter '{name}' contains non-finite values.")
        columns.append(values)
    return names, np.column_stack(columns)


def plot_sampling_correlations(inference_data, parameter_names, output_path: str | Path):
    names, samples = _sampling_parameter_matrix(inference_data, parameter_names)
    correlation = np.corrcoef(samples, rowvar=False)
    n_parameters = len(names)
    size = max(CORRELATION_MIN_SIZE_IN, CORRELATION_SIZE_PER_PARAMETER_IN * n_parameters)
    fig, ax = plt.subplots(figsize=(size, size))
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    image = ax.imshow(np.ma.masked_where(mask, correlation), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(n_parameters), labels=names, rotation=90)
    ax.set_yticks(np.arange(n_parameters), labels=names)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    _bold_axis_text(ax)
    for row in range(n_parameters):
        for col in range(row + 1):
            value = float(correlation[row, col])
            text_color = "white" if abs(value) >= 0.65 else "black"
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center", color=text_color,
                fontsize=CORRELATION_ANNOTATION_SIZE, fontweight=FONT_WEIGHT,
            )
    ax.set_xlim(-0.5, n_parameters - 0.5)
    ax.set_ylim(n_parameters - 0.5, -0.5)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Pearson r", fontsize=FONT_SIZE_AXIS_LABEL, fontweight=FONT_WEIGHT)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontsize(FONT_SIZE_TICK)
        label.set_fontweight(FONT_WEIGHT)
    fig.subplots_adjust(left=0.23, right=0.91, bottom=0.23, top=0.94)
    _add_fixed_gap_global_title(fig, [ax], "Posterior Pearson correlations")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def _simplify_pair_tick_labels(fig):
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if not axes:
        return
    positions = {ax: ax.get_position() for ax in axes}
    min_left = min(position.x0 for position in positions.values())
    min_bottom = min(position.y0 for position in positions.values())
    tolerance = 0.005
    for ax, position in positions.items():
        is_left = abs(position.x0 - min_left) <= tolerance
        is_bottom = abs(position.y0 - min_bottom) <= tolerance
        ax.tick_params(
            axis="x", which="both", bottom=is_bottom, labelbottom=is_bottom,
            top=False, labeltop=False,
        )
        ax.tick_params(
            axis="y", which="both", left=is_left, labelleft=is_left,
            right=False, labelright=False,
        )
        ax.xaxis.get_offset_text().set_visible(is_bottom)
        ax.yaxis.get_offset_text().set_visible(is_left)


def plot_sampling_pairs(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    n_parameters = len(plotted_names)
    size = max(PAIR_MIN_SIZE_IN, PAIR_SIZE_PER_PARAMETER_IN * n_parameters)
    base_visuals = {
        "scatter": {"alpha": 0.12, "s": 5, "rasterized": True},
        "divergence": {"color": "black", "marker": "x", "alpha": 0.9, "s": 16},
    }
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        with azb.rc_context({"plot.max_subplots": max(40, n_parameters**2)}):
            try:
                pm = azp.plot_pair(
                    data,
                    var_names=plotted_names,
                    group="posterior",
                    marginal=True,
                    marginal_kind="kde",
                    triangle="lower",
                    levels=[0.5, 0.9],
                    backend="matplotlib",
                    aes={"color": ["chain"]},
                    aes_by_visuals={"scatter": ["color"]},
                    visuals={**base_visuals, "contour": True},
                    figure_kwargs={"figsize": (size, size), "layout": "none"},
                )
            except ValueError as error:
                message = str(error)
                if "contour" not in message and "levels" not in message and "aesthetic" not in message:
                    raise
                plt.close("all")
                pm = azp.plot_pair(
                    data,
                    var_names=plotted_names,
                    group="posterior",
                    marginal=True,
                    marginal_kind="kde",
                    triangle="lower",
                    backend="matplotlib",
                    aes={"color": ["chain"]},
                    aes_by_visuals={"scatter": ["color"]},
                    visuals=base_visuals,
                    figure_kwargs={"figsize": (size, size), "layout": "none"},
                )
        fig = plt.gcf()
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            ax.grid(False)
            _nice_linear_ticks(ax.xaxis)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
        _simplify_pair_tick_labels(fig)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pm.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        if output_path.suffix.lower() == ".png":
            pm.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
        plt.close("all")

def plot_alpha_comparison(comparison, output_path: str | Path, material):
    """Plot alpha for all KOH/CO conditions in one 4 x 3-style grid."""
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            data = comparison[
                (comparison["C_KOH_M"] == c_koh) & (comparison["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            if not data.empty:
                _plot_summary_curve(ax, data, "E_V_SHE", color=posterior_color, linewidth=2.0)
                ax.errorbar(
                    data["E_V_SHE"], data["alpha_mean"], yerr=data["alpha_sd"], fmt="o",
                    color=experiment_color, markersize=3.4, linewidth=0.9, capsize=1.7, zorder=2.5,
                )
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: transfer coefficient", xlabel="Potential (V vs SHE)",
        ylabel="Transfer coefficient",
        legend=False, left=0.075, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_delta_oh_comparison(comparison, output_path: str | Path, material):
    """Plot OH reaction order as one vertical row per CO fraction."""
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), 1,
        figsize=(OH_ORDER_FIGURE_WIDTH_IN, MATERIAL_PANEL_HEIGHT_IN * len(co_values) + OH_ORDER_EXTRA_HEIGHT_IN),
        sharex=True, sharey=True, squeeze=False
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, co_fraction in enumerate(co_values):
        ax = axes[row, 0]
        data = comparison.loc[comparison["CO_mole_fraction"] == co_fraction].sort_values("E_V_SHE")
        _plot_summary_curve(ax, data, "E_V_SHE", color=posterior_color, linewidth=2.0)
        ax.errorbar(
            data["E_V_SHE"], data["delta_OH"], yerr=data["delta_OH_sd"], fmt="o",
            color=experiment_color, markersize=3.4, linewidth=0.9, capsize=1.7, zorder=2.5,
        )
        ax.text(
            1.02, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
            rotation=-90, va="center", fontweight=FONT_WEIGHT,
        )
        _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: OH reaction order", xlabel="Potential (V vs SHE)", ylabel="OH order",
        legend=False, left=0.20, right=0.84, bottom=0.06, top=0.925, hspace=0.08,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_delta_co_comparison(comparison, output_path: str | Path, material):
    """Plot adjacent CO reaction orders as CO-interval rows by KOH columns."""
    koh_values = sorted(comparison["C_KOH_M"].unique())
    intervals = (
        comparison[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]]
        .drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"])
    )
    interval_values = list(intervals.itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        len(interval_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(interval_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, (lower_co, upper_co) in enumerate(interval_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            data = comparison[
                (comparison["C_KOH_M"] == c_koh)
                & (comparison["CO_lower_mole_fraction"] == lower_co)
                & (comparison["CO_upper_mole_fraction"] == upper_co)
            ].sort_values("E_V_SHE")
            if not data.empty:
                _plot_summary_curve(ax, data, "E_V_SHE", color=posterior_color, linewidth=2.0)
                ax.errorbar(
                    data["E_V_SHE"], data["delta_CO"], yerr=data["delta_CO_sd"], fmt="o",
                    color=experiment_color, markersize=3.4, linewidth=0.9, capsize=1.7, zorder=2.5,
                )
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * lower_co:g}% → {100 * upper_co:g}% CO",
                    transform=ax.transAxes, rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: adjacent CO reaction order", xlabel="Potential (V vs SHE)",
        ylabel="CO order", legend=False, left=0.075, right=0.91, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_second_order_difference(comparison, output_path: str | Path, material):
    """Plot d(alpha)/dE - d(delta_OH)/dE using posterior draws and experimental point estimates."""
    data = comparison.loc[comparison["observable"] == "delta2"].copy()
    if data.empty:
        raise ValueError("No delta2 second-order points were provided.")
    koh_values = sorted(data["C_KOH_M"].dropna().unique())
    co_values = sorted(data["CO_mole_fraction"].dropna().unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = data[
                (data["C_KOH_M"] == c_koh) & (data["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            if not condition.empty:
                _plot_summary_curve(ax, condition, "E_V_SHE", color=posterior_color, linewidth=2.0)
                ax.scatter(
                    condition["E_V_SHE"], condition["experimental"], color=experiment_color,
                    s=18, zorder=2.5,
                )
            ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, alpha=0.65)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: second-order kinetic difference", xlabel="Potential (V vs SHE)",
        ylabel=r"$d\alpha/dE-d\delta_{\mathbf{OH}}/dE$ / V$^{-1}$", legend=False,
        left=0.08, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def _overlay_title(base_title, materials):
    return f"{base_title} across materials" if len(materials) > 1 else base_title


def _summarize_rate_overlay_condition(data):
    if data.empty:
        return data
    required = {
        "E_V_SHE", "rate", "rate_model_median", "rate_model_hdi80_lower", "rate_model_hdi80_upper",
        "rate_model_hdi95_lower", "rate_model_hdi95_upper",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Rate overlay is missing required columns: {missing}.")
    grouped = data.groupby("E_V_SHE", as_index=False).agg(
        experimental_mean=("rate", "mean"),
        experimental_sd=("rate", "std"),
        rate_model_median=("rate_model_median", "first"),
        rate_model_hdi80_lower=("rate_model_hdi80_lower", "first"),
        rate_model_hdi80_upper=("rate_model_hdi80_upper", "first"),
        rate_model_hdi95_lower=("rate_model_hdi95_lower", "first"),
        rate_model_hdi95_upper=("rate_model_hdi95_upper", "first"),
    )
    grouped["experimental_sd"] = grouped["experimental_sd"].fillna(0.0)
    return grouped.sort_values("E_V_SHE")


def _plot_rate_overlay_panel(ax, data, *, color):
    summary = _summarize_rate_overlay_condition(data)
    if summary.empty:
        return
    ax.errorbar(
        summary["E_V_SHE"], summary["experimental_mean"], yerr=summary["experimental_sd"], fmt="o",
        color=color, markersize=2.8, linewidth=0.65, capsize=1.2, alpha=0.72, zorder=2.5,
    )
    _plot_summary_curve(ax, summary, "E_V_SHE", prefix="rate_model", color=color, linewidth=1.65)


def _overlay_condition_legend(fig, condition_labels, *, n_conditions, y=0.952):
    shades = [_mix_with_white("#4D4D4D", amount) for amount in np.linspace(0.58, 0.0, n_conditions)]
    handles = [Line2D([0], [0], color=color, linewidth=2.0) for color in shades]
    legend = fig.legend(
        handles, condition_labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(handles),
        columnspacing=0.8, handlelength=1.5, prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
    )
    _style_legend(legend)
    return legend

def _plot_overlay_panel(ax, data, *, color, observed_column, observed_sd_column=None):
    if data.empty:
        return
    _plot_summary_curve(ax, data, "E_V_SHE", color=color, linewidth=1.65)
    if observed_sd_column is None:
        ax.scatter(data["E_V_SHE"], data[observed_column], color=color, s=12, zorder=2.5)
    else:
        ax.errorbar(
            data["E_V_SHE"], data[observed_column], yerr=data[observed_sd_column], fmt="o",
            color=color, markersize=2.8, linewidth=0.65, capsize=1.2, zorder=2.5,
        )

def _finish_overlay_grid(fig, axes, *, title, ylabel, condition_labels, row_labels=None, wspace=0.09):
    _set_presentation_axes(axes)
    for ax in np.asarray(axes, dtype=object).flat:
        _style_axis(ax, potential_x=True)
    if row_labels is not None:
        for row, label in enumerate(row_labels):
            axes[row, -1].text(
                1.0, 0.5, label, transform=axes[row, -1].transAxes, rotation=-90,
                va="center", fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE,
            )
    legend = _overlay_condition_legend(fig, condition_labels, n_conditions=len(condition_labels), y=0.955)
    fig.subplots_adjust(left=0.05, right=0.965, bottom=0.07, top=0.875, hspace=0.09, wspace=wspace)
    _add_fixed_gap_global_labels(fig, axes, xlabel="Potential (V vs SHE)", ylabel=ylabel)
    _position_row_labels(fig, axes)
    _position_legend_above_column_titles(fig, legend, axes)
    _add_fixed_gap_global_title(fig, axes, title, legend=legend)

def plot_alpha_overlay_pco(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(koh_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(koh_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, c_koh in enumerate(koh_values):
            for color, co_fraction in zip(colors, co_values, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["C_KOH_M"] == c_koh)
                    & (comparison["CO_mole_fraction"] == co_fraction)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="alpha_mean", observed_sd_column="alpha_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Transfer coefficient", materials), ylabel="Transfer coefficient",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_alpha_overlay_koh(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(co_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, co_fraction in enumerate(co_values):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["CO_mole_fraction"] == co_fraction)
                    & (comparison["C_KOH_M"] == c_koh)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="alpha_mean", observed_sd_column="alpha_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Transfer coefficient", materials), ylabel="Transfer coefficient",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_delta_co_overlay_pco(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    intervals = list(
        comparison[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]].drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"]).itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(
        len(koh_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(koh_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(intervals))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, c_koh in enumerate(koh_values):
            for color, (lower_co, upper_co) in zip(colors, intervals, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["C_KOH_M"] == c_koh)
                    & (comparison["CO_lower_mole_fraction"] == lower_co)
                    & (comparison["CO_upper_mole_fraction"] == upper_co)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="delta_CO", observed_sd_column="delta_CO_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Adjacent CO reaction order", materials), ylabel="CO order",
        condition_labels=[f"{100 * lo:g}–{100 * hi:g}% CO" for lo, hi in intervals],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_delta_co_overlay_koh(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    intervals = list(
        comparison[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]].drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"]).itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(
        len(intervals), len(materials),
        figsize=_grid_figsize(
            len(materials), len(intervals), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, (lower_co, upper_co) in enumerate(intervals):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["C_KOH_M"] == c_koh)
                    & (comparison["CO_lower_mole_fraction"] == lower_co)
                    & (comparison["CO_upper_mole_fraction"] == upper_co)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="delta_CO", observed_sd_column="delta_CO_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Adjacent CO reaction order", materials), ylabel="CO order",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * lo:g}–{100 * hi:g}% CO" for lo, hi in intervals],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_delta_oh_overlay_pco(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        1, len(materials),
        figsize=(OVERLAY_PANEL_WIDTH_IN * len(materials) + OVERLAY_EXTRA_WIDTH_IN, OVERLAY_SINGLE_ROW_HEIGHT_IN),
        sharex=True, sharey=True, squeeze=False
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for color, co_fraction in zip(colors, co_values, strict=True):
            data = comparison[
                (comparison["material"] == material) & (comparison["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            _plot_overlay_panel(
                axes[0, col], data, color=color, observed_column="delta_OH", observed_sd_column="delta_OH_sd"
            )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("OH reaction order", materials), ylabel="OH order",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_rate_overlay_pco(observations, output_path: str | Path):
    materials = _ordered_materials(observations["material"].unique())
    koh_values = sorted(observations["electrolyte_concentration_M"].unique())
    co_values = sorted(observations["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(koh_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(koh_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight=FONT_WEIGHT)
        for row, c_koh in enumerate(koh_values):
            for color, co_fraction in zip(colors, co_values, strict=True):
                data = observations[
                    (observations["material"] == material)
                    & (observations["electrolyte_concentration_M"] == c_koh)
                    & (observations["CO_mole_fraction"] == co_fraction)
                ]
                _plot_rate_overlay_panel(axes[row, col], data, color=color)
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Model rate", materials), ylabel=r"rate / s$^{-1}$",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values], wspace=0.24,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_rate_overlay_koh(observations, output_path: str | Path):
    materials = _ordered_materials(observations["material"].unique())
    koh_values = sorted(observations["electrolyte_concentration_M"].unique())
    co_values = sorted(observations["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(co_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight=FONT_WEIGHT)
        for row, co_fraction in enumerate(co_values):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = observations[
                    (observations["material"] == material)
                    & (observations["CO_mole_fraction"] == co_fraction)
                    & (observations["electrolyte_concentration_M"] == c_koh)
                ]
                _plot_rate_overlay_panel(axes[row, col], data, color=color)
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Model rate", materials), ylabel=r"rate / s$^{-1}$",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values], wspace=0.24,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_loo_comparison(compare_table, output_path: str | Path):
    frame = compare_table.set_index("model")

    pc = azp.plot_compare(
        frame,
        relative_scale=True,
        rotated=True,
        backend="matplotlib",
        visuals={"similar_line": True},
    )
    pc.add_title("PSIS-LOO model comparison")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_pareto_k(loo_result, model_name, output_path: str | Path):
    pc = azp.plot_khat(
        loo_result,
        threshold=float(loo_result.good_k),
        backend="matplotlib",
        visuals={"hlines": True, "bin_text": True},
    )
    pc.add_title(f"Pareto-k diagnostics: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_pointwise_elpd_difference(frame, numerator_model, denominator_model, output_path: str | Path):
    comparison = f"{numerator_model}_minus_{denominator_model}"
    data = frame.loc[frame["comparison"] == comparison].copy()
    if data.empty:
        raise ValueError(f"No pointwise ELPD comparison found for '{comparison}'.")
    koh_values = sorted(data["electrolyte_concentration_M"].unique())
    co_values = sorted(data["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_PANEL_WIDTH_IN,
            panel_height=LOO_PANEL_HEIGHT_IN, extra_height=LOO_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    max_abs = float(np.max(np.abs(data["elpd_difference"])))
    y_limit = 1.05 * max_abs if max_abs > 0 else 1.0
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = data[
                (data["electrolyte_concentration_M"] == c_koh)
                & (data["CO_mole_fraction"] == co_fraction)
            ]
            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["elpd_difference"], marker="o", markersize=2.5,
                    linewidth=1.0, alpha=0.8, label=replicate,
                )
            ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.55, color=REFERENCE_COLOR)
            ax.set_ylim(-y_limit, y_limit)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    legend = axes[0, 0].legend(
        title="replicate", prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND}
    )
    _style_legend(legend)
    _finish_condition_grid(
        fig, axes, title="Pointwise PSIS-LOO difference", xlabel="Potential (V vs SHE)",
        ylabel=f"Pointwise ELPD: {numerator_model} - {denominator_model}", legend=False,
        left=0.075, right=0.93, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_pointwise_loo(pointwise, model_name, output_path: str | Path, context_label=None):
    """Plot observation-level PSIS-LOO contributions for one material."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_PANEL_WIDTH_IN,
            panel_height=LOO_PANEL_HEIGHT_IN, extra_height=LOO_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = pointwise[
                (pointwise["electrolyte_concentration_M"] == c_koh)
                & (pointwise["CO_mole_fraction"] == co_fraction)
            ]
            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["elpd_loo"], marker="o", markersize=2.5,
                    linewidth=1.0, alpha=0.8, label=replicate,
                )
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    legend = axes[0, 0].legend(
        title="replicate", prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND}
    )
    _style_legend(legend)
    prefix = f"{context_label}: " if context_label else ""
    _finish_condition_grid(
        fig, axes, title=f"{prefix}Pointwise PSIS-LOO", xlabel="Potential (V vs SHE)",
        ylabel="Pointwise PSIS-LOO ELPD", legend=False, left=0.075, right=0.93, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_pointwise_loo_pit(pointwise_loo, pointwise_pit, output_path: str | Path, context_label=None):
    """Plot pointwise PSIS-LOO ELPD and LOO-PIT together, pairing replicates by color."""
    required_loo = {"electrolyte_concentration_M", "CO_mole_fraction", "replicate", "E_V_SHE", "elpd_loo"}
    required_pit = {"electrolyte_concentration_M", "CO_mole_fraction", "replicate", "E_V_SHE", "loo_pit"}
    missing_loo = sorted(required_loo - set(pointwise_loo.columns))
    missing_pit = sorted(required_pit - set(pointwise_pit.columns))
    if missing_loo or missing_pit:
        raise ValueError(f"Combined LOO/PIT plot is missing columns: LOO={missing_loo}, PIT={missing_pit}.")

    koh_values = sorted(pointwise_loo["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise_loo["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_POINTWISE_PANEL_WIDTH_IN,
            panel_height=LOO_POINTWISE_PANEL_HEIGHT_IN, extra_width=LOO_POINTWISE_EXTRA_WIDTH_IN,
            extra_height=LOO_POINTWISE_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    twin_axes = np.empty_like(axes, dtype=object)
    replicates = sorted(set(pointwise_loo["replicate"].astype(str)) | set(pointwise_pit["replicate"].astype(str)))
    color_map = {replicate: DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index, replicate in enumerate(replicates)}

    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        for row, co_fraction in enumerate(co_values):
            for col, c_koh in enumerate(koh_values):
                ax = axes[row, col]
                ax_pit = ax.twinx()
                twin_axes[row, col] = ax_pit
                loo_condition = pointwise_loo[
                    (pointwise_loo["electrolyte_concentration_M"] == c_koh)
                    & (pointwise_loo["CO_mole_fraction"] == co_fraction)
                ]
                pit_condition = pointwise_pit[
                    (pointwise_pit["electrolyte_concentration_M"] == c_koh)
                    & (pointwise_pit["CO_mole_fraction"] == co_fraction)
                ]
                for replicate in replicates:
                    color = color_map[replicate]
                    loo_curve = loo_condition.loc[
                        loo_condition["replicate"].astype(str) == replicate
                    ].sort_values("E_V_SHE")
                    pit_curve = pit_condition.loc[
                        pit_condition["replicate"].astype(str) == replicate
                    ].sort_values("E_V_SHE")
                    if not loo_curve.empty:
                        ax.plot(
                            loo_curve["E_V_SHE"], loo_curve["elpd_loo"], color=color,
                            linewidth=1.0, marker="o", markersize=2.2, alpha=0.85,
                        )
                    if not pit_curve.empty:
                        ax_pit.plot(
                            pit_curve["E_V_SHE"], pit_curve["loo_pit"], color=color,
                            linewidth=1.0, linestyle="--", alpha=0.80,
                        )
                ax_pit.set_ylim(0.0, 1.0)
                ax_pit.set_yticks([0.0, 0.5, 1.0])
                ax_pit.grid(False)
                if col != len(koh_values) - 1:
                    ax_pit.tick_params(axis="y", right=False, labelright=False)
                else:
                    for label in ax_pit.get_yticklabels():
                        label.set_fontweight(FONT_WEIGHT)
                        label.set_fontsize(FONT_SIZE_TICK)
                if row == 0:
                    ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
                if col == len(koh_values) - 1:
                    ax.text(
                        1.17, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                        rotation=-90, va="center", fontweight=FONT_WEIGHT,
                    )
                _style_axis(ax, potential_x=True)

        legend_handles = [
            Line2D([0], [0], color=DEFAULT_COLORS[0], linewidth=1.5, marker="o", markersize=3, label="PSIS-LOO ELPD"),
            Line2D([0], [0], color=DEFAULT_COLORS[0], linewidth=1.5, linestyle="--", label="LOO-PIT"),
        ]
        legend = axes[0, 0].legend(
            handles=legend_handles, loc="best", ncol=1,
            prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
        )
        _style_legend(legend)
        prefix = f"{context_label}: " if context_label else ""
        fig.subplots_adjust(left=0.075, right=0.91, bottom=0.10, top=0.90, hspace=0.14, wspace=0.22)
        _add_fixed_gap_global_labels(
            fig, axes, xlabel="Potential (V vs SHE)", ylabel="Pointwise PSIS-LOO ELPD"
        )
        _position_row_labels(fig, axes)
        _add_fixed_gap_right_ylabel(fig, twin_axes[:, -1], axes[:, -1], ylabel="LOO-PIT")
        _add_fixed_gap_global_title(fig, axes, f"{prefix}Pointwise PSIS-LOO and LOO-PIT")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)


def _plot_loo_pit_axis(ax, loo_pit, *, coverage, title):
    values = np.asarray(loo_pit, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("LOO-PIT values are empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("LOO-PIT values contain non-finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("LOO-PIT values must lie inside [0, 1].")
    if coverage:
        values = 2.0 * np.abs(values - 0.5)
    distribution = xr.Dataset({"rate_observed": ("observation", values)})
    ecdf = distribution.azstats.ecdf(dim=["observation"], pit=True, npoints=len(values))
    uniformity_result = distribution.azstats.uniformity_test(dim=["observation"], method="pot_c")
    if len(uniformity_result) < 2:
        raise ValueError("Uniformity test did not return p-values and pointwise contributions.")
    p_values = uniformity_result[0]
    shapley_values = uniformity_result[1]
    ecdf_values = ecdf["rate_observed"]
    x = np.asarray(ecdf_values.sel(plot_axis="x"), dtype=float)
    y = np.asarray(ecdf_values.sel(plot_axis="y"), dtype=float)
    p_value = float(np.asarray(p_values["rate_observed"], dtype=float).reshape(-1)[0])
    shapley = np.asarray(shapley_values["rate_observed"], dtype=float).reshape(-1)
    alpha = 0.05
    expected_max = np.sqrt(np.log(2.0 / alpha) / (2.0 * len(values))) * 1.3
    actual_max = float(np.max(np.abs(y)))
    epsilon = max(expected_max, actual_max)
    suspicious = (shapley > 0.0) & (p_value < alpha)
    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.65)
    ax.step(x, y, where="pre", color=DEFAULT_COLORS[0], linewidth=1.5)
    if np.any(suspicious):
        ax.scatter(x[suspicious], y[suspicious], color=DEFAULT_COLORS[1], s=22, marker="x", linewidths=1.2)
    ax.text(
        0.01, 0.92, f"p={p_value:.2f} (α={alpha:.2f})", transform=ax.transAxes,
        va="top", fontweight=FONT_WEIGHT,
    )
    ax.set_ylim(-epsilon, epsilon)
    ax.set_ylabel(r"$\Delta$ ECDF", fontweight="bold")
    ax.set_title(title, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
    if coverage:
        ax.set_xticks([0.0, 0.5, 1.0], labels=["0", "50", "100"])
        ax.set_xlabel("ETI %", fontweight="bold")
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel("PIT", fontweight="bold")
    _nice_linear_ticks(ax.yaxis)
    _bold_axis_text(ax)
    ax.grid(False)


def _plot_pareto_k_axis(ax, loo_result, *, title):
    pareto_k = loo_result.pareto_k
    if isinstance(pareto_k, xr.Dataset):
        if len(pareto_k.data_vars) != 1:
            raise ValueError("Pareto-k dataset must contain exactly one variable.")
        pareto_k = pareto_k[next(iter(pareto_k.data_vars))]
    values = np.asarray(pareto_k, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Pareto-k values contain no finite entries.")
    indices = np.arange(len(values))
    good_k = float(loo_result.good_k)
    acceptable = finite & (values <= good_k)
    problematic = finite & (values > good_k)
    ax.plot(
        indices[acceptable], values[acceptable], color=DEFAULT_COLORS[0],
        linestyle="none", marker=".", markersize=4.0,
    )
    if np.any(problematic):
        ax.plot(
            indices[problematic], values[problematic], color=DEFAULT_COLORS[1],
            linestyle="none", marker="o", markersize=3.5,
        )
    ax.axhline(good_k, color=REFERENCE_COLOR, linestyle="--", linewidth=1.1, label=f"good k = {good_k:g}")
    for level in (0.5, 0.7, 1.0):
        if not np.isclose(level, good_k):
            ax.axhline(level, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Observation index", fontweight="bold")
    ax.set_ylabel("Pareto k", fontweight="bold")
    ax.set_title(title, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
    legend = ax.legend(prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND})
    _style_legend(legend)
    _style_axis(ax)


def plot_loo_pit_ecdf(loo_pit, model_name, output_path: str | Path, context_label=None):
    from mkm.postprocessing.calibration import build_loo_pit_datatree

    data = build_loo_pit_datatree(loo_pit)
    pc = azp.plot_ecdf_pit(
        data,
        var_names=["rate_observed"],
        group="loo_pit",
        sample_dims=["observation"],
        method="pot_c",
        envelope_prob=0.95,
        coverage=False,
        backend="matplotlib",
    )
    prefix = f"{context_label}: " if context_label else ""
    pc.add_title(f"{prefix}LOO-PIT calibration: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_loo_pit_coverage(loo_pit, model_name, output_path: str | Path, context_label=None):
    from mkm.postprocessing.calibration import build_loo_pit_datatree

    data = build_loo_pit_datatree(loo_pit)
    pc = azp.plot_ecdf_pit(
        data,
        var_names=["rate_observed"],
        group="loo_pit",
        sample_dims=["observation"],
        method="pot_c",
        envelope_prob=0.95,
        coverage=True,
        backend="matplotlib",
    )
    prefix = f"{context_label}: " if context_label else ""
    pc.add_title(f"{prefix}LOO predictive coverage: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_loo_pit_summary(loo_pit, model_name, output_path: str | Path, context_label=None):
    fig, axes = plt.subplots(2, 1, figsize=LOO_PIT_SUMMARY_SIZE_IN)
    _plot_loo_pit_axis(axes[0], loo_pit, coverage=True, title="Predictive coverage")
    _plot_loo_pit_axis(axes[1], loo_pit, coverage=False, title="LOO-PIT calibration")
    prefix = f"{context_label}: " if context_label else ""
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.90, hspace=0.58)
    _add_fixed_gap_global_title(fig, axes, f"{prefix}LOO-PIT diagnostics")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_loo_diagnostics(loo_result, loo_pit, model_name, output_path: str | Path, context_label=None):
    fig, axes = plt.subplots(3, 1, figsize=LOO_DIAGNOSTICS_SIZE_IN)
    _plot_pareto_k_axis(axes[0], loo_result, title="Pareto-k")
    if loo_pit is None:
        for ax, title in zip(axes[1:], ("Predictive coverage", "LOO-PIT calibration"), strict=True):
            ax.set_title(title, fontweight="bold")
            ax.text(
                0.5, 0.5, "LOO-PIT unavailable", ha="center", va="center",
                transform=ax.transAxes, fontweight=FONT_WEIGHT,
            )
            ax.set_axis_off()
    else:
        _plot_loo_pit_axis(axes[1], loo_pit, coverage=True, title="Predictive coverage")
        _plot_loo_pit_axis(axes[2], loo_pit, coverage=False, title="LOO-PIT calibration")
    prefix = f"{context_label}: " if context_label else ""
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.065, top=0.92, hspace=0.66)
    _add_fixed_gap_global_title(fig, axes, f"{prefix}LOO diagnostics")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_loo_pit_conditions(pointwise, model_name, output_path: str | Path, context_label=None):
    """Plot potential-resolved raw LOO-PIT values for one material."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_PANEL_WIDTH_IN,
            panel_height=LOO_PANEL_HEIGHT_IN, extra_height=LOO_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = pointwise[
                (pointwise["electrolyte_concentration_M"] == c_koh)
                & (pointwise["CO_mole_fraction"] == co_fraction)
            ]
            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["loo_pit"], marker="o", markersize=2.5,
                    linewidth=0.8, alpha=0.8, label=replicate,
                )
            ax.axhline(0.50, linestyle="--", linewidth=1.0, alpha=0.60, color=REFERENCE_COLOR)
            ax.axhline(0.05, linestyle=":", linewidth=0.8, alpha=0.40, color=REFERENCE_COLOR)
            ax.axhline(0.95, linestyle=":", linewidth=0.8, alpha=0.40, color=REFERENCE_COLOR)
            ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True, bounded01_y=True)
    legend = axes[0, 0].legend(
        title="replicate", prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND}
    )
    _style_legend(legend)
    prefix = f"{context_label}: " if context_label else ""
    _finish_condition_grid(
        fig, axes, title=f"{prefix}Condition-resolved LOO-PIT", xlabel="Potential (V vs SHE)", ylabel="LOO-PIT",
        legend=False, left=0.075, right=0.93, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_transition_state_drc(summary, model_name, output_path: str | Path):
    koh_values = sorted(summary["electrolyte_concentration_M"].unique())
    co_values = sorted(summary["CO_mole_fraction"].unique())
    controls = summary[["control", "label"]].drop_duplicates().itertuples(index=False)
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for control, label in controls:
        control_data = summary.loc[summary["control"] == control]
        for row, co_fraction in enumerate(co_values):
            for col, c_koh in enumerate(koh_values):
                ax = axes[row, col]
                condition = control_data[
                    (control_data["electrolyte_concentration_M"] == c_koh)
                    & (control_data["CO_mole_fraction"] == co_fraction)
                ].sort_values("E_V_SHE")
                _plot_summary_curve(
                    ax, condition, "E_V_SHE", color=PATHWAY_COLORS.get(control), label=label, linewidth=1.8
                )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8, alpha=0.55)
            ax.axhline(1.0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8, alpha=0.45)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _add_axis_legend(axes[0, 0], loc="best")
    _finish_condition_grid(
        fig, axes, title="Transition-state degree of rate control", xlabel="Potential (V vs SHE)",
        ylabel=r"$X_{\mathbf{TS}}$", legend=False, left=0.075, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)
