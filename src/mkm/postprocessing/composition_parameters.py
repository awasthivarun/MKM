"""Posterior mechanism-parameter trends for AgPd all-material models."""

from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

from mkm.models.agpd_basic import (
    _UNIT_INTERVAL_PARAMETERS,
    get_agpd_fixed_parameters,
    get_agpd_model_definition,
    get_agpd_parameterization,
)
from mkm.postprocessing.diagnostics import summarize_samples
from mkm.postprocessing.plotting import (
    FONT_SIZE_AXIS_LABEL,
    FONT_WEIGHT,
    HDI80_ALPHA,
    HDI95_ALPHA,
    TEXT_GAP_PT,
    _add_fixed_gap_global_title,
    _bold_axis_text,
    _nice_linear_ticks,
    _save_presentation_figure,
    _style_legend,
)


# Composition overview layout. These are intentionally local because this figure has a distinct 2 x 2 structure.
COMPOSITION_FIGURE_SIZE_IN = (8.8, 6.6)
COMPOSITION_PANEL_BOX_ASPECT = 0.68
COMPOSITION_HSPACE = 0.30
COMPOSITION_WSPACE = 0.35
COMPOSITION_X_TICKS = (0.00, 0.25, 0.50, 0.75)
COMPOSITION_X_PAD = 0.025
COMPOSITION_X_REFERENCE_MARKER = "D"
COMPOSITION_MARKER_SIZE = 5.0
COMPOSITION_REFERENCE_MARKER_SIZE = 6.3


_PARAMETER_PANELS = {
    "adsorption": (
        ("deltaG1_0", "CO*"),
        ("deltaG4_0", "OH*"),
        ("deltaG5_0", "OH#"),
    ),
    "barriers": (
        ("Gact1_0", "CO ads"),
        ("Gact2_ER_0", "ER"),
        ("Gact2_BF_0", "BF"),
        ("Gact2_LH_0", "LH"),
    ),
    "beta": (
        ("beta_2_ER", "ER"),
        ("beta_2_BF", "BF"),
        ("beta_2", "shared"),
    ),
    "q": (("q", "q"),),
}


def _posterior_dataset(posterior):
    if isinstance(posterior, xr.Dataset):
        return posterior
    if hasattr(posterior, "to_dataset"):
        return posterior.to_dataset()
    raise TypeError("Posterior must be an xarray Dataset or DataTree node.")


def _scalar_draws(posterior, name):
    if name not in posterior:
        raise ValueError(f"Posterior is missing parameter '{name}'.")

    values = np.asarray(posterior[name], dtype=float)
    if values.ndim < 2:
        raise ValueError(f"Posterior parameter '{name}' must contain chain and draw dimensions.")

    trailing_size = int(np.prod(values.shape[2:], dtype=int)) if values.ndim > 2 else 1
    if trailing_size != 1:
        raise ValueError(
            f"Posterior parameter '{name}' must be scalar; found trailing shape {values.shape[2:]}."
        )

    values = values.reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Posterior parameter '{name}' contains non-finite values.")
    return values


def _composition_x_values(config):
    records = [
        {
            "material": material,
            "xAg": float(composition["Ag_fraction"]),
        }
        for material, composition in config["surface_composition"].items()
    ]
    return pd.DataFrame(records).sort_values("xAg").reset_index(drop=True)


def build_agpd_composition_parameter_trends(
    inference_data,
    config,
    model_name,
    parameterization="linear_xAg",
    n_grid=181,
):
    """Summarize effective mechanism parameters across the configured xAg range."""
    posterior = _posterior_dataset(inference_data.posterior)
    definition = get_agpd_model_definition(model_name)
    parameter_names = tuple(field.name for field in fields(definition.parameter_class))
    fixed_parameters = get_agpd_fixed_parameters(model_name)
    n_samples = int(posterior.sizes["chain"]) * int(posterior.sizes["draw"])
    x_reference, slope_specs = get_agpd_parameterization(
        config=config,
        model_name=model_name,
        parameterization=parameterization,
    )
    materials = _composition_x_values(config)
    x_grid = np.linspace(
        float(materials["xAg"].min()),
        float(materials["xAg"].max()),
        int(n_grid),
    )
    x_grid = np.unique(np.concatenate([x_grid, materials["xAg"].to_numpy(dtype=float)]))
    records = []
    for parameter in parameter_names:
        is_fixed = parameter in fixed_parameters
        if is_fixed:
            base = np.full(n_samples, float(fixed_parameters[parameter]), dtype=float)
        else:
            base = _scalar_draws(posterior, parameter)
        is_x_dependent = parameter in slope_specs

        if is_x_dependent:
            slope = _scalar_draws(posterior, f"{parameter}_xAg_slope")
            if parameter in _UNIT_INTERVAL_PARAMETERS:
                if not np.isclose(x_reference, 0.5):
                    raise ValueError("Bounded linear_xAg parameters currently require x_reference = 0.5.")
                max_abs_slope = 2.0 * np.minimum(base, 1.0 - base)
                slope = slope * max_abs_slope
            draws = base[:, None] + slope[:, None] * (x_grid[None, :] - float(x_reference))
        else:
            draws = np.broadcast_to(base[:, None], (len(base), len(x_grid)))
        summary = summarize_samples(draws)
        for index, x_ag in enumerate(x_grid):
            records.append(
                {
                    "parameterization": parameterization,
                    "parameter": parameter,
                    "xAg": float(x_ag),
                    "x_dependent": bool(is_x_dependent),
                    "fixed": bool(is_fixed),
                    "mean": float(summary["mean"][index]),
                    "sd": float(summary["sd"][index]),
                    "median": float(summary["median"][index]),
                    "hdi80_lower": float(summary["hdi80_lower"][index]),
                    "hdi80_upper": float(summary["hdi80_upper"][index]),
                    "hdi95_lower": float(summary["hdi95_lower"][index]),
                    "hdi95_upper": float(summary["hdi95_upper"][index]),
                }
            )
    return pd.DataFrame(records)


def _plot_parameter_panel(ax, trends, parameter_specs, title, ylabel, material_x, x_reference, x_limits):
    for parameter, label in parameter_specs:
        frame = trends.loc[trends["parameter"] == parameter].sort_values("xAg")
        if frame.empty:
            continue
        linestyle = "-" if bool(frame["x_dependent"].iloc[0]) else "--"
        line, = ax.plot(frame["xAg"], frame["median"], linestyle=linestyle, linewidth=2.0, label=label)
        ax.fill_between(
            frame["xAg"], frame["hdi95_lower"], frame["hdi95_upper"],
            color=line.get_color(), alpha=HDI95_ALPHA, linewidth=0,
        )
        ax.fill_between(
            frame["xAg"], frame["hdi80_lower"], frame["hdi80_upper"],
            color=line.get_color(), alpha=HDI80_ALPHA, linewidth=0,
        )

        point_mask = np.isclose(frame["xAg"].to_numpy()[:, None], material_x[None, :]).any(axis=1)
        points = frame.loc[point_mask].sort_values("xAg")
        reference_mask = np.isclose(points["xAg"].to_numpy(dtype=float), float(x_reference))
        for point_frame, marker, marker_size in (
            (points.loc[~reference_mask], "o", COMPOSITION_MARKER_SIZE),
            (points.loc[reference_mask], COMPOSITION_X_REFERENCE_MARKER, COMPOSITION_REFERENCE_MARKER_SIZE),
        ):
            if point_frame.empty:
                continue
            ax.errorbar(
                point_frame["xAg"],
                point_frame["median"],
                yerr=np.vstack(
                    [
                        point_frame["median"] - point_frame["hdi95_lower"],
                        point_frame["hdi95_upper"] - point_frame["median"],
                    ]
                ),
                fmt=marker,
                color=line.get_color(),
                markersize=marker_size,
                linewidth=1.0,
                capsize=2,
                zorder=4 if marker == COMPOSITION_X_REFERENCE_MARKER else 3,
            )

    ax.set_title(title, fontweight=FONT_WEIGHT, pad=TEXT_GAP_PT)
    ax.set_ylabel(ylabel, fontweight=FONT_WEIGHT, labelpad=TEXT_GAP_PT)
    ax.set_box_aspect(COMPOSITION_PANEL_BOX_ASPECT)
    ax.set_xlim(*x_limits)
    ax.set_xticks(COMPOSITION_X_TICKS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelbottom=True)
    ax.grid(False)
    _nice_linear_ticks(ax.yaxis)
    _bold_axis_text(ax)

    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        legend = ax.legend(loc="best", frameon=False)
        _style_legend(legend)


def _add_fixed_gap_xlabel(fig, axes, xlabel):
    xlabel_artist = fig.supxlabel(
        xlabel,
        y=0.0,
        fontweight=FONT_WEIGHT,
        fontsize=FONT_SIZE_AXIS_LABEL,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = TEXT_GAP_PT * fig.dpi / 72.0
    visible_axes = [ax for ax in np.asarray(axes, dtype=object).flat if ax.get_visible()]
    tick_boxes = [
        label.get_window_extent(renderer)
        for ax in visible_axes
        for label in ax.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    if tick_boxes:
        bottom = min(box.y0 for box in tick_boxes)
        label_box = xlabel_artist.get_window_extent(renderer)
        delta_y = bottom - gap_px - label_box.y1
        current_x, current_y = xlabel_artist.get_position()
        xlabel_artist.set_position((current_x, current_y + delta_y / fig.bbox.height))
    return xlabel_artist



def _resolve_error_structure(error_structure, output_path, parameterization):
    if error_structure is not None:
        return str(error_structure)

    # Backward-compatible fallback for existing callers that only pass the output path.
    parts = Path(output_path).parts
    matches = [index for index, part in enumerate(parts) if part == str(parameterization)]
    if matches:
        index = matches[-1]
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return None

def plot_agpd_composition_parameter_overview(
    trends,
    config,
    model_name,
    parameterization,
    output_path: str | Path,
    *,
    error_structure=None,
):
    """Plot physical parameter trends only; error parameters use standard posterior plots."""
    composition = _composition_x_values(config)
    material_x = composition["xAg"].to_numpy(dtype=float)
    x_reference, _ = get_agpd_parameterization(
        config=config,
        model_name=model_name,
        parameterization=parameterization,
    )

    x_min = float(material_x.min())
    x_max = float(material_x.max())
    x_limits = (x_min - COMPOSITION_X_PAD, x_max + COMPOSITION_X_PAD)

    fig, axes = plt.subplots(2, 2, figsize=COMPOSITION_FIGURE_SIZE_IN, sharex=True)
    panels = (
        (axes[0, 0], "adsorption", "Adsorption free energies", r"$\Delta G^0$ (eV)"),
        (axes[0, 1], "barriers", "Activation free energies", r"$G^{\ddagger}$ (eV)"),
        (axes[1, 0], "beta", "Transfer coefficients", r"$\beta$"),
        (axes[1, 1], "q", "Fractional charge transfer", r"$q$"),
    )
    for ax, panel, title, ylabel in panels:
        _plot_parameter_panel(
            ax, trends, _PARAMETER_PANELS[panel], title, ylabel, material_x, x_reference, x_limits
        )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.91,
        hspace=COMPOSITION_HSPACE,
        wspace=COMPOSITION_WSPACE,
    )
    _add_fixed_gap_xlabel(fig, axes, r"Ag composition ($x_{\mathbf{Ag}}$)")

    resolved_error = _resolve_error_structure(error_structure, output_path, parameterization)
    title = f"AgPd: {model_name}"
    if resolved_error is not None:
        title += f" / {resolved_error} error structure"
    _add_fixed_gap_global_title(fig, axes, title)

    _save_presentation_figure(fig, output_path)
    plt.close(fig)
