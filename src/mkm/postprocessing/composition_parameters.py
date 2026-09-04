"""Posterior mechanism-parameter trends for AgPd all-material models."""

from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mkm.models.agpd_basic import (
    _UNIT_INTERVAL_PARAMETERS,
    get_agpd_model_definition,
    get_agpd_parameterization,
)
from mkm.postprocessing.diagnostics import summarize_samples


_PARAMETER_PANELS = {
    "adsorption": (
        ("deltaG1_0", r"$\Delta G_1^0$ (CO*)"),
        ("deltaG4_0", r"$\Delta G_4^0$ (OH*)"),
        ("deltaG5_0", r"$\Delta G_5^0$ (OH#)"),
    ),
    "barriers": (
        ("Gact1_0", r"$G_1^{\ddagger}$"),
        ("Gact2_ER_0", r"$G_{\mathrm{ER}}^{\ddagger}$"),
        ("Gact2_BF_0", r"$G_{\mathrm{BF}}^{\ddagger}$"),
        ("Gact2_LH_0", r"$G_{\mathrm{LH}}^{\ddagger}$"),
    ),
    "beta": (
        ("beta_2_ER", r"$\beta_{2,\mathrm{ER}}$"),
        ("beta_2_BF", r"$\beta_{2,\mathrm{BF}}$"),
        ("beta_2", r"$\beta_2$"),
    ),
    "q": (("q", r"$q$"),),
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
        raise ValueError(
            f"Posterior parameter '{name}' must contain chain and draw dimensions."
        )

    trailing_size = int(np.prod(values.shape[2:], dtype=int)) if values.ndim > 2 else 1
    if trailing_size != 1:
        raise ValueError(
            f"Posterior parameter '{name}' must be scalar; found trailing shape "
            f"{values.shape[2:]}."
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
    x_grid = np.unique(
        np.concatenate([x_grid, materials["xAg"].to_numpy(dtype=float)])
    )

    records = []
    for parameter in parameter_names:
        base = _scalar_draws(posterior, parameter)
        is_x_dependent = parameter in slope_specs

        if is_x_dependent:
            slope = _scalar_draws(posterior, f"{parameter}_xAg_slope")

            if parameter in _UNIT_INTERVAL_PARAMETERS:
                if not np.isclose(x_reference, 0.5):
                    raise ValueError(
                        "Bounded linear_xAg parameters currently require x_reference = 0.5."
                    )

                max_abs_slope = 2.0 * np.minimum(base, 1.0 - base)
                slope = slope * max_abs_slope

            draws = base[:, None] + slope[:, None] * (
                x_grid[None, :] - float(x_reference)
            )
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
                    "mean": float(summary["mean"][index]),
                    "sd": float(summary["sd"][index]),
                    "median": float(summary["median"][index]),
                    "hdi95_lower": float(summary["hdi95_lower"][index]),
                    "hdi95_upper": float(summary["hdi95_upper"][index]),
                }
            )

    return pd.DataFrame(records)


def _plot_parameter_panel(ax, trends, parameter_specs, title, ylabel, material_x):
    for parameter, label in parameter_specs:
        frame = trends.loc[trends["parameter"] == parameter].sort_values("xAg")
        if frame.empty:
            continue

        linestyle = "-" if bool(frame["x_dependent"].iloc[0]) else "--"
        line, = ax.plot(
            frame["xAg"],
            frame["median"],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        ax.fill_between(
            frame["xAg"],
            frame["hdi95_lower"],
            frame["hdi95_upper"],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

        point_mask = np.isclose(
            frame["xAg"].to_numpy()[:, None],
            material_x[None, :],
        ).any(axis=1)
        points = frame.loc[point_mask].sort_values("xAg")
        ax.errorbar(
            points["xAg"],
            points["median"],
            yerr=np.vstack(
                [
                    points["median"] - points["hdi95_lower"],
                    points["hdi95_upper"] - points["median"],
                ]
            ),
            fmt="o",
            color=line.get_color(),
            markersize=5,
            linewidth=1.0,
            capsize=2,
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Ag composition ($x_{\mathrm{Ag}}$)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)


def plot_agpd_composition_parameter_overview(
    trends,
    config,
    model_name,
    parameterization,
    output_path: str | Path,
):
    """Plot physical parameter trends only; error parameters use standard posterior plots."""
    composition = _composition_x_values(config)
    material_x = composition["xAg"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    panels = (
        (axes[0, 0], "adsorption", "Adsorption free energies", r"$\Delta G^0$ (eV)"),
        (axes[0, 1], "barriers", "Activation barriers", r"$G^{\ddagger}$ (eV)"),
        (axes[1, 0], "beta", "Transfer coefficients", r"$\beta$"),
        (axes[1, 1], "q", "Fractional charge transfer", r"$q$"),
    )
    for ax, panel, title, ylabel in panels:
        _plot_parameter_panel(
            ax,
            trends,
            _PARAMETER_PANELS[panel],
            title,
            ylabel,
            material_x,
        )

    fig.suptitle(
        f"Joint AgPd fit: {model_name}, {parameterization}",
        fontsize=15,
        fontweight="bold",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
