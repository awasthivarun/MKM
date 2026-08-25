"""Posterior parameter trends for AgPd composition models."""

from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mkm.models.agpd_basic import get_agpd_composition_parameterization, get_agpd_model_definition
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
        raise ValueError(f"Posterior parameter '{name}' must contain chain and draw dimensions.")

    trailing_size = int(np.prod(values.shape[2:], dtype=int)) if values.ndim > 2 else 1
    if trailing_size != 1:
        raise ValueError(f"Posterior parameter '{name}' must be scalar; found trailing shape {values.shape[2:]}")

    values = values.reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Posterior parameter '{name}' contains non-finite values.")
    return values


def _composition_x_values(config):
    records = []
    for material, composition in config["surface_composition"].items():
        records.append(
            {
                "material": material,
                "xAg": float(composition["Ag_fraction"]),
            }
        )
    return pd.DataFrame(records).sort_values("xAg").reset_index(drop=True)


def build_agpd_composition_parameter_trends(
    inference_data,
    config,
    model_name,
    composition_model="linear_xAg",
    n_grid=181,
):
    """Summarize effective mechanism parameters across the observed Ag-composition range."""
    posterior = _posterior_dataset(inference_data.posterior)
    definition = get_agpd_model_definition(model_name)
    parameter_names = tuple(field.name for field in fields(definition.parameter_class))
    x_reference, slope_specs = get_agpd_composition_parameterization(
        config=config,
        model_name=model_name,
        composition_model=composition_model,
    )

    materials = _composition_x_values(config)
    x_min = float(materials["xAg"].min())
    x_max = float(materials["xAg"].max())
    x_grid = np.linspace(x_min, x_max, int(n_grid))
    x_grid = np.unique(np.concatenate([x_grid, materials["xAg"].to_numpy(dtype=float)]))

    records = []
    for parameter in parameter_names:
        base = _scalar_draws(posterior, parameter)
        is_x_dependent = parameter in slope_specs

        if is_x_dependent:
            slope_name = f"{parameter}_xAg_slope"
            slope = _scalar_draws(posterior, slope_name)
            draws = base[:, None] + slope[:, None] * (x_grid[None, :] - float(x_reference))
        else:
            draws = np.broadcast_to(base[:, None], (len(base), len(x_grid)))

        summary = summarize_samples(draws)
        for index, x_ag in enumerate(x_grid):
            records.append(
                {
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


def build_agpd_material_noise_summary(inference_data, config, variable="sigma_ln_rate_material"):
    """Summarize the material-specific residual scale against Ag composition."""
    posterior = _posterior_dataset(inference_data.posterior)
    if variable not in posterior:
        return pd.DataFrame()

    data = posterior[variable]
    if "material" not in data.dims:
        raise ValueError(f"Posterior variable '{variable}' does not have a material dimension.")

    composition = _composition_x_values(config).set_index("material")
    records = []
    for material in data.coords["material"].values.astype(str):
        if material not in composition.index:
            raise ValueError(f"Material '{material}' is missing from surface_composition configuration.")

        values = np.asarray(data.sel(material=material), dtype=float).reshape(-1, 1)
        summary = summarize_samples(values)
        records.append(
            {
                "material": material,
                "xAg": float(composition.loc[material, "xAg"]),
                "variable": variable,
                "mean": float(summary["mean"][0]),
                "sd": float(summary["sd"][0]),
                "median": float(summary["median"][0]),
                "hdi95_lower": float(summary["hdi95_lower"][0]),
                "hdi95_upper": float(summary["hdi95_upper"][0]),
            }
        )

    return pd.DataFrame(records).sort_values("xAg").reset_index(drop=True)


def _plot_parameter_panel(ax, trends, parameter_specs, title, ylabel, material_x):
    for parameter, label in parameter_specs:
        frame = trends.loc[trends["parameter"] == parameter].sort_values("xAg")
        if frame.empty:
            continue

        linestyle = "-" if bool(frame["x_dependent"].iloc[0]) else "--"
        line, = ax.plot(frame["xAg"], frame["median"], linestyle=linestyle, linewidth=2.0, label=label)
        ax.fill_between(
            frame["xAg"],
            frame["hdi95_lower"],
            frame["hdi95_upper"],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

        points = frame.loc[np.isclose(frame["xAg"].to_numpy()[:, None], material_x[None, :]).any(axis=1)]
        points = points.sort_values("xAg")
        yerr = np.vstack(
            [
                points["median"].to_numpy() - points["hdi95_lower"].to_numpy(),
                points["hdi95_upper"].to_numpy() - points["median"].to_numpy(),
            ]
        )
        ax.errorbar(
            points["xAg"],
            points["median"],
            yerr=yerr,
            fmt="o",
            color=line.get_color(),
            markersize=5,
            linewidth=1.0,
            capsize=2,
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Ag Composition ($x_{\mathrm{Ag}}$)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)


def plot_agpd_composition_parameter_overview(
    trends,
    noise_summary,
    config,
    model_name,
    composition_model,
    output_path: str | Path,
):
    """Plot a meeting-style overview analogous to the historical independent-fit figure."""
    if model_name != "CO_BF_ER_LH":
        raise ValueError("The parameter-overview layout is currently defined for CO_BF_ER_LH only.")

    composition = _composition_x_values(config)
    material_x = composition["xAg"].to_numpy(dtype=float)
    x_min, x_max = float(material_x.min()), float(material_x.max())

    fig = plt.figure(figsize=(15.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.0, 1.0))

    ax_ads = fig.add_subplot(grid[:, 0])
    ax_barriers = fig.add_subplot(grid[0, 1])
    ax_beta = fig.add_subplot(grid[0, 2])
    ax_q = fig.add_subplot(grid[1, 1])
    ax_sigma = fig.add_subplot(grid[1, 2])

    ax_ads.set_box_aspect(0.95)
    ax_ads.set_anchor("C")

    _plot_parameter_panel(
        ax_ads,
        trends,
        _PARAMETER_PANELS["adsorption"],
        title="Adsorption Free Energies",
        ylabel=r"$\Delta G^0$ (eV)",
        material_x=material_x,
    )
    ax_ads.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.45)

    _plot_parameter_panel(
        ax_barriers,
        trends,
        _PARAMETER_PANELS["barriers"],
        title="Activation Energy Barriers",
        ylabel=r"$G^{\ddagger}$ (eV)",
        material_x=material_x,
    )
    _plot_parameter_panel(
        ax_beta,
        trends,
        _PARAMETER_PANELS["beta"],
        title="Transfer Coefficients",
        ylabel=r"$\beta$",
        material_x=material_x,
    )
    _plot_parameter_panel(
        ax_q,
        trends,
        _PARAMETER_PANELS["q"],
        title="Fractional Charge Transfer ($q$)",
        ylabel=r"$q$ (–)",
        material_x=material_x,
    )

    if noise_summary.empty:
        ax_sigma.text(0.5, 0.5, "No material-specific residual scale", ha="center", va="center")
    else:
        noise = noise_summary.sort_values("xAg")
        yerr = np.vstack(
            [
                noise["median"].to_numpy() - noise["hdi95_lower"].to_numpy(),
                noise["hdi95_upper"].to_numpy() - noise["median"].to_numpy(),
            ]
        )
        ax_sigma.errorbar(
            noise["xAg"],
            noise["median"],
            yerr=yerr,
            fmt="o",
            markersize=6,
            linewidth=1.2,
            capsize=3,
        )
    ax_sigma.set_title("Residual Scale", fontweight="bold")
    ax_sigma.set_xlabel(r"Ag Composition ($x_{\mathrm{Ag}}$)")
    ax_sigma.set_ylabel(r"$\sigma_{\ln r}$")
    ax_sigma.grid(alpha=0.18)

    for ax in (ax_ads, ax_barriers, ax_beta, ax_q, ax_sigma):
        ax.set_xlim(x_min - 0.025, x_max + 0.025)

    fig.suptitle(
        f"Joint AgPd fit: {model_name}, {composition_model}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Solid = composition-dependent parameter; dashed = shared parameter; bands/error bars = 95% posterior HDI. "
        "Residual scale is material-specific and is not linearly parameterized.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
