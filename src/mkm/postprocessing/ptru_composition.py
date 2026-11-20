"""Physical-parameter trends inferred from pooled PtRu posteriors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mkm.models.ptru import ptru_parameter_names
from mkm.models.composition_bounds import bounded_linear_slope_scale
from mkm.postprocessing.diagnostics import summarize_samples


_PANELS = (
    ("Adsorption free energies", r"$\Delta G^0$ (eV)", ("deltaG1_0", "deltaG4_0", "deltaG5_0")),
    ("Activation free energies", r"$G^{\ddagger}$ (eV)",
     ("Gact1_0", "Gact2_LH_0", "Gact2_ER_0", "Gact2_BF_0")),
    ("Transfer coefficients", r"$\beta$", ("beta_2_ER",)),
)


def _scalar_draws(posterior, name):
    if name not in posterior:
        raise ValueError(f"PtRu posterior is missing '{name}'.")
    variable = posterior[name]
    if set(variable.dims) != {"chain", "draw"}:
        raise ValueError(f"Expected scalar sampled parameter '{name}', got dimensions {variable.dims}.")
    values = np.asarray(variable.transpose("chain", "draw"), dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"PtRu parameter '{name}' contains nonfinite draws.")
    return values


def build_ptru_composition_parameter_trends(inference_data, config, prior_config, specification, n_grid=181):
    """Summarize shared, independent or linear_xRu posteriors over observed nominal Ru fractions."""
    if not specification.is_all_materials:
        raise ValueError("PtRu composition plotting requires an all-material posterior.")
    posterior = inference_data.posterior
    if not isinstance(posterior, xr.Dataset):
        posterior = posterior.to_dataset()
    if n_grid < 2:
        raise ValueError("n_grid must be at least two.")

    compositions = {name: float(config["surface_composition"][name]["Ru_fraction"])
                    for name in specification.materials}
    ordered_materials = sorted(compositions, key=compositions.get)
    x_values = np.array([compositions[name] for name in ordered_materials])
    parameterization = specification.parameterization
    parameters = ptru_parameter_names(specification.model_name)
    x_reference = float(config["parameterizations"]["linear_xRu"]["x_reference"])
    x_grid = np.unique(np.concatenate([np.linspace(x_values.min(), x_values.max(), n_grid), x_values]))
    profile = prior_config["prior_profiles"][specification.medium]["parameters"]
    rows = []

    for name in parameters:
        if parameterization == "independent":
            for material in ordered_materials:
                if name not in ptru_parameter_names(specification.model_name, material):
                    continue
                draws = _scalar_draws(posterior, f"{name}_{material}")[:, None]
                stats = summarize_samples(draws)
                rows.append({
                    "parameterization": parameterization, "parameter": name, "material": material,
                    "xRu": compositions[material], "independent": True, "x_dependent": True,
                    "active": True,
                    **{key: float(values[0]) for key, values in stats.items()},
                })
            continue

        base = _scalar_draws(posterior, name)
        changing = parameterization == "linear_xRu"
        if changing:
            slope = _scalar_draws(posterior, f"{name}_xRu_slope")
            if name == "beta_2_ER":
                limits = profile[name]
                slope *= bounded_linear_slope_scale(
                    base, float(limits["lower"]), float(limits["upper"]), x_reference,
                    0.0, 0.5, minimum=np.minimum,
                )
            draws = base[:, None] + slope[:, None] * (x_grid[None, :] - x_reference)
        else:
            draws = np.broadcast_to(base[:, None], (len(base), len(x_grid)))
        stats = summarize_samples(draws)
        for index, x_value in enumerate(x_grid):
            material = next((label for label, fraction in compositions.items()
                             if np.isclose(fraction, x_value, atol=1e-12)), None)
            active = material is None or name in ptru_parameter_names(specification.model_name, material)
            rows.append({
                "parameterization": parameterization, "parameter": name, "material": material,
                "xRu": float(x_value), "independent": False, "x_dependent": changing, "active": active,
                **{key: float(values[index]) for key, values in stats.items()},
            })
    return pd.DataFrame(rows)


def plot_ptru_composition_parameter_overview(trends, specification, output_path: str | Path):
    """One panel per physical parameter category; error parameters are intentionally excluded."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    independent = specification.parameterization == "independent"
    for ax, (title, ylabel, names) in zip(axes, _PANELS):
        for name in names:
            subset = trends.loc[trends["parameter"] == name].sort_values("xRu")
            if subset.empty:
                continue
            if independent:
                line, = ax.plot(subset["xRu"], subset["median"], "-o", label=name, markersize=4)
                ax.errorbar(subset["xRu"], subset["median"],
                            yerr=[subset["median"] - subset["hdi95_lower"],
                                  subset["hdi95_upper"] - subset["median"]],
                            fmt="none", color=line.get_color(), capsize=2)
            else:
                line, = ax.plot(subset["xRu"], subset["median"],
                                "-" if subset["x_dependent"].iloc[0] else "--", label=name)
                ax.fill_between(subset["xRu"], subset["hdi95_lower"], subset["hdi95_upper"],
                                color=line.get_color(), alpha=0.15, linewidth=0)
                observed = subset.loc[subset["material"].notna() & subset["active"]]
                ax.scatter(observed["xRu"], observed["median"], color=line.get_color(), s=20, zorder=3)
        ax.set_title(title)
        ax.set_xlabel(r"Ru fraction ($x_{\mathrm{Ru}}$)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"PtRu {specification.medium}: {specification.model_name} / {specification.parameterization}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
