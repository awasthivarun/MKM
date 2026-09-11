import math
from pathlib import Path

import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, lognorm, norm, truncnorm

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
    "theta_empty_Pd": "#7F7F7F",
    "theta_OH_Ag": "#009E73",
    "theta_empty_Ag": "#7F7F7F",
}

POINTWISE_LABELS = {
    "theta_CO": r"$\theta_{\mathrm{CO}}$",
    "theta_OH_Pd": r"$\theta_{\mathrm{OH,Pd}}$",
    "theta_empty_Pd": r"$\theta_{*,\mathrm{Pd}}$",
    "theta_OH_Ag": r"$\theta_{\mathrm{OH,Ag}}$",
    "theta_empty_Ag": r"$\theta_{*,\mathrm{Ag}}$",
    "rate_fraction_BF": "BF",
    "rate_fraction_ER": "ER",
    "rate_fraction_LH": "LH",
}



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
        return lognorm.pdf(
            x,
            s=float(spec["log_sd"]),
            scale=float(spec["median"]),
        )

    raise ValueError(f"Unsupported prior distribution '{distribution}'.")


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
    """Plot physical and error parameters together, using coordinate labels for vectors."""
    components = _posterior_components(posterior, parameter_specs)
    if not components:
        raise ValueError("No parameter specifications were provided.")

    ncols = min(3, len(components))
    nrows = math.ceil(len(components) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.1 * nrows),
        squeeze=False,
    )

    for ax, (label, values, spec) in zip(axes.flat, components):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Posterior parameter '{label}' contains non-finite values.")

        lower, upper = np.quantile(values, [0.001, 0.999])
        if np.isclose(lower, upper):
            width = max(abs(float(lower)) * 0.05, 1e-9)
            lower -= width
            upper += width
        x = np.linspace(lower, upper, 500)

        if len(np.unique(values)) > 1:
            posterior_density = gaussian_kde(values)(x)
            ax.fill_between(x, 0.0, posterior_density, alpha=0.20)
            ax.plot(x, posterior_density, linewidth=1.6, label="posterior")
            density_max = float(np.max(posterior_density))
        else:
            ax.axvline(values[0], linewidth=1.6, label="posterior")
            density_max = 1.0

        prior = _prior_pdf(x, spec)
        if np.any(np.isfinite(prior)) and np.nanmax(prior) > 0:
            prior_scaled = prior / np.nanmax(prior) * 0.35 * density_max
            ax.plot(x, prior_scaled, linestyle="--", linewidth=1.1, alpha=0.6, label="prior")

        mean = float(np.mean(values))
        hdi = np.asarray(azs.hdi(values, prob=0.95), dtype=float).reshape(-1)
        ax.axvline(mean, linewidth=1.0, alpha=0.7)
        ax.axvspan(float(hdi[0]), float(hdi[1]), alpha=0.08)
        ax.set_title(label)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.15)

    for ax in axes.flat[len(components):]:
        ax.set_visible(False)

    fig.suptitle("Posterior parameter distributions")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
        len(CO_values),
        len(KOH_values),
        figsize=(3.4 * len(KOH_values), 2.5 * len(CO_values)),
        sharex=True,
        squeeze=False,
    )

    if not residual:
        prefix = f"rate_{distribution}"
        median_column, lower_column, upper_column = _posterior_interval_columns(
            observations,
            prefix,
        )

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
                        curve["E_V_SHE"],
                        curve["residual"],
                        linewidth=1.2,
                        alpha=0.65,
                        label=replicate,
                    )
                ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
            else:
                for replicate, curve in condition.groupby("replicate", sort=True):
                    curve = curve.sort_values("E_V_SHE")
                    ax.plot(
                        curve["E_V_SHE"],
                        curve["rate"],
                        linewidth=1.0,
                        alpha=0.45,
                        label=f"{replicate} observed",
                    )

                if distribution == "model":
                    posterior_curve = condition.sort_values("E_V_SHE").drop_duplicates(
                        "model_point_id"
                    )
                    ax.fill_between(
                        posterior_curve["E_V_SHE"],
                        posterior_curve[lower_column],
                        posterior_curve[upper_column],
                        alpha=0.16,
                        linewidth=0,
                    )
                    ax.plot(
                        posterior_curve["E_V_SHE"],
                        posterior_curve[median_column],
                        linewidth=1.8,
                        label="model posterior",
                    )
                else:
                    posterior_curve = condition.sort_values("E_V_SHE").drop_duplicates(
                        "model_point_id"
                    )
                    line, = ax.plot(
                        posterior_curve["E_V_SHE"],
                        posterior_curve[median_column],
                        linewidth=1.8,
                        label="posterior predictive",
                    )
                    ax.fill_between(
                        posterior_curve["E_V_SHE"],
                        posterior_curve[lower_column],
                        posterior_curve[upper_column],
                        color=line.get_color(),
                        alpha=0.16,
                        linewidth=0,
                    )
                    ax.axhline(0.0, linestyle=":", linewidth=0.8, alpha=0.35)

            if not residual and y_scale == "log":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )
            ax.grid(alpha=0.20)

    prefix = f"{context_label}: " if context_label else ""
    if residual:
        fig.suptitle(f"{prefix}Rate residuals")
        fig.supylabel(r"observed rate - posterior model median / s$^{-1}$")
    else:
        axis_label = "log y-axis" if y_scale == "log" else "linear y-axis"
        fig.suptitle(
            f"{prefix}Posterior {distribution} rate: median and 95% HDI ({axis_label})"
        )
        fig.supylabel(r"rate / s$^{-1}$")

    fig.supxlabel("Potential (V vs SHE)")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
def plot_pointwise_variable(summary, variable_name, output_path: str | Path, context_label=None):
    KOH_values = sorted(summary["electrolyte_concentration_M"].unique())
    CO_values = sorted(summary["CO_mole_fraction"].unique())

    fig, axes = plt.subplots(
        len(CO_values),
        len(KOH_values),
        figsize=(3.4 * len(KOH_values), 2.5 * len(CO_values)),
        sharex=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]

            condition = summary[
                (summary["electrolyte_concentration_M"] == c_koh)
                & (summary["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")

            median_column, lower_column, upper_column = _posterior_interval_columns(condition)

            ax.fill_between(
                condition["E_V_SHE"],
                condition[lower_column],
                condition[upper_column],
                alpha=0.20,
                linewidth=0,
            )

            ax.plot(condition["E_V_SHE"], condition[median_column], linewidth=2.0)

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")

            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )

            ax.grid(alpha=0.20)

    title = f"{context_label}: {variable_name}" if context_label else variable_name
    fig.suptitle(title)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(variable_name)
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
    """Plot several pointwise posterior variables together on the same condition grid."""
    variable_names = [
        variable
        for variable in variable_names
        if f"{variable}_median" in summary.columns
    ]
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
        len(CO_values),
        len(KOH_values),
        figsize=(3.4 * len(KOH_values), 2.5 * len(CO_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = summary[
                (summary["electrolyte_concentration_M"] == c_koh)
                & (summary["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")

            for variable in variable_names:
                median_column = f"{variable}_median"
                lower_column = f"{variable}_hdi95_lower"
                upper_column = f"{variable}_hdi95_upper"
                if not all(
                    column in condition.columns
                    for column in (median_column, lower_column, upper_column)
                ):
                    continue

                color = colors.get(variable)
                label = labels.get(variable, variable)
                line, = ax.plot(
                    condition["E_V_SHE"],
                    condition[median_column],
                    linewidth=1.8,
                    color=color,
                    label=label,
                )
                ax.fill_between(
                    condition["E_V_SHE"],
                    condition[lower_column],
                    condition[upper_column],
                    color=line.get_color(),
                    alpha=0.14,
                    linewidth=0,
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
                            mean_rate["E_V_SHE"],
                            rate_values / max_rate,
                            color="#7F7F7F",
                            linestyle=":",
                            linewidth=1.5,
                            label="_nolegend_",
                        )

            if bounded:
                ax.set_ylim(-0.02, 1.02)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )
            ax.grid(alpha=0.20)

    axes[0, 0].legend(fontsize=8)
    prefix = f"{context_label}: " if context_label else ""
    fig.suptitle(f"{prefix}{title}")
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(ylabel)
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True

def plot_sampling_trace(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    nrows = math.ceil(len(plotted_names) / 3)

    pc = azp.plot_trace(
        data,
        var_names=plotted_names,
        group="posterior",
        backend="matplotlib",
        visuals={"divergence": True},
        col_wrap=3,
        figure_kwargs={"figsize": (14, 3.8 * nrows), "layout": "none"},
    )

    fig = pc.get_target(plotted_names[0], {}).figure
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.06,
        top=0.92,
        hspace=0.55,
        wspace=0.20,
    )
    fig.suptitle("MCMC sampling traces")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_rank(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    nrows = math.ceil(len(plotted_names) / 3)

    pc = azp.plot_rank(
        data,
        var_names=plotted_names,
        group="posterior",
        backend="matplotlib",
        col_wrap=3,
        figure_kwargs={"figsize": (14, 3.8 * nrows), "layout": "none"},
    )

    fig = pc.get_target(plotted_names[0], {}).figure
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.06,
        top=0.92,
        hspace=0.55,
        wspace=0.20,
    )
    fig.suptitle("Chain rank diagnostics")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_energy(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)

    pc = azp.plot_energy(data, backend="matplotlib", show_bfmi=True, threshold=0.3)
    pc.add_title("Hamiltonian energy and BFMI")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_sampling_pairs(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    n_parameters = len(parameter_names)
    size = max(10.0, 2.1 * n_parameters)

    # PlotMatrix allocates the full N x N matrix even when only the lower triangle is rendered.
    with azb.rc_context({"plot.max_subplots": max(40, n_parameters**2)}):
        pm = azp.plot_pair(
            data,
            var_names=list(parameter_names),
            group="posterior",
            marginal=True,
            marginal_kind="kde",
            triangle="lower",
            levels=[0.5, 0.9],
            backend="matplotlib",
            visuals={
                "scatter": {"alpha": 0.12, "s": 5},
                "contour": True,
                "divergence": {"alpha": 0.9, "s": 18},
            },
            figure_kwargs={"figsize": (size, size), "layout": "none"},
        )

    pm.add_title("Posterior parameter pair structure")
    pm.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")

def plot_alpha_comparison(comparison, output_path: str | Path, material):
    """Plot alpha for all KOH/CO conditions in one 4 x 3-style grid."""
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values),
        len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            data = comparison[
                (comparison["C_KOH_M"] == c_koh)
                & (comparison["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")

            if not data.empty:
                ax.fill_between(
                    data["E_V_SHE"],
                    data["hdi95_lower"],
                    data["hdi95_upper"],
                    alpha=0.20,
                    linewidth=0,
                )
                ax.plot(data["E_V_SHE"], data["median"], linewidth=1.5, label="posterior")
                ax.errorbar(
                    data["E_V_SHE"],
                    data["alpha_mean"],
                    yerr=data["alpha_sd"],
                    fmt="o",
                    markersize=3,
                    linewidth=0.8,
                    label="experiment",
                )

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(koh_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )
            ax.grid(alpha=0.20)

    axes[0, 0].legend()
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$\alpha$")
    fig.suptitle(f"{material}: transfer coefficient")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_delta_oh_comparison(comparison, output_path: str | Path, material):
    """Plot OH reaction order as one vertical row per CO fraction."""
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values),
        1,
        figsize=(8.0, 2.5 * len(co_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        ax = axes[row, 0]
        data = comparison.loc[
            comparison["CO_mole_fraction"] == co_fraction
        ].sort_values("E_V_SHE")
        ax.fill_between(
            data["E_V_SHE"], data["hdi95_lower"], data["hdi95_upper"], alpha=0.20, linewidth=0
        )
        ax.plot(data["E_V_SHE"], data["median"], linewidth=1.5, label="posterior")
        ax.errorbar(
            data["E_V_SHE"], data["delta_OH"], yerr=data["delta_OH_sd"], fmt="o",
            markersize=3, linewidth=0.8, label="experiment",
        )
        ax.text(
            1.01,
            0.5,
            f"{100 * co_fraction:g}% CO",
            transform=ax.transAxes,
            va="center",
        )
        ax.grid(alpha=0.20)

    axes[0, 0].legend()
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$\delta_{\mathrm{OH}}$")
    fig.suptitle(f"{material}: OH reaction order")
    fig.tight_layout(rect=(0.06, 0.04, 0.94, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
        len(interval_values),
        len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(interval_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, (lower_co, upper_co) in enumerate(interval_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            data = comparison[
                (comparison["C_KOH_M"] == c_koh)
                & (comparison["CO_lower_mole_fraction"] == lower_co)
                & (comparison["CO_upper_mole_fraction"] == upper_co)
            ].sort_values("E_V_SHE")

            if not data.empty:
                ax.fill_between(
                    data["E_V_SHE"], data["hdi95_lower"], data["hdi95_upper"], alpha=0.20, linewidth=0
                )
                ax.plot(data["E_V_SHE"], data["median"], linewidth=1.5, label="posterior")
                ax.errorbar(
                    data["E_V_SHE"], data["delta_CO"], yerr=data["delta_CO_sd"], fmt="o",
                    markersize=3, linewidth=0.8, label="experiment",
                )

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(koh_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * lower_co:g}% -> {100 * upper_co:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )
            ax.grid(alpha=0.20)

    axes[0, 0].legend()
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$\delta_{\mathrm{CO}}$")
    fig.suptitle(f"{material}: adjacent CO reaction order")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_second_order_difference(comparison, output_path: str | Path, material):
    """Plot d(alpha)/dE - d(delta_OH)/dE using posterior draws and experimental point estimates."""
    data = comparison.loc[comparison["observable"] == "delta2"].copy()
    if data.empty:
        raise ValueError("No delta2 second-order points were provided.")

    koh_values = sorted(data["C_KOH_M"].dropna().unique())
    co_values = sorted(data["CO_mole_fraction"].dropna().unique())
    fig, axes = plt.subplots(
        len(co_values),
        len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = data[
                (data["C_KOH_M"] == c_koh)
                & (data["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")

            if not condition.empty:
                line, = ax.plot(
                    condition["E_V_SHE"],
                    condition["median"],
                    linewidth=1.5,
                    label="posterior",
                )
                ax.fill_between(
                    condition["E_V_SHE"],
                    condition["hdi95_lower"],
                    condition["hdi95_upper"],
                    color=line.get_color(),
                    alpha=0.20,
                    linewidth=0,
                )
                ax.scatter(
                    condition["E_V_SHE"],
                    condition["experimental"],
                    s=12,
                    label="experiment",
                )

            ax.axhline(0.0, linestyle="--", linewidth=0.9, alpha=0.5)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(koh_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )
            ax.grid(alpha=0.20)

    axes[0, 0].legend()
    step = float(data["potential_step_V"].iloc[0])
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$d\alpha/dE-d\delta_{\mathrm{OH}}/dE$ / V$^{-1}$")
    fig.suptitle(f"{material}: second-order kinetic difference (symmetric h = {step:g} V)")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
        len(co_values),
        len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
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
                    curve["E_V_SHE"],
                    curve["elpd_difference"],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.0,
                    alpha=0.75,
                    label=replicate,
                )

            ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.55)
            ax.set_ylim(-y_limit, y_limit)
            ax.grid(alpha=0.20)

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")

            if col == len(koh_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )

    axes[0, 0].legend(title="replicate", fontsize=8)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(f"Pointwise ELPD: {numerator_model} - {denominator_model}")
    fig.suptitle(f"Pointwise PSIS-LOO difference: {numerator_model} vs {denominator_model}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pointwise_loo(pointwise, model_name, output_path: str | Path, context_label=None):
    """Plot observation-level PSIS-LOO contributions for one material."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())

    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
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
                    linewidth=1.0, alpha=0.75, label=replicate,
                )

            ax.grid(alpha=0.20)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(koh_values) - 1:
                ax.text(
                    1.04, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center",
                )

    axes[0, 0].legend(title="replicate", fontsize=8)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel("Pointwise PSIS-LOO ELPD")
    prefix = f"{context_label}: " if context_label else ""
    fig.suptitle(f"{prefix}Pointwise PSIS-LOO: {model_name}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
    ecdf = distribution.azstats.ecdf(
        dim=["observation"],
        pit=True,
        npoints=len(values),
    )
    uniformity_result = distribution.azstats.uniformity_test(
        dim=["observation"],
        method="pot_c",
    )
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

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.step(x, y, where="pre", linewidth=1.5)
    if np.any(suspicious):
        ax.scatter(x[suspicious], y[suspicious], s=18, marker="x")
    ax.text(0.01, 0.92, f"p={p_value:.2f} (alpha={alpha:.2f})", transform=ax.transAxes, va="top")
    ax.set_ylim(-epsilon, epsilon)
    ax.set_ylabel(r"$\Delta$ ECDF")
    ax.set_title(title)
    ax.grid(alpha=0.20)

    if coverage:
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], labels=["0", "25", "50", "75", "100"])
        ax.set_xlabel("ETI %")
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT")


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
    ax.plot(indices[finite], values[finite], linestyle="none", marker=".", markersize=3.0)
    ax.axhline(good_k, linestyle="--", linewidth=1.1, label=f"good k = {good_k:g}")
    for level in (0.5, 0.7, 1.0):
        if not np.isclose(level, good_k):
            ax.axhline(level, linestyle=":", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Pareto k")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.20)


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
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0))
    prefix = f"{context_label}: " if context_label else ""
    _plot_loo_pit_axis(axes[0], loo_pit, coverage=True, title="LOO predictive coverage")
    _plot_loo_pit_axis(axes[1], loo_pit, coverage=False, title="LOO-PIT calibration")
    fig.suptitle(f"{prefix}LOO-PIT diagnostics: {model_name}")
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loo_diagnostics(loo_result, loo_pit, model_name, output_path: str | Path, context_label=None):
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.5))
    _plot_pareto_k_axis(axes[0], loo_result, title="Pareto-k diagnostics")

    if loo_pit is None:
        for ax, title in zip(axes[1:], ("LOO predictive coverage", "LOO-PIT calibration")):
            ax.set_title(title)
            ax.text(0.5, 0.5, "LOO-PIT unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        _plot_loo_pit_axis(axes[1], loo_pit, coverage=True, title="LOO predictive coverage")
        _plot_loo_pit_axis(axes[2], loo_pit, coverage=False, title="LOO-PIT calibration")

    prefix = f"{context_label}: " if context_label else ""
    fig.suptitle(f"{prefix}LOO diagnostics: {model_name}")
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_loo_pit_conditions(pointwise, model_name, output_path: str | Path, context_label=None):
    """Plot potential-resolved raw LOO-PIT values for one material."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())

    fig, axes = plt.subplots(
        len(co_values),
        len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
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
                    curve["E_V_SHE"],
                    curve["loo_pit"],
                    marker="o",
                    markersize=2.5,
                    linewidth=0.8,
                    alpha=0.75,
                    label=replicate,
                )

            ax.axhline(0.50, linestyle="--", linewidth=1.0, alpha=0.60)
            ax.axhline(0.05, linestyle=":", linewidth=0.8, alpha=0.40)
            ax.axhline(0.95, linestyle=":", linewidth=0.8, alpha=0.40)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.20)

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(koh_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )

    axes[0, 0].legend(title="replicate", fontsize=8)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel("LOO-PIT")
    prefix = f"{context_label}: " if context_label else ""
    fig.suptitle(f"{prefix}Condition-resolved LOO-PIT: {model_name}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_transition_state_drc(summary, model_name, output_path: str | Path):
    koh_values = sorted(summary["electrolyte_concentration_M"].unique())
    co_values = sorted(summary["CO_mole_fraction"].unique())
    controls = summary[["control", "label"]].drop_duplicates().itertuples(index=False)

    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
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

                color = PATHWAY_COLORS.get(control)
                line, = ax.plot(
                    condition["E_V_SHE"],
                    condition["median"],
                    linewidth=1.5,
                    color=color,
                    label=label,
                )
                ax.fill_between(
                    condition["E_V_SHE"], condition["hdi95_lower"], condition["hdi95_upper"],
                    color=line.get_color(), alpha=0.15, linewidth=0,
                )

    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.45)
            ax.axhline(1.0, linestyle=":", linewidth=0.8, alpha=0.35)
            ax.grid(alpha=0.20)

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")

            if col == len(koh_values) - 1:
                ax.text(
                    1.04, 0.5, f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes, rotation=-90, va="center",
                )

    axes[0, 0].legend(title="transition state", fontsize=8)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$X_{\mathrm{TS}}$")
    fig.suptitle(f"Transition-state degree of rate control: {model_name}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
