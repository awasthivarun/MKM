import math
from pathlib import Path

import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, truncnorm

from mkm.postprocessing.sampling import build_sampling_datatree

def _prior_pdf(x, spec):
    distribution = spec["distribution"]

    if distribution == "normal":
        return norm.pdf(x, loc=float(spec["mu"]), scale=float(spec["sigma"]))

    if distribution == "uniform":
        lower, upper = float(spec["lower"]), float(spec["upper"])
        return np.where((x >= lower) & (x <= upper), 1.0 / (upper - lower), 0.0)

    if distribution == "truncated_normal":
        mu, sigma = float(spec["mu"]), float(spec["sigma"])
        lower, upper = float(spec.get("lower", -np.inf)), float(spec.get("upper", np.inf))
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        return truncnorm.pdf(x, a=a, b=b, loc=mu, scale=sigma)

    raise ValueError(f"Unsupported prior distribution '{distribution}'.")


def plot_parameter_posteriors(posterior, parameter_specs, output_path: str | Path):
    names = list(parameter_specs)
    if not names:
        raise ValueError("No parameter specifications were provided.")

    missing = [name for name in names if name not in posterior]
    if missing:
        raise ValueError(f"Posterior is missing configured parameters: {missing}")

    ncols = min(3, len(names))
    nrows = math.ceil(len(names) / ncols)

    # Modern ArviZ plotting operates on a DataTree with named groups.
    data = xr.DataTree.from_dict({"/posterior": posterior})

    pc = azp.plot_dist(
        data,
        var_names=names,
        group="posterior",
        kind="kde",
        point_estimate="mean",
        ci_kind="hdi",
        ci_prob=0.95,
        backend="matplotlib",
        col_wrap=ncols,
        figure_kwargs={"figsize": (4.2 * ncols, 3.1 * nrows)},
        visuals={
            "remove_axis": True,
            "face": {"alpha": 0.12},
            "point_estimate_text": False,
        },
    )

    for name in names:
        ax = pc.get_target(name, {})
        values = np.asarray(posterior[name], dtype=float).reshape(-1)

        mean = float(np.mean(values))
        hdi = np.asarray(azs.hdi(values, prob=0.95), dtype=float).reshape(-1)
        hdi_lower, hdi_upper = float(hdi[0]), float(hdi[1])

        # ArviZ chooses the posterior-focused x-range. Show only the local prior shape over that range.
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 500)
        prior = _prior_pdf(x, parameter_specs[name])

        if np.any(np.isfinite(prior)) and np.nanmax(prior) > 0:
            _, y_max = ax.get_ylim()
            prior_scaled = prior / np.nanmax(prior) * 0.30 * y_max
            ax.fill_between(x, 0.0, prior_scaled, alpha=0.06, linewidth=0, zorder=0)
            ax.plot(x, prior_scaled, linestyle="--", linewidth=1.0, alpha=0.25, zorder=0.5)

        _, y_max = ax.get_ylim()

        # Keep the statistics explicit rather than relying on backend-specific annotation behavior.
        ax.text(mean, 0.96 * y_max, f"mean = {mean:.4g}", ha="center", va="top", fontsize=8)
        ax.text(hdi_lower, 0.06 * y_max, f"{hdi_lower:.4g}", ha="center", va="bottom", fontsize=8)
        ax.text(hdi_upper, 0.06 * y_max, f"{hdi_upper:.4g}", ha="center", va="bottom", fontsize=8)
        ax.text(
            0.5 * (hdi_lower + hdi_upper),
            0.14 * y_max,
            "95% HDI",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        ax.set_title(name)
        ax.grid(axis="x", alpha=0.15)

    pc.add_title("Posterior parameter distributions")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")

    fig = pc.get_target(names[0], {}).figure
    plt.close(fig)


def _posterior_interval_columns(frame, prefix=None):
    base = "" if prefix is None else f"{prefix}_"

    new = (
        f"{base}median",
        f"{base}hdi95_lower",
        f"{base}hdi95_upper",
    )
    if all(column in frame.columns for column in new):
        return new

    legacy = (
        f"{base}q50",
        f"{base}q025",
        f"{base}q975",
    )
    if all(column in frame.columns for column in legacy):
        return legacy

    raise ValueError(f"Could not find posterior median/95% interval columns for prefix '{prefix}'.")


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
    if distribution not in {"mechanism", "conditional", "predictive"}:
        raise ValueError("distribution must be 'mechanism', 'conditional', or 'predictive'.")

    KOH_values = sorted(observations["electrolyte_concentration_M"].unique())
    CO_values = sorted(observations["CO_mole_fraction"].unique())

    fig, axes = plt.subplots(
        len(CO_values),
        len(KOH_values),
        figsize=(3.4 * len(KOH_values), 2.5 * len(CO_values)),
        sharex=True,
        squeeze=False,
    )

    observed_column = "rate"

    if not residual:
        prefix = f"rate_{distribution}"
        median_column, lower_column, upper_column = _posterior_interval_columns(observations, prefix)

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
                        curve["residual_conditional"],
                        linewidth=1.2,
                        alpha=0.65,
                        label=replicate,
                    )
            else:
                for replicate, curve in condition.groupby("replicate", sort=True):
                    curve = curve.sort_values("E_V_SHE")
                    ax.plot(
                        curve["E_V_SHE"],
                        curve[observed_column],
                        linewidth=1.0,
                        alpha=0.45,
                        label=f"{replicate} observed",
                    )

                if distribution == "mechanism":
                    posterior_curve = (
                        condition.sort_values("E_V_SHE")
                        .drop_duplicates("model_point_id")
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
                        label="mechanism posterior",
                    )
                else:
                    for replicate, curve in condition.groupby("replicate", sort=True):
                        curve = curve.sort_values("E_V_SHE")
                        line, = ax.plot(
                            curve["E_V_SHE"],
                            curve[median_column],
                            linewidth=1.5,
                            label=f"{replicate} {distribution}",
                        )
                        ax.fill_between(
                            curve["E_V_SHE"],
                            curve[lower_column],
                            curve[upper_column],
                            color=line.get_color(),
                            alpha=0.10,
                            linewidth=0,
                        )

            if residual:
                ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)

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
        fig.suptitle(f"{prefix}Conditional log-rate residuals")
        fig.supylabel("ln(rate) observed - posterior conditional median")
    else:
        label = distribution.replace("_", " ")
        axis_label = "log y-axis" if y_scale == "log" else "linear y-axis"
        fig.suptitle(f"{prefix}Posterior {label} rate: median and 95% HDI ({axis_label})")
        fig.supylabel("rate / s$^{-1}$")

    fig.supxlabel("Potential (V vs SHE)")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
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

def plot_sampling_trace(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    nrows = math.ceil(len(parameter_names) / 3)

    pc = azp.plot_trace(
        data,
        var_names=parameter_names,
        group="posterior",
        backend="matplotlib",
        visuals={"divergence": True},
        col_wrap=3,
        figure_kwargs={"figsize": (14, 3.8 * nrows), "layout": "none"},
    )

    fig = pc.get_target(parameter_names[0], {}).figure
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.92, hspace=0.55, wspace=0.20)
    fig.suptitle("MCMC sampling traces")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_rank(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    nrows = math.ceil(len(parameter_names) / 3)

    pc = azp.plot_rank(
        data,
        var_names=parameter_names,
        group="posterior",
        backend="matplotlib",
        col_wrap=3,
        figure_kwargs={"figsize": (14, 3.8 * nrows), "layout": "none"},
    )

    fig = pc.get_target(parameter_names[0], {}).figure
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.92, hspace=0.55, wspace=0.20)
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
            var_names=parameter_names,
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

def plot_alpha_comparison(comparison, output_dir: str | Path, material):
    for c_koh, koh_data in comparison.groupby("C_KOH_M", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        axes = axes.ravel()

        for ax, (co_fraction, data) in zip(axes, koh_data.groupby("CO_mole_fraction", sort=True)):
            data = data.sort_values("E_V_SHE")
            ax.fill_between(data["E_V_SHE"], data["hdi95_lower"], data["hdi95_upper"], alpha=0.20, linewidth=0)
            ax.plot(data["E_V_SHE"], data["median"], linewidth=1.5, label="posterior")
            ax.errorbar(
                data["E_V_SHE"], data["alpha_mean"], yerr=data["alpha_sd"], fmt="o",
                markersize=3, linewidth=0.8, label="experiment",
            )
            ax.set_title(f"CO = {100 * co_fraction:g}%")
            ax.set_ylabel(r"$\alpha$")
            ax.grid(alpha=0.20)

        axes[-2].set_xlabel("Potential (V vs SHE)")
        axes[-1].set_xlabel("Potential (V vs SHE)")
        axes[0].legend()
        fig.suptitle(f"{material}, {c_koh:g} M KOH")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"alpha_KOH_{c_koh:g}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_delta_oh_comparison(comparison, output_path: str | Path, material):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()

    for ax, (co_fraction, data) in zip(axes, comparison.groupby("CO_mole_fraction", sort=True)):
        data = data.sort_values("E_V_SHE")
        ax.fill_between(data["E_V_SHE"], data["hdi95_lower"], data["hdi95_upper"], alpha=0.20, linewidth=0)
        ax.plot(data["E_V_SHE"], data["median"], linewidth=1.5, label="posterior")
        ax.errorbar(
            data["E_V_SHE"], data["delta_OH"], yerr=data["delta_OH_sd"], fmt="o",
            markersize=3, linewidth=0.8, label="experiment",
        )
        ax.set_title(f"CO = {100 * co_fraction:g}%")
        ax.set_ylabel(r"$\delta_{\mathrm{OH}}$")
        ax.grid(alpha=0.20)

    axes[-2].set_xlabel("Potential (V vs SHE)")
    axes[-1].set_xlabel("Potential (V vs SHE)")
    axes[0].legend()
    fig.suptitle(material)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_co_comparison(comparison, output_dir: str | Path, material):
    for c_koh, koh_data in comparison.groupby("C_KOH_M", sort=True):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

        grouped = koh_data.groupby(["CO_lower_mole_fraction", "CO_upper_mole_fraction"], sort=True)
        for ax, ((lower_co, upper_co), data) in zip(axes, grouped):
            data = data.sort_values("E_V_SHE")
            ax.fill_between(data["E_V_SHE"], data["hdi95_lower"], data["hdi95_upper"], alpha=0.20, linewidth=0)
            ax.plot(data["E_V_SHE"], data["median"], linewidth=1.5, label="posterior")
            ax.errorbar(
                data["E_V_SHE"], data["delta_CO"], yerr=data["delta_CO_sd"], fmt="o",
                markersize=3, linewidth=0.8, label="experiment",
            )
            ax.set_title(f"{100 * lower_co:g}% -> {100 * upper_co:g}% CO")
            ax.set_xlabel("Potential (V vs SHE)")
            ax.set_ylabel(r"$\delta_{\mathrm{CO}}$")
            ax.grid(alpha=0.20)

        axes[0].legend()
        fig.suptitle(f"{material}, {c_koh:g} M KOH")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"delta_CO_KOH_{c_koh:g}.png", dpi=220, bbox_inches="tight")
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


def plot_pointwise_loo(pointwise, model_name, output_path: str | Path):
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
    fig.suptitle(f"Pointwise PSIS-LOO: {model_name}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loo_pit_ecdf(loo_pit, model_name, output_path: str | Path):
    from mkm.postprocessing.calibration import build_loo_pit_datatree

    data = build_loo_pit_datatree(loo_pit)

    pc = azp.plot_ecdf_pit(
        data,
        var_names=["ln_rate_observed"],
        group="loo_pit",
        sample_dims=["observation"],
        method="pot_c",
        envelope_prob=0.95,
        coverage=False,
        backend="matplotlib",
    )
    pc.add_title(f"LOO-PIT calibration: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_loo_pit_coverage(loo_pit, model_name, output_path: str | Path):
    from mkm.postprocessing.calibration import build_loo_pit_datatree

    data = build_loo_pit_datatree(loo_pit)

    pc = azp.plot_ecdf_pit(
        data,
        var_names=["ln_rate_observed"],
        group="loo_pit",
        sample_dims=["observation"],
        method="pot_c",
        envelope_prob=0.95,
        coverage=True,
        backend="matplotlib",
    )
    pc.add_title(f"LOO predictive coverage: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_loo_pit_conditions(pointwise, model_name, output_path: str | Path):
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
                    linewidth=1.0,
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
    fig.suptitle(f"Condition-resolved LOO-PIT: {model_name}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
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

                line, = ax.plot(condition["E_V_SHE"], condition["median"], linewidth=1.5, label=label)
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
