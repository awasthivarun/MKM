import math
from pathlib import Path

import arviz_plots as azp
import arviz_stats as azs
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, truncnorm


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


def plot_observation_grid(observations, output_path: str | Path, residual=False):
    KOH_values = sorted(observations["electrolyte_concentration_M"].unique())
    CO_values = sorted(observations["CO_mole_fraction"].unique())

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

            condition = observations[
                (observations["electrolyte_concentration_M"] == c_koh)
                & (observations["CO_mole_fraction"] == co_fraction)
            ]

            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")

                if residual:
                    ax.plot(
                        curve["E_V_SHE"],
                        curve["residual_conditional"],
                        linewidth=1.2,
                        alpha=0.65,
                        label=replicate,
                    )
                else:
                    observed_line, = ax.plot(
                        curve["E_V_SHE"],
                        curve["ln_rate"],
                        linewidth=1.0,
                        alpha=0.45,
                        label=f"{replicate} observed",
                    )

                    ax.fill_between(
                        curve["E_V_SHE"],
                        curve["ln_rate_predictive_q025"],
                        curve["ln_rate_predictive_q975"],
                        color=observed_line.get_color(),
                        alpha=0.08,
                        linewidth=0,
                    )

                    ax.plot(
                        curve["E_V_SHE"],
                        curve["ln_rate_predictive_q50"],
                        color=observed_line.get_color(),
                        linewidth=1.5,
                    )

            if residual:
                ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)

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

    if residual:
        fig.suptitle("Conditional log-rate residuals")
        fig.supylabel("ln(rate) observed - posterior conditional median")
    else:
        fig.suptitle("Posterior predictive log rates")
        fig.supylabel("ln(rate / s$^{-1}$)")

    fig.supxlabel("Potential (V vs SHE)")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pointwise_variable(summary, variable_name, output_path: str | Path):
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

            ax.fill_between(
                condition["E_V_SHE"],
                condition["q025"],
                condition["q975"],
                alpha=0.20,
                linewidth=0,
            )

            ax.plot(condition["E_V_SHE"], condition["q50"], linewidth=2.0)

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

    fig.suptitle(variable_name)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(variable_name)
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)