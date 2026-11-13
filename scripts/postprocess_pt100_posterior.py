
"""Posterior predictive diagnostics for acidic/basic Pt100 CO oxidation."""

from argparse import ArgumentParser
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkm.inference.posterior import compute_posterior_deterministics, load_inference_data
from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES
from mkm.project_paths import ProjectPaths
from mkm.provenance import read_run_metadata, sha256_file, sha256_mapping
from mkm.workflows.pt100_fit import build_pt100_fit


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(PT100_PARAMETER_CLASSES))
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--draws-per-chain", type=int, default=125)
    parser.add_argument("--random-seed", type=int, default=1)
    return parser.parse_args()


def validate_checkpoint(fit, paths, metadata):
    """Ensure predictions use the same data, configuration, and model code as sampling."""
    if metadata.get("status") != "complete":
        raise ValueError("Posterior run is not complete.")

    specification = metadata["model_specification"]
    if specification["medium"] != fit.medium or specification["model_name"] != fit.model_name:
        raise ValueError("Posterior mechanism or medium does not match the requested fit.")

    if metadata["inputs"]["data_sha256"] != sha256_file(fit.data_path):
        raise ValueError("Selected data have changed since sampling.")

    config_paths = (fit.model_config_path, fit.prior_config_path)
    config_hash = sha256_mapping({
        str(path.relative_to(paths.root)): sha256_file(path) for path in config_paths
    })
    if metadata["inputs"]["model_config_sha256"] != config_hash:
        raise ValueError("Model or prior configuration has changed since sampling.")

    for relative_path, expected_hash in specification["source_sha256"].items():
        path = paths.root / relative_path
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Model source changed since sampling: {relative_path}")


def sample_matrix(posterior, name):
    """Return a (posterior samples, model points) array."""
    return np.asarray(
        posterior[name].transpose("chain", "draw", "model_point"), dtype=float
    ).reshape(-1, posterior.sizes["model_point"])


def add_interval(frame, name, samples):
    """Attach posterior 95% intervals and medians to the model-point table."""
    lower, median, upper = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
    frame[f"{name}_lower"] = lower
    frame[f"{name}_median"] = median
    frame[f"{name}_upper"] = upper


def plot_conditions(points, observations, kind, title, output_path):
    """Plot rates, residuals, or surface coverages on a CO/electrolyte grid."""
    concentrations = sorted(points["C_electrolyte_M"].unique())
    co_values = sorted(points["CO_mole_fraction"].unique())

    fig, axes = plt.subplots(
        len(co_values), len(concentrations),
        figsize=(14, 11), sharex=True, squeeze=False,
    )

    coverage_lines = (
        ("theta_empty", "Free Pt sites", "tab:gray"),
        ("theta_CO", "CO*", "tab:blue"),
        ("theta_OH", "OH*", "tab:orange"),
    )

    for row, co in enumerate(co_values):
        for col, concentration in enumerate(concentrations):
            ax = axes[row, col]
            mask = (
                np.isclose(points["C_electrolyte_M"], concentration)
                & np.isclose(points["CO_mole_fraction"], co)
            )
            panel = points.loc[mask].sort_values("E_V_SHE")
            observed_panel = observations.loc[
                np.isclose(observations["C_electrolyte_M"], concentration)
                & np.isclose(observations["CO_mole_fraction"], co)
            ]

            E = panel["E_V_SHE"].to_numpy()

            if kind == "coverages":
                for name, label, color in coverage_lines:
                    ax.plot(E, panel[f"{name}_median"], color=color, lw=1.5, label=label)
                    ax.fill_between(
                        E, panel[f"{name}_lower"], panel[f"{name}_upper"],
                        color=color, alpha=0.15,
                    )
                ax.set_ylim(-0.02, 1.02)

            elif kind == "residuals":
                ax.axhline(0, color="black", lw=0.8)
                ax.axhline(2, color="tab:red", ls="--", lw=0.8)
                ax.axhline(-2, color="tab:red", ls="--", lw=0.8)
                ax.scatter(
                    observed_panel["E_V_SHE"], observed_panel["standardized_residual"],
                    color="tab:blue", alpha=0.5, s=9,
                )

            else:
                prefix = "rate_model" if kind == "rates" else "rate_predictive"
                ax.fill_between(
                    E, panel[f"{prefix}_lower"], panel[f"{prefix}_upper"],
                    color="tab:red", alpha=0.2, label="95% interval",
                )
                ax.plot(
                    E, panel[f"{prefix}_median"],
                    color="tab:red", lw=1.6, label="Posterior median",
                )
                ax.scatter(
                    observed_panel["E_V_SHE"], observed_panel["rate_observed"],
                    color="0.45", alpha=0.5, s=8, label="Experiment",
                )
                if kind == "rates":
                    ax.set_yscale("log")

            if row == 0:
                ax.set_title(f"{concentration:g} M")
            if col == 0:
                ax.set_ylabel(f"{100 * co:g}% CO")
            if row == 0 and col == 0 and kind != "residuals":
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(title, fontsize=15)
    fig.supxlabel("Potential / V vs SHE")

    labels = {
        "rates": "Model TOF / s$^{-1}$ (log scale)",
        "predictive": "Posterior predictive TOF / s$^{-1}$ (linear scale)",
        "residuals": "Standardized residual",
        "coverages": "Surface coverage",
    }
    fig.supylabel(labels[kind])
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_diagnostics(posterior, summary_path, figures_dir):
    """Plot posterior correlations and reported prior/posterior intervals."""
    variables = [
        name for name in posterior.data_vars
        if name != "ln_rate_model" and not name.startswith(("theta_", "rate_fraction_", "ln_rate_"))
    ]

    draws = pd.DataFrame({
        name: np.asarray(posterior[name]).reshape(-1) for name in variables
    })
    correlations = draws.corr()
    correlations.to_csv(figures_dir.parent / "tables" / "parameter_correlations.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(variables)), variables, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(variables)), variables)

    for row in range(len(variables)):
        for col in range(len(variables)):
            value = correlations.iloc[row, col]
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center",
                color="white" if abs(value) > 0.6 else "black", fontsize=8,
            )

    fig.colorbar(image, ax=ax, label="Posterior Pearson correlation")
    fig.tight_layout()
    fig.savefig(figures_dir / "parameter_correlations.png", dpi=180)
    plt.close(fig)

    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()

    for ax, (_, item) in zip(axes, summary.iterrows()):
        ax.hlines(
            1, item["prior_q025"], item["prior_q975"],
            color="tab:gray", linewidth=3, label="Prior 95%",
        )
        ax.plot(item["prior_q50"], 1, "o", color="tab:gray")

        ax.hlines(
            0, item["hdi95_lower"], item["hdi95_upper"],
            color="tab:blue", linewidth=3, label="Posterior 95% HDI",
        )
        ax.plot(item["median"], 0, "o", color="tab:blue")

        ax.set_title(item["parameter"], fontsize=10)
        ax.set_yticks([0, 1], ["Posterior", "Prior"])
        ax.set_ylim(-0.5, 1.5)
        ax.grid(axis="x", alpha=0.2)

    for ax in axes[len(summary):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(figures_dir / "prior_posterior.png", dpi=180)
    plt.close(fig)



def main():
    args = parse_args()
    if args.draws_per_chain < 1:
        raise ValueError("--draws-per-chain must be positive.")

    paths = ProjectPaths.discover(__file__)
    fit = build_pt100_fit(paths, args.medium, args.model)
    output_dir = fit.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    metadata = read_run_metadata(output_dir / "run_metadata.yaml")
    validate_checkpoint(fit, paths, metadata)

    inference_data = load_inference_data(output_dir / "posterior.nc")

    # ArviZ 1.3 returns an xarray DataTree. PyMC reconstruction needs a Dataset.
    posterior = inference_data.posterior
    if hasattr(posterior, "to_dataset"):
        posterior = posterior.to_dataset()

    free_names = [rv.name for rv in fit.built_model.model.free_RVs]
    missing = [name for name in free_names if name not in posterior]
    if missing:
        raise ValueError(f"Checkpoint is missing sampled parameters: {missing}")

    # Select evenly spaced draws and use consecutive draw coordinates.
    n_draws = posterior.sizes["draw"]
    stride = max(1, int(np.ceil(n_draws / args.draws_per_chain)))
    selected = posterior.isel(draw=slice(None, None, stride)).copy()
    selected = selected.assign_coords(draw=np.arange(selected.sizes["draw"]))

    required = ["ln_rate_model", "theta_empty", "theta_CO", "theta_OH"]
    required += [
        name for name in ("rate_fraction_ER", "rate_fraction_LH")
        if name in fit.built_model.mechanism_result.pointwise
    ]

    n_samples = selected.sizes["chain"] * selected.sizes["draw"]
    print(f"Reconstructing {n_samples} posterior samples.")

    reconstructed = compute_posterior_deterministics(
        SimpleNamespace(posterior=selected),
        fit.built_model,
        var_names=required,
        backend="numba",
        progressbar=True,
    )

    if hasattr(reconstructed, "to_dataset"):
        reconstructed = reconstructed.to_dataset()

    # Check reconstructed deterministics individually before computing diagnostics.
    for name in required:
        if name not in reconstructed:
            raise ValueError(f"Reconstruction did not return {name}.")

        values = np.asarray(reconstructed[name], dtype=float)
        n_invalid = np.count_nonzero(~np.isfinite(values))

        if n_invalid:
            raise ValueError(
                f"Reconstructed {name} contains {n_invalid:,}/{values.size:,} "
                "non-finite values."
            )

    rates = np.exp(sample_matrix(reconstructed, "ln_rate_model"))
    theta_empty = sample_matrix(reconstructed, "theta_empty")
    theta_co = sample_matrix(reconstructed, "theta_CO")
    theta_oh = sample_matrix(reconstructed, "theta_OH")

    site_error = np.max(np.abs(theta_empty + theta_co + theta_oh - 1))
    theta_min = min(np.min(x) for x in (theta_empty, theta_co, theta_oh))
    theta_max = max(np.max(x) for x in (theta_empty, theta_co, theta_oh))

    print(f"Maximum posterior site-balance error: {site_error:.3e}")
    print(f"Posterior coverage range: [{theta_min:.6f}, {theta_max:.6f}]")

    if not np.isfinite(rates).all() or not np.isfinite(site_error):
        raise ValueError("Non-finite posterior rates or coverages.")

    if site_error > 1e-7 or theta_min < -1e-7 or theta_max > 1 + 1e-7:
        raise ValueError("Posterior coverages violate physical site constraints.")

    inputs = fit.inputs
    point_conditions = np.asarray(inputs.model_point_condition_index, dtype=int)
    concentrations = np.exp(inputs.condition_ln_electrolyte_concentration)
    co_fractions = np.exp(inputs.condition_ln_CO_mole_fraction)

    points = pd.DataFrame({
        "model_point_id": np.arange(rates.shape[1]),
        "C_electrolyte_M": concentrations[point_conditions],
        "CO_mole_fraction": co_fractions[point_conditions],
        "E_V_SHE": inputs.model_point_E_V_SHE,
    })

    add_interval(points, "rate_model", rates)
    add_interval(points, "theta_empty", theta_empty)
    add_interval(points, "theta_CO", theta_co)
    add_interval(points, "theta_OH", theta_oh)

    for name in ("rate_fraction_ER", "rate_fraction_LH"):
        if name in reconstructed:
            fraction = sample_matrix(reconstructed, name)
            add_interval(points, name, fraction)
            print(f"{name} median over all points: {np.median(fraction):.6g}")

    # Generate posterior predictive draws using the fitted measurement model.
    sigma_abs = np.asarray(selected["sigma_rate_abs"]).reshape(-1, 1)
    sigma_rel = np.asarray(selected["sigma_rate_rel"]).reshape(-1, 1)
    sigma = sigma_abs + sigma_rel * rates

    rng = np.random.default_rng(args.random_seed)
    predictive_rates = rng.normal(rates, sigma)

    add_interval(points, "rate_predictive", predictive_rates)
    predictive_sd = np.std(predictive_rates, axis=0, ddof=1)
    predictive_negative = np.mean(predictive_rates < 0)

    observation_points = np.asarray(inputs.observation_model_point_index, dtype=int)
    observations = points.iloc[observation_points].reset_index(drop=True).copy()
    observations["rate_observed"] = inputs.observation_rate
    observations["residual"] = (
        observations["rate_observed"] - observations["rate_model_median"]
    )
    observations["standardized_residual"] = (
        observations["residual"] / predictive_sd[observation_points]
    )

    condition_summary = (
        observations.groupby(["C_electrolyte_M", "CO_mole_fraction"])
        .agg(
            n=("standardized_residual", "size"),
            mean_z=("standardized_residual", "mean"),
            rmse_z=("standardized_residual", lambda x: np.sqrt(np.mean(x**2))),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    points.to_csv(tables_dir / "posterior_model_points.csv", index=False)
    observations.to_csv(tables_dir / "posterior_observations.csv", index=False)
    condition_summary.to_csv(tables_dir / "posterior_condition_summary.csv", index=False)

    title = f"Pt100 {args.medium}: {args.model}"

    plot_conditions(
        points, observations, "rates",
        f"{title} - model uncertainty",
        figures_dir / "rates_model.png",
    )
    plot_conditions(
        points, observations, "predictive",
        f"{title} - predictive uncertainty",
        figures_dir / "rates_predictive.png",
    )
    plot_conditions(
        points, observations, "residuals",
        f"{title} - standardized residuals",
        figures_dir / "residuals.png",
    )
    plot_conditions(
        points, observations, "coverages",
        f"{title} - Pt site coverages",
        figures_dir / "coverages.png",
    )

    plot_parameter_diagnostics(
        posterior,
        output_dir / "posterior_parameters.csv",
        figures_dir,
    )

    print(f"Predictive draws below zero: {predictive_negative:.1%}")
    print(
        "Standardized residuals with |z| > 2: "
        f"{np.mean(np.abs(observations['standardized_residual']) > 2):.1%}"
    )
    print("\nConditions with largest standardized residual RMSE:")
    print(condition_summary.nlargest(5, "rmse_z").to_string(index=False))
    print(f"\nSaved figures: {figures_dir}")
    print(f"Saved tables: {tables_dir}")

if __name__ == "__main__":
    main()

