
"""Quick Pt100 mechanism screening with the existing linear-rate Normal error structure."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from mkm.mechanisms.pt100 import evaluate_pt100_co
from mkm.models.ptru_registry import available_ptru_models, get_ptru_model_definition
from mkm.project_paths import ProjectPaths
from mkm.workflows.pt100_fit import load_pt100_config
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays, build_model_point_inputs


# Temporary fixed noise scales for screening, not estimated posterior parameters.
SIGMA_ABS = 2.0e-4
SIGMA_REL = 0.18

INITIAL = {
    "deltaG1_0": -0.25,
    "deltaG4_0": {"acidic": 0.78, "basic": -0.04},
    "Gact1_0": 0.65,
    "Gact2_ER_0": {"acidic": 1.15, "basic": 0.83},
    "Gact2_LH_0": 0.80,
    "beta_2_ER": 0.50,
}

BOUNDS = {
    "deltaG1_0": (-0.90, 0.20),
    "deltaG4_0": {"acidic": (0.20, 1.30), "basic": (-0.60, 0.50)},
    "Gact1_0": (0.30, 1.50),
    "Gact2_ER_0": {"acidic": (0.60, 1.80), "basic": (0.35, 1.50)},
    "Gact2_LH_0": (0.40, 1.60),
    "beta_2_ER": (0.05, 0.95),
}


def medium_value(mapping, name, medium):
    value = mapping[name]
    return value[medium] if isinstance(value, dict) else value



def fit_model(inputs, medium, model_name, temperature_K):
    """Fit Pt100 kinetics and diagnose rates, reaction orders, coverages, and pathways."""
    parameter_class = get_ptru_model_definition(model_name).parameter_class
    names = [field.name for field in fields(parameter_class)]
    point_inputs = build_model_point_inputs(inputs)

    p = pt.vector("parameters")
    parameters = parameter_class(**{name: p[i] for i, name in enumerate(names)})
    evaluated = evaluate_pt100_co(point_inputs, parameters, temperature_K, medium, model_name)

    predicted_points = pt.exp(evaluated.ln_rate)
    point_index = np.asarray(inputs.observation_model_point_index, dtype=int)
    predicted = predicted_points[point_index]
    observed = np.asarray(inputs.observation_rate, dtype=float)

    sigma = SIGMA_ABS + SIGMA_REL * predicted
    nll = pt.mean(0.5 * ((observed - predicted) / sigma) ** 2 + pt.log(sigma))
    gradient = pt.grad(nll, p)

    coverage_names = ("theta_empty", "theta_CO", "theta_OH")
    coverages = pt.stack([evaluated.pointwise[name] for name in coverage_names], axis=1)

    zero_fraction = pt.zeros_like(evaluated.ln_rate)
    fraction_er = evaluated.pointwise.get("rate_fraction_ER", zero_fraction)
    fraction_lh = evaluated.pointwise.get("rate_fraction_LH", zero_fraction)

    evaluate = pytensor.function(
        [p], [nll, gradient, predicted, coverages, fraction_er, fraction_lh]
    )

    initial = np.array([medium_value(INITIAL, name, medium) for name in names])
    bounds = [medium_value(BOUNDS, name, medium) for name in names]

    def objective(values):
        value, grad, *_ = evaluate(values)
        return float(value), np.asarray(grad, dtype=float)

    # Reproducible multistart optimization.
    rng = np.random.default_rng(42)
    starts = [initial]
    starts.extend([
        np.array([rng.uniform(lower, upper) for lower, upper in bounds])
        for _ in range(11)
    ])

    results = [
        minimize(
            objective, start, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-10},
        )
        for start in starts
    ]

    valid = [result for result in results if result.success and np.isfinite(result.fun)]
    if not valid:
        raise RuntimeError(f"No successful optimization: {medium}, {model_name}")

    result = min(valid, key=lambda item: item.fun)
    _, _, fitted, theta, er_fraction, lh_fraction = evaluate(result.x)

    fitted = np.asarray(fitted, dtype=float)
    theta = np.asarray(theta, dtype=float)
    er_fraction = np.asarray(er_fraction, dtype=float)
    lh_fraction = np.asarray(lh_fraction, dtype=float)

    rmse = np.sqrt(np.mean((observed - fitted) ** 2))
    site_balance_error = np.max(np.abs(theta.sum(axis=1) - 1.0))

    print(f"\n{medium.upper()} / {model_name}")
    print(f"  Successful optimizations: {len(valid)}/{len(starts)}")
    print(f"  Converged: {result.success} ({result.message})")
    print(f"  Observations: {len(observed):,}")
    print(f"  Mean negative log likelihood: {result.fun:.6f}")
    print(f"  Rate RMSE: {rmse:.6g} s^-1")

    print(
        f"  Observed rates (min/median/max): {observed.min():.4g} / "
        f"{np.median(observed):.4g} / {observed.max():.4g}"
    )
    print(
        f"  Predicted rates (min/median/max): {fitted.min():.4g} / "
        f"{np.median(fitted):.4g} / {fitted.max():.4g}"
    )

    for name, value, (lower, upper) in zip(names, result.x, bounds):
        near_bound = min(value - lower, upper - value) < 0.01 * (upper - lower)
        flag = " [BOUND]" if near_bound else ""
        print(f"  {name}: {value:.6f}{flag}")

    # Surface-state diagnostics.
    print(f"  Maximum site-balance error: {site_balance_error:.3e}")

    for i, name in enumerate(coverage_names):
        values = theta[:, i]
        print(
            f"  {name} (min/median/max): "
            f"{values.min():.4f} / {np.median(values):.4f} / {values.max():.4f}"
        )

    if not np.isfinite(theta).all() or (theta < -1e-8).any() or (theta > 1 + 1e-8).any():
        print("  WARNING: Invalid surface coverages detected.")

    if "ER" in model_name:
        print(
            f"  ER rate fraction (median/max): "
            f"{np.median(er_fraction):.6f} / {er_fraction.max():.6f}"
        )

    if "LH" in model_name:
        print(
            f"  LH rate fraction (median/max): "
            f"{np.median(lh_fraction):.6f} / {lh_fraction.max():.6f}"
        )

    # Match predictions to the experimental conditions.
    condition_index = np.asarray(inputs.model_point_condition_index, dtype=int)
    observation_conditions = condition_index[point_index]

    condition_concentrations = np.exp(np.asarray(inputs.condition_ln_electrolyte_concentration))
    condition_co = np.exp(np.asarray(inputs.condition_ln_CO_mole_fraction))
    point_potential = np.asarray(inputs.model_point_E_V_SHE)

    comparison = pd.DataFrame({
        "C_electrolyte_M": condition_concentrations[observation_conditions],
        "CO_mole_fraction": condition_co[observation_conditions],
        "E_V_SHE": point_potential[point_index],
        "rate_observed": observed,
        "rate_predicted": fitted,
    })

    coverage_data = pd.DataFrame({
        "C_electrolyte_M": condition_concentrations[condition_index],
        "CO_mole_fraction": condition_co[condition_index],
        "E_V_SHE": point_potential,
        "theta_empty": theta[:, 0],
        "theta_CO": theta[:, 1],
        "theta_OH": theta[:, 2],
    })

    concentrations = sorted(comparison["C_electrolyte_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())

    # CO reaction-order sign diagnostics.
    keys = ["C_electrolyte_M", "CO_mole_fraction", "E_V_SHE"]
    means = comparison.groupby(keys, as_index=False)[["rate_observed", "rate_predicted"]].mean()

    print("  Negative CO-order fractions (experiment / model):")

    for lower_co, upper_co in zip(co_values[:-1], co_values[1:]):
        lower = means.loc[np.isclose(means["CO_mole_fraction"], lower_co)]
        upper = means.loc[np.isclose(means["CO_mole_fraction"], upper_co)]

        joined = lower.merge(
            upper, on=["C_electrolyte_M", "E_V_SHE"], suffixes=("_lower", "_upper"),
        )
        if joined.empty:
            continue

        obs_order = np.log(joined["rate_observed_upper"] / joined["rate_observed_lower"])
        pred_order = np.log(joined["rate_predicted_upper"] / joined["rate_predicted_lower"])

        print(
            f"    {100 * lower_co:g}% to {100 * upper_co:g}% CO: "
            f"{np.mean(obs_order < -1e-8):.1%} / {np.mean(pred_order < -1e-8):.1%} "
            f"(n={len(joined)})"
        )

    # Electrolyte reaction-order sign diagnostics.
    ion_orders = []

    for (_, _), group in means.groupby(["CO_mole_fraction", "E_V_SHE"]):
        if group["C_electrolyte_M"].nunique() != len(concentrations):
            continue

        x = np.log(group["C_electrolyte_M"].to_numpy(float))
        obs_order = np.polyfit(x, np.log(group["rate_observed"].to_numpy(float)), 1)[0]
        pred_order = np.polyfit(x, np.log(group["rate_predicted"].to_numpy(float)), 1)[0]
        ion_orders.append((obs_order, pred_order))

    if ion_orders:
        ion_orders = np.asarray(ion_orders)
        ion_label = "H+" if medium == "acidic" else "OH-"
        print(
            f"  Negative {ion_label}-order fractions (experiment / model): "
            f"{np.mean(ion_orders[:, 0] < -1e-8):.1%} / "
            f"{np.mean(ion_orders[:, 1] < -1e-8):.1%} (n={len(ion_orders)})"
        )

    output_dir = ProjectPaths.discover(__file__).figures_dir / "quick_fit_pt100"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experimental versus predicted rate figures.
    fig, axes = plt.subplots(
        len(co_values), len(concentrations),
        figsize=(14, 11), sharex=True, squeeze=False,
    )

    for row, co in enumerate(co_values):
        for col, concentration in enumerate(concentrations):
            ax = axes[row, col]
            panel = comparison.loc[
                np.isclose(comparison["C_electrolyte_M"], concentration)
                & np.isclose(comparison["CO_mole_fraction"], co)
            ].sort_values("E_V_SHE")

            ax.plot(
                panel["E_V_SHE"], panel["rate_observed"],
                ".", color="0.6", markersize=3, label="Experiment",
            )

            predicted_panel = panel.drop_duplicates("E_V_SHE")
            ax.plot(
                predicted_panel["E_V_SHE"], predicted_panel["rate_predicted"],
                color="tab:red", linewidth=1.7, label="Model",
            )

            ax.set_yscale("log")

            if row == 0:
                ax.set_title(f"{concentration:g} M")
            if col == len(concentrations) - 1:
                ax.text(
                    1.03, 0.5, f"{100 * co:g}% CO",
                    transform=ax.transAxes, rotation=-90, va="center",
                )
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"Pt100 {medium}: {model_name}", fontsize=15)
    fig.supxlabel("Potential / V vs SHE")
    fig.supylabel("TOF / s$^{-1}$")
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.96))

    rate_path = output_dir / f"{medium}_{model_name}.png"
    fig.savefig(rate_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Surface coverage figures at the optimized parameter values.
    fig, axes = plt.subplots(
        len(co_values), len(concentrations),
        figsize=(14, 11), sharex=True, sharey=True, squeeze=False,
    )

    coverage_styles = {
        "theta_empty": ("tab:gray", "Free sites"),
        "theta_CO": ("tab:blue", "CO*"),
        "theta_OH": ("tab:orange", "OH*"),
    }

    for row, co in enumerate(co_values):
        for col, concentration in enumerate(concentrations):
            ax = axes[row, col]
            panel = coverage_data.loc[
                np.isclose(coverage_data["C_electrolyte_M"], concentration)
                & np.isclose(coverage_data["CO_mole_fraction"], co)
            ].sort_values("E_V_SHE")

            for column, (color, label) in coverage_styles.items():
                ax.plot(
                    panel["E_V_SHE"], panel[column],
                    color=color, linewidth=1.6, label=label,
                )

            ax.set_ylim(-0.02, 1.02)

            if row == 0:
                ax.set_title(f"{concentration:g} M")
            if col == len(concentrations) - 1:
                ax.text(
                    1.03, 0.5, f"{100 * co:g}% CO",
                    transform=ax.transAxes, rotation=-90, va="center",
                )
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"Pt100 {medium}: {model_name} surface coverages", fontsize=15)
    fig.supxlabel("Potential / V vs SHE")
    fig.supylabel("Surface coverage")
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.96))

    coverage_path = output_dir / f"{medium}_{model_name}_coverages.png"
    fig.savefig(coverage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {rate_path}")
    print(f"  Saved: {coverage_path}")



def main():
    paths = ProjectPaths.discover(__file__)
    config, _ = load_pt100_config(paths)

    for medium in ("acidic", "basic"):
        data = pd.read_parquet(paths.ptru_selected_path(medium))
        data = data.loc[data["material"] == "Pt100"].copy()

        tables = build_model_data(data, electrolyte_concentration_column="C_electrolyte_M")
        inputs = build_model_input_arrays(tables)

        for model_name in available_ptru_models(medium=medium):
            fit_model(inputs, medium, model_name, config["temperature_K"])


if __name__ == "__main__":
    main()
