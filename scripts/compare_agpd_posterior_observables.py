"""Compare posterior observables to experimental summaries for AgPd models.

This script currently combines numerical diagnostics and plotting in one place.
It is functional and retained as-is for reproducibility, with extraction to reusable
post-processing modules planned in a future refactor that must preserve scientific behavior.
"""

from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from mkm.postprocessing.observables import summarize_posterior_linear_observable
from mkm.model_data import build_model_data
from mkm.observable_maps import build_adjacent_log_order_map, build_alpha_map, build_log_slope_order_map
from mkm.models.agpd_basic import available_agpd_models


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_ROOT = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
SELECTED_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_selected.parquet"
SUMMARY_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_summary.parquet"
DELTA_OH_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_delta_OH.parquet"
DELTA_CO_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_delta_CO.parquet"

MODEL_CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
PREPROCESSING_CONFIG_PATH = ROOT / "config" / "preprocessing" / "agpd_basic.yaml"

POSTERIOR_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument(
        "--likelihood",
        choices=["iid", "setup_intercept"],
        default="setup_intercept",
    )
    return parser.parse_args()


def _add_comparison_columns(frame, observed_column, observed_sd_column):

    result = frame.copy()

    result["residual_q50"] = result["q50"] - result[observed_column]
    result["abs_residual_q50"] = np.abs(result["residual_q50"])
    result["standardized_residual_q50"] = result["residual_q50"] / result[observed_sd_column]
    result["experimental_value_inside_posterior_95"] = (
        (result[observed_column] >= result["q025"]) & (result[observed_column] <= result["q975"])
    )

    return result


def _add_chain_spread(frame, observed_sd_column):

    group_columns = [column for column in frame.columns if column not in {"chain", "mean", "sd", "q025", "q50", "q975"}]

    spread = (
        frame.groupby(group_columns, dropna=False)["q50"]
        .agg(chain_q50_min="min", chain_q50_max="max")
        .reset_index()
    )

    spread["chain_q50_range"] = spread["chain_q50_max"] - spread["chain_q50_min"]
    spread["chain_q50_range_over_exp_sd"] = spread["chain_q50_range"] / spread[observed_sd_column]

    return spread


def _print_comparison_summary(name, comparison, chain_spread=None):

    finite_standardized = comparison["standardized_residual_q50"].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\n{name}:")
    print(f"  points: {len(comparison)}")
    print(f"  median |posterior median - experiment|: {comparison['abs_residual_q50'].median():.4f}")
    print(f"  median |standardized residual|: {finite_standardized.abs().median():.3f}")
    print(f"  95% posterior interval contains experimental mean: {comparison['experimental_value_inside_posterior_95'].mean():.1%}")

    if chain_spread is not None:
        finite_chain = chain_spread["chain_q50_range_over_exp_sd"].replace([np.inf, -np.inf], np.nan).dropna()

        print(f"  median chain-median range / experimental SD: {finite_chain.median():.3f}")
        print(f"  95th percentile chain-median range / experimental SD: {finite_chain.quantile(0.95):.3f}")
        print(f"  maximum chain-median range / experimental SD: {finite_chain.max():.3f}")


def _plot_alpha(comparison, output_dir):

    for c_koh, koh_data in comparison.groupby("C_KOH_M", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        axes = axes.ravel()

        for ax, (co_fraction, data) in zip(axes, koh_data.groupby("CO_mole_fraction", sort=True)):
            data = data.sort_values("E_V_SHE")

            ax.fill_between(data["E_V_SHE"], data["q025"], data["q975"], alpha=0.2)
            ax.plot(data["E_V_SHE"], data["q50"], linewidth=1.5, label="posterior")
            ax.errorbar(
                data["E_V_SHE"],
                data["alpha_mean"],
                yerr=data["alpha_sd"],
                fmt="o",
                markersize=3,
                linewidth=0.8,
                label="experiment",
            )

            ax.set_title(f"CO = {co_fraction:g}")
            ax.set_ylabel("alpha")

        axes[-2].set_xlabel("E / V vs SHE")
        axes[-1].set_xlabel("E / V vs SHE")
        axes[0].legend()

        fig.suptitle(f"{MATERIAL}, KOH = {c_koh:g} M")
        fig.tight_layout()
        fig.savefig(output_dir / f"alpha_KOH_{c_koh:g}.png", dpi=180)
        plt.close(fig)


def _plot_delta_oh(comparison, output_dir):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()

    for ax, (co_fraction, data) in zip(axes, comparison.groupby("CO_mole_fraction", sort=True)):
        data = data.sort_values("E_V_SHE")

        ax.fill_between(data["E_V_SHE"], data["q025"], data["q975"], alpha=0.2)
        ax.plot(data["E_V_SHE"], data["q50"], linewidth=1.5, label="posterior")
        ax.errorbar(
            data["E_V_SHE"],
            data["delta_OH"],
            yerr=data["delta_OH_sd"],
            fmt="o",
            markersize=3,
            linewidth=0.8,
            label="experiment",
        )

        ax.set_title(f"CO = {co_fraction:g}")
        ax.set_ylabel("delta_OH")

    axes[-2].set_xlabel("E / V vs SHE")
    axes[-1].set_xlabel("E / V vs SHE")
    axes[0].legend()

    fig.suptitle(MATERIAL)
    fig.tight_layout()
    fig.savefig(output_dir / "delta_OH.png", dpi=180)
    plt.close(fig)


def _plot_delta_co(comparison, output_dir):

    for c_koh, koh_data in comparison.groupby("C_KOH_M", sort=True):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

        grouped = koh_data.groupby(["CO_lower_mole_fraction", "CO_upper_mole_fraction"], sort=True)

        for ax, ((lower_co, upper_co), data) in zip(axes, grouped):
            data = data.sort_values("E_V_SHE")

            ax.fill_between(data["E_V_SHE"], data["q025"], data["q975"], alpha=0.2)
            ax.plot(data["E_V_SHE"], data["q50"], linewidth=1.5, label="posterior")
            ax.errorbar(
                data["E_V_SHE"],
                data["delta_CO"],
                yerr=data["delta_CO_sd"],
                fmt="o",
                markersize=3,
                linewidth=0.8,
                label="experiment",
            )

            ax.set_title(f"{lower_co:g} -> {upper_co:g}")
            ax.set_xlabel("E / V vs SHE")
            ax.set_ylabel("delta_CO")

        axes[0].legend()

        fig.suptitle(f"{MATERIAL}, KOH = {c_koh:g} M")
        fig.tight_layout()
        fig.savefig(output_dir / f"delta_CO_KOH_{c_koh:g}.png", dpi=180)
        plt.close(fig)


def _plot_chain_diagnostics(name, comparison, by_chain, key_columns, observed_column, observed_sd_column, output_dir):

    merged = by_chain.merge(
        comparison[key_columns + [observed_column, observed_sd_column]],
        on=key_columns,
        how="inner",
        validate="many_to_one",
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    for chain, chain_data in merged.groupby("chain", sort=True):
        residual = (chain_data["q50"] - chain_data[observed_column]) / chain_data[observed_sd_column]
        ax.hist(residual.replace([np.inf, -np.inf], np.nan).dropna(), bins=40, alpha=0.35, label=f"chain {chain}")

    ax.set_xlabel("(posterior chain median - experiment) / experimental SD")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_chain_standardized_residuals.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    model_name = args.model
    likelihood_name = args.likelihood

    with open(MODEL_CONFIG_PATH, "r") as file:
        model_config = yaml.safe_load(file)

    with open(PREPROCESSING_CONFIG_PATH, "r") as file:
        preprocessing_config = yaml.safe_load(file)

    selected = pd.read_parquet(SELECTED_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()

    experimental_alpha = pd.read_parquet(SUMMARY_PATH)
    experimental_alpha = experimental_alpha.loc[experimental_alpha["material"] == MATERIAL].copy()

    experimental_oh = pd.read_parquet(DELTA_OH_PATH)
    experimental_oh = experimental_oh.loc[experimental_oh["material"] == MATERIAL].copy()

    experimental_co = pd.read_parquet(DELTA_CO_PATH)
    experimental_co = experimental_co.loc[experimental_co["material"] == MATERIAL].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )

    model_points = model_data.model_points

    if likelihood_name == "setup_intercept":
        posterior_dir = POSTERIOR_ROOT / MATERIAL / "setup_intercept" / model_name
    else:
        new_iid_dir = POSTERIOR_ROOT / MATERIAL / "iid" / model_name
        legacy_iid_dir = POSTERIOR_ROOT / MATERIAL / model_name

        posterior_dir = new_iid_dir if (new_iid_dir / "posterior.nc").exists() else legacy_iid_dir

    posterior_path = posterior_dir / "posterior.nc"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    diagnostic_dir = posterior_dir / "diagnostics" / "observable_comparison"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(posterior_path)

    alpha_map = build_alpha_map(model_points=model_points, temperature_K=model_config["temperature_K"])

    delta_oh_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=preprocessing_config["KOH_concentrations_M"],
        group_columns=["material", "CO_mole_fraction"],
    )

    delta_co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=preprocessing_config["CO_mole_fractions"],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    alpha_summary = summarize_posterior_linear_observable(idata, alpha_map)
    oh_summary = summarize_posterior_linear_observable(idata, delta_oh_map)
    co_summary = summarize_posterior_linear_observable(idata, delta_co_map)

    condition_metadata = model_data.conditions[
        ["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]
    ].copy()

    condition_metadata = condition_metadata.rename(columns={"electrolyte_concentration_M": "C_KOH_M"})

    alpha_pooled = alpha_summary.pooled.merge(condition_metadata, on="condition_id", how="left", validate="many_to_one")
    alpha_by_chain = alpha_summary.by_chain.merge(
        condition_metadata, on="condition_id", how="left", validate="many_to_one"
    )

    alpha_keys = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index"]

    alpha_comparison = alpha_pooled.merge(
        experimental_alpha[alpha_keys + ["alpha_mean", "alpha_sd"]],
        on=alpha_keys,
        how="inner",
        validate="one_to_one",
    )
    alpha_comparison = _add_comparison_columns(alpha_comparison, "alpha_mean", "alpha_sd")

    oh_keys = ["material", "CO_mole_fraction", "analysis_grid_index"]

    oh_comparison = oh_summary.pooled.merge(
        experimental_oh[oh_keys + ["delta_OH", "delta_OH_sd"]],
        on=oh_keys,
        how="inner",
        validate="one_to_one",
    )
    oh_comparison = _add_comparison_columns(oh_comparison, "delta_OH", "delta_OH_sd")

    co_pooled = co_summary.pooled.rename(
        columns={
            "electrolyte_concentration_M": "C_KOH_M",
            "lower_CO_mole_fraction": "CO_lower_mole_fraction",
            "upper_CO_mole_fraction": "CO_upper_mole_fraction",
        }
    )

    co_by_chain = co_summary.by_chain.rename(
        columns={
            "electrolyte_concentration_M": "C_KOH_M",
            "lower_CO_mole_fraction": "CO_lower_mole_fraction",
            "upper_CO_mole_fraction": "CO_upper_mole_fraction",
        }
    )

    co_keys = [
        "material",
        "C_KOH_M",
        "CO_lower_mole_fraction",
        "CO_upper_mole_fraction",
        "analysis_grid_index",
    ]

    co_comparison = co_pooled.merge(
        experimental_co[co_keys + ["delta_CO", "delta_CO_sd"]],
        on=co_keys,
        how="inner",
        validate="one_to_one",
    )
    co_comparison = _add_comparison_columns(co_comparison, "delta_CO", "delta_CO_sd")

    alpha_comparison.to_parquet(diagnostic_dir / "alpha_comparison.parquet", index=False)
    oh_comparison.to_parquet(diagnostic_dir / "delta_OH_comparison.parquet", index=False)
    co_comparison.to_parquet(diagnostic_dir / "delta_CO_comparison.parquet", index=False)

    _plot_alpha(alpha_comparison, diagnostic_dir)
    _plot_delta_oh(oh_comparison, diagnostic_dir)
    _plot_delta_co(co_comparison, diagnostic_dir)

    print(f"\n{MATERIAL}: {model_name}")
    print(f"Likelihood: {likelihood_name}")

    if model_name == "CO_BF_ER_LH":
        alpha_chain = alpha_by_chain.merge(
            alpha_comparison[alpha_keys + ["alpha_sd"]],
            on=alpha_keys,
            how="inner",
            validate="many_to_one",
        )

        oh_chain = oh_summary.by_chain.merge(
            oh_comparison[oh_keys + ["delta_OH_sd"]],
            on=oh_keys,
            how="inner",
            validate="many_to_one",
        )

        co_chain = co_by_chain.merge(
            co_comparison[co_keys + ["delta_CO_sd"]],
            on=co_keys,
            how="inner",
            validate="many_to_one",
        )

        alpha_spread = _add_chain_spread(alpha_chain, "alpha_sd")
        oh_spread = _add_chain_spread(oh_chain, "delta_OH_sd")
        co_spread = _add_chain_spread(co_chain, "delta_CO_sd")

        alpha_spread.to_parquet(diagnostic_dir / "alpha_chain_spread.parquet", index=False)
        oh_spread.to_parquet(diagnostic_dir / "delta_OH_chain_spread.parquet", index=False)
        co_spread.to_parquet(diagnostic_dir / "delta_CO_chain_spread.parquet", index=False)

        _print_comparison_summary("alpha", alpha_comparison, alpha_spread)
        _print_comparison_summary("delta_OH", oh_comparison, oh_spread)
        _print_comparison_summary("delta_CO", co_comparison, co_spread)

        _plot_chain_diagnostics(
            "alpha",
            alpha_comparison,
            alpha_by_chain,
            alpha_keys,
            "alpha_mean",
            "alpha_sd",
            diagnostic_dir,
        )

        _plot_chain_diagnostics(
            "delta_OH",
            oh_comparison,
            oh_summary.by_chain,
            oh_keys,
            "delta_OH",
            "delta_OH_sd",
            diagnostic_dir,
        )

        _plot_chain_diagnostics(
            "delta_CO",
            co_comparison,
            co_by_chain,
            co_keys,
            "delta_CO",
            "delta_CO_sd",
            diagnostic_dir,
        )

    else:
        _print_comparison_summary("alpha", alpha_comparison)
        _print_comparison_summary("delta_OH", oh_comparison)
        _print_comparison_summary("delta_CO", co_comparison)


if __name__ == "__main__":
    main()