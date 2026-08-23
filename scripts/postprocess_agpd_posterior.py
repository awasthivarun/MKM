"""Generate persistent numerical and graphical post-processing products for an AgPd posterior fit."""

from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import yaml

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.agpd_basic import available_agpd_models
from mkm.observable_maps import build_adjacent_log_order_map, build_alpha_map, build_log_slope_order_map
from mkm.postprocessing.diagnostics import (
    build_balance_summary,
    build_noise_summary,
    build_parameter_contraction,
    build_physical_summary,
)
from mkm.postprocessing.observable_comparison import (
    build_experimental_observable_comparison,
    summarize_experimental_observable,
)
from mkm.postprocessing.observables import (
    summarize_pointwise_posterior_variable,
    summarize_posterior_linear_observable,
)
from mkm.postprocessing.plotting import (
    plot_alpha_comparison,
    plot_delta_co_comparison,
    plot_delta_oh_comparison,
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_variable,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import summarize_residual_curves, summarize_shared_replicate_residuals
from mkm.postprocessing.sampling import build_sampling_diagnostics, sampling_parameter_names


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_ROOT = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
DATA_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_selected.parquet"
SUMMARY_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_summary.parquet"
DELTA_OH_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_delta_OH.parquet"
DELTA_CO_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_delta_CO.parquet"

CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
PREPROCESSING_CONFIG_PATH = ROOT / "config" / "preprocessing" / "agpd_basic.yaml"
POSTERIOR_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"

POINTWISE_VARIABLES = (
    "theta_CO",
    "theta_OH_Pd",
    "theta_empty_Pd",
    "theta_OH_Ag",
    "theta_empty_Ag",
    "rate_fraction_BF",
    "rate_fraction_ER",
    "rate_fraction_LH",
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="setup_intercept")
    return parser.parse_args()


def _get_posterior_dir(model_name, likelihood_name):
    if likelihood_name == "setup_intercept":
        path = POSTERIOR_ROOT / MATERIAL / "setup_intercept" / model_name
    else:
        new_path = POSTERIOR_ROOT / MATERIAL / "iid" / model_name
        legacy_path = POSTERIOR_ROOT / MATERIAL / model_name
        path = new_path if (new_path / "posterior.nc").exists() else legacy_path

    posterior_path = path / "posterior.nc"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    return path


def _load_configs():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    with open(PREPROCESSING_CONFIG_PATH, "r") as file:
        preprocessing_config = yaml.safe_load(file)

    return config, preprocessing_config


def _load_experimental_observables():
    experimental_alpha = pd.read_parquet(SUMMARY_PATH)
    experimental_oh = pd.read_parquet(DELTA_OH_PATH)
    experimental_co = pd.read_parquet(DELTA_CO_PATH)

    experimental_alpha = experimental_alpha.loc[experimental_alpha["material"] == MATERIAL].copy()
    experimental_oh = experimental_oh.loc[experimental_oh["material"] == MATERIAL].copy()
    experimental_co = experimental_co.loc[experimental_co["material"] == MATERIAL].copy()

    return experimental_alpha, experimental_oh, experimental_co


def _build_observable_comparisons(
    idata,
    model_data,
    config,
    preprocessing_config,
    experimental_alpha,
    experimental_oh,
    experimental_co,
):
    model_points = model_data.model_points

    alpha_map = build_alpha_map(model_points=model_points, temperature_K=config["temperature_K"])

    oh_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=preprocessing_config["KOH_concentrations_M"],
        group_columns=["material", "CO_mole_fraction"],
    )

    co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=preprocessing_config["CO_mole_fractions"],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    alpha_summary = summarize_posterior_linear_observable(idata, alpha_map)
    oh_summary = summarize_posterior_linear_observable(idata, oh_map)
    co_summary = summarize_posterior_linear_observable(idata, co_map)

    condition_metadata = model_data.conditions[
        ["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]
    ].rename(columns={"electrolyte_concentration_M": "C_KOH_M"})

    alpha_pooled = alpha_summary.pooled.merge(
        condition_metadata,
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    alpha_by_chain = alpha_summary.by_chain.merge(
        condition_metadata,
        on="condition_id",
        how="left",
        validate="many_to_one",
    )

    co_rename = {
        "electrolyte_concentration_M": "C_KOH_M",
        "lower_CO_mole_fraction": "CO_lower_mole_fraction",
        "upper_CO_mole_fraction": "CO_upper_mole_fraction",
    }
    co_pooled = co_summary.pooled.rename(columns=co_rename)
    co_by_chain = co_summary.by_chain.rename(columns=co_rename)

    alpha_keys = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index"]
    oh_keys = ["material", "CO_mole_fraction", "analysis_grid_index"]
    co_keys = [
        "material",
        "C_KOH_M",
        "CO_lower_mole_fraction",
        "CO_upper_mole_fraction",
        "analysis_grid_index",
    ]

    alpha_comparison = build_experimental_observable_comparison(
        pooled=alpha_pooled,
        by_chain=alpha_by_chain,
        experimental=experimental_alpha,
        key_columns=alpha_keys,
        observed_column="alpha_mean",
        observed_sd_column="alpha_sd",
    )

    oh_comparison = build_experimental_observable_comparison(
        pooled=oh_summary.pooled,
        by_chain=oh_summary.by_chain,
        experimental=experimental_oh,
        key_columns=oh_keys,
        observed_column="delta_OH",
        observed_sd_column="delta_OH_sd",
    )

    co_comparison = build_experimental_observable_comparison(
        pooled=co_pooled,
        by_chain=co_by_chain,
        experimental=experimental_co,
        key_columns=co_keys,
        observed_column="delta_CO",
        observed_sd_column="delta_CO_sd",
    )

    return {
        "alpha": alpha_comparison,
        "delta_OH": oh_comparison,
        "delta_CO": co_comparison,
    }


def _save_observable_comparisons(comparisons, tables_dir, derived_dir):
    for name, result in comparisons.items():
        result.pooled.to_parquet(derived_dir / f"{name}_comparison.parquet", index=False)
        result.by_chain.to_parquet(derived_dir / f"{name}_by_chain.parquet", index=False)
        result.chain_spread.to_parquet(derived_dir / f"{name}_chain_spread.parquet", index=False)

    summary = pd.DataFrame(
        [
            summarize_experimental_observable("alpha", comparisons["alpha"]),
            summarize_experimental_observable("delta_OH", comparisons["delta_OH"]),
            summarize_experimental_observable("delta_CO", comparisons["delta_CO"]),
        ]
    )
    summary.to_csv(tables_dir / "experimental_observable_summary.csv", index=False)

    return summary


def main():
    args = parse_args()
    model_name = args.model
    likelihood_name = args.likelihood

    config, preprocessing_config = _load_configs()
    experimental_alpha, experimental_oh, experimental_co = _load_experimental_observables()

    posterior_dir = _get_posterior_dir(model_name=model_name, likelihood_name=likelihood_name)

    output_dir = posterior_dir / "postprocessing"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(posterior_dir / "posterior.nc")
    posterior = idata.posterior
    parameter_specs = config["prior_profiles"][MATERIAL][model_name]["parameters"]

    selected = pd.read_parquet(DATA_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )

    if likelihood_name == "setup_intercept":
        setup_config = config["likelihood"]["setup_intercept"]
        inputs = build_model_input_arrays(
            model_data,
            setup_group_columns=setup_config["group_columns"],
            setup_zero_sum_columns=setup_config["zero_sum_within"],
        )
    else:
        inputs = build_model_input_arrays(model_data)

    parameter_contraction = build_parameter_contraction(
        posterior=posterior,
        config=config,
        material=MATERIAL,
        model_name=model_name,
    )
    parameter_contraction.to_csv(tables_dir / "parameter_contraction.csv", index=False)

    plot_parameter_posteriors(
        posterior=posterior,
        parameter_specs=parameter_specs,
        output_path=figures_dir / "posterior_parameters.png",
    )

    sampling_names = sampling_parameter_names(posterior, parameter_specs)
    sampling = build_sampling_diagnostics(idata, sampling_names)

    sampling.parameter_summary.to_csv(tables_dir / "sampler_parameter_diagnostics.csv", index=False)
    sampling.run_summary.to_csv(tables_dir / "sampler_run_summary.csv", index=False)
    sampling.bfmi_by_chain.to_csv(tables_dir / "sampler_bfmi_by_chain.csv", index=False)

    plot_sampling_trace(idata, sampling_names, figures_dir / "sampler_trace.png")
    plot_sampling_rank(idata, sampling_names, figures_dir / "sampler_rank.png")
    plot_sampling_energy(idata, sampling_names, figures_dir / "sampler_energy.png")
    plot_sampling_pairs(idata, sampling_names, figures_dir / "sampler_pairs.png")

    noise_summary = build_noise_summary(posterior)
    noise_summary.to_csv(tables_dir / "noise_summary.csv", index=False)

    observation_diagnostics = build_observation_diagnostics(
        inference_data=idata,
        model_data=model_data,
        inputs=inputs,
        likelihood_name=likelihood_name,
    )
    observation_diagnostics.to_parquet(derived_dir / "observation_diagnostics.parquet", index=False)

    curve_residuals = summarize_residual_curves(observation_diagnostics)
    curve_residuals.to_csv(tables_dir / "residual_curve_summary.csv", index=False)

    shared_residuals = summarize_shared_replicate_residuals(observation_diagnostics)
    shared_residuals.to_csv(tables_dir / "shared_replicate_residual_summary.csv", index=False)

    plot_observation_grid(
        observations=observation_diagnostics,
        output_path=figures_dir / "posterior_predictive_log_rate.png",
        residual=False,
    )
    plot_observation_grid(
        observations=observation_diagnostics,
        output_path=figures_dir / "conditional_log_rate_residuals.png",
        residual=True,
    )

    physical_summary = build_physical_summary(posterior)
    physical_summary.to_csv(tables_dir / "physical_summary.csv", index=False)

    balance_summary = build_balance_summary(posterior)
    balance_summary.to_csv(tables_dir / "balance_summary.csv", index=False)

    for name in POINTWISE_VARIABLES:
        if name not in posterior:
            continue

        summary = summarize_pointwise_posterior_variable(
            inference_data=idata,
            model_points=model_data.model_points,
            variable_name=name,
        )
        summary.to_parquet(derived_dir / f"{name}.parquet", index=False)
        plot_pointwise_variable(summary=summary, variable_name=name, output_path=figures_dir / f"{name}.png")

    observable_comparisons = _build_observable_comparisons(
        idata=idata,
        model_data=model_data,
        config=config,
        preprocessing_config=preprocessing_config,
        experimental_alpha=experimental_alpha,
        experimental_oh=experimental_oh,
        experimental_co=experimental_co,
    )
    observable_summary = _save_observable_comparisons(
        comparisons=observable_comparisons,
        tables_dir=tables_dir,
        derived_dir=derived_dir,
    )

    plot_alpha_comparison(
        comparison=observable_comparisons["alpha"].pooled,
        output_dir=figures_dir,
        material=MATERIAL,
    )
    plot_delta_oh_comparison(
        comparison=observable_comparisons["delta_OH"].pooled,
        output_path=figures_dir / "delta_OH.png",
        material=MATERIAL,
    )
    plot_delta_co_comparison(
        comparison=observable_comparisons["delta_CO"].pooled,
        output_dir=figures_dir,
        material=MATERIAL,
    )

    print(f"\n{MATERIAL}: {model_name}")
    print(f"Likelihood: {likelihood_name}")

    print("\n=== SAMPLER ===")
    print(sampling.run_summary.to_string(index=False))

    print("\n=== PARAMETER CONTRACTION ===")
    contraction_columns = [
        "parameter",
        "prior_sd",
        "posterior_sd",
        "sd_ratio_posterior_over_prior",
        "posterior_q025",
        "posterior_q50",
        "posterior_q975",
    ]
    print(
        parameter_contraction[contraction_columns]
        .sort_values("sd_ratio_posterior_over_prior")
        .to_string(index=False)
    )

    print("\n=== NOISE ===")
    print(noise_summary.to_string(index=False))

    print("\n=== RESIDUALS ===")
    residual = observation_diagnostics["residual_conditional"]
    standardized = observation_diagnostics["standardized_residual_conditional"]

    print(f"conditional residual mean: {residual.mean():.4f}")
    print(f"conditional residual RMS: {np.sqrt(np.mean(residual.to_numpy() ** 2)):.4f}")
    print(f"median |conditional standardized residual|: {np.median(np.abs(standardized)):.3f}")
    print(
        "observations inside posterior predictive 95%: "
        f"{observation_diagnostics['observed_inside_predictive_95'].mean():.1%}"
    )
    print(f"median curve lag-1 residual correlation: {curve_residuals['lag1_residual_correlation'].median():.3f}")
    print(f"median |residual slope| per V: {curve_residuals['residual_slope_per_V'].abs().median():.3f}")

    if not shared_residuals.empty:
        shared_fraction = shared_residuals["shared_fraction_squared_residual"].median()
        print(f"median shared squared-residual fraction: {shared_fraction:.3f}")

    print("\n=== EXPERIMENTAL OBSERVABLES ===")
    print(observable_summary.to_string(index=False))

    print("\n=== PHYSICAL VARIABLES ===")
    if physical_summary.empty:
        print("No configured coverage/pathway variables found.")
    else:
        print(physical_summary.to_string(index=False))

    print("\n=== BALANCE CHECKS ===")
    if balance_summary.empty:
        print("No applicable balance checks.")
    else:
        print(balance_summary.to_string(index=False))

    print(f"\nSaved post-processing to: {output_dir}")


if __name__ == "__main__":
    main()