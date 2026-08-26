"""Generate persistent numerical and graphical post-processing products for an AgPd posterior fit."""

from argparse import ArgumentParser
import arviz as az
import numpy as np
import pandas as pd

from mkm.models.agpd_basic import available_agpd_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import (
    build_agpd_inputs,
    build_agpd_model_data,
    load_agpd_model_config,
    load_agpd_preprocessing_config,
    validate_agpd_material,
)
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
    plot_loo_pit_conditions,
    plot_loo_pit_coverage,
    plot_loo_pit_ecdf,
    plot_pareto_k,
    plot_pointwise_loo,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import summarize_residual_curves, summarize_shared_replicate_residuals
from mkm.postprocessing.sampling import build_sampling_diagnostics, sampling_parameter_names
from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.loo import compute_loo_diagnostics
from mkm.postprocessing.materials import summarize_material_noise


DEFAULT_MATERIAL = "Ag10Pd90"

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
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept", "mvn"], default="iid")
    return parser.parse_args()


def _load_experimental_observables(paths, material):
    experimental_alpha = pd.read_parquet(paths.agpd_summary_path)
    experimental_oh = pd.read_parquet(paths.agpd_delta_oh_path)
    experimental_co = pd.read_parquet(paths.agpd_delta_co_path)

    experimental_alpha = experimental_alpha.loc[experimental_alpha["material"] == material].copy()
    experimental_oh = experimental_oh.loc[experimental_oh["material"] == material].copy()
    experimental_co = experimental_co.loc[experimental_co["material"] == material].copy()

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
    material = args.material
    likelihood_name = args.likelihood

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    preprocessing_config = load_agpd_preprocessing_config(paths)
    validate_agpd_material(config, material)
    experimental_alpha, experimental_oh, experimental_co = _load_experimental_observables(paths, material)

    posterior_dir = paths.agpd_posterior_dir(material, model_name, likelihood_name)

    output_dir = posterior_dir / "postprocessing"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(posterior_dir / "posterior.nc")
    posterior = idata.posterior
    parameter_specs = config["prior_profiles"][material][model_name]["parameters"]

    model_data = build_agpd_model_data(paths, material)
    inputs = build_agpd_inputs(model_data, config, likelihood_name)

    parameter_contraction = build_parameter_contraction(
        posterior=posterior,
        config=config,
        material=material,
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

    material_noise = summarize_material_noise(posterior)
    material_noise.to_csv(tables_dir / "noise_summary_by_material.csv", index=False)

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

    for distribution in ("mechanism", "conditional", "predictive"):
        plot_observation_grid(
            observations=observation_diagnostics,
            output_path=figures_dir / f"posterior_{distribution}_rate_linear.png",
            residual=False,
            y_scale="linear",
            distribution=distribution,
        )
        plot_observation_grid(
            observations=observation_diagnostics,
            output_path=figures_dir / f"posterior_{distribution}_rate_log.png",
            residual=False,
            y_scale="log",
            distribution=distribution,
        )

    plot_observation_grid(
        observations=observation_diagnostics,
        output_path=figures_dir / "conditional_log_rate_residuals.png",
        residual=True,
        distribution="conditional",
    )

    if likelihood_name == "mvn":
        loo = None
        calibration = None
    else:
        loo = compute_loo_diagnostics(
            inference_data=idata,
            observations=model_data.observations,
            model_name=model_name,
            var_name="ln_rate_observed",
        )
        loo.summary.to_csv(tables_dir / "loo_summary.csv", index=False)
        loo.pointwise.to_parquet(derived_dir / "loo_pointwise.parquet", index=False)

        calibration = compute_normal_loo_pit(
            inference_data=idata,
            loo_result=loo.loo_result,
            observations=model_data.observations,
            inputs=inputs,
            likelihood_name=likelihood_name,
            var_name="ln_rate_observed",
        )
        calibration.summary.to_csv(tables_dir / "loo_pit_summary.csv", index=False)
        calibration.pointwise.to_parquet(derived_dir / "loo_pit.parquet", index=False)

        plot_pointwise_loo(
            pointwise=loo.pointwise,
            model_name=model_name,
            output_path=figures_dir / "pointwise_loo.png",
        )
        plot_pareto_k(
            loo_result=loo.loo_result,
            model_name=model_name,
            output_path=figures_dir / "pareto_k.png",
        )

        pit = calibration.pointwise["loo_pit"].to_numpy(dtype=float)
        plot_loo_pit_ecdf(
            loo_pit=pit,
            model_name=model_name,
            output_path=figures_dir / "loo_pit_ecdf.png",
        )
        plot_loo_pit_coverage(
            loo_pit=pit,
            model_name=model_name,
            output_path=figures_dir / "loo_pit_coverage.png",
        )
        plot_loo_pit_conditions(
            pointwise=calibration.pointwise,
            model_name=model_name,
            output_path=figures_dir / "loo_pit_conditions.png",
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
        material=material,
    )
    plot_delta_oh_comparison(
        comparison=observable_comparisons["delta_OH"].pooled,
        output_path=figures_dir / "delta_OH.png",
        material=material,
    )
    plot_delta_co_comparison(
        comparison=observable_comparisons["delta_CO"].pooled,
        output_dir=figures_dir,
        material=material,
    )

    print(f"\n{material}: {model_name}")
    print(f"Likelihood: {likelihood_name}")

    print("\n=== SAMPLER ===")
    print(sampling.run_summary.to_string(index=False))

    print("\n=== PARAMETER CONTRACTION ===")
    contraction_columns = [
        "parameter",
        "prior_sd",
        "posterior_sd",
        "sd_ratio_posterior_over_prior",
        "posterior_median",
        "posterior_hdi95_lower",
        "posterior_hdi95_upper",
    ]
    print(
        parameter_contraction[contraction_columns]
        .sort_values("sd_ratio_posterior_over_prior")
        .to_string(index=False)
    )

    print("\n=== NOISE ===")
    print(noise_summary.to_string(index=False))
    if not material_noise.empty:
        print("\nMaterial noise parameters:")
        print(material_noise.to_string(index=False))

    print("\n=== RESIDUALS ===")
    mechanism_residual = observation_diagnostics["residual_mechanism"]
    conditional_residual = observation_diagnostics["residual_conditional"]
    standardized = observation_diagnostics["standardized_residual_conditional"]

    print(f"mechanism residual mean: {mechanism_residual.mean():.4f}")
    print(
        "mechanism residual RMS: "
        f"{np.sqrt(np.mean(mechanism_residual.to_numpy() ** 2)):.4f}"
    )
    print(f"conditional residual mean: {conditional_residual.mean():.4f}")
    print(
        "conditional residual RMS: "
        f"{np.sqrt(np.mean(conditional_residual.to_numpy() ** 2)):.4f}"
    )
    print(
        "median |conditional standardized residual|: "
        f"{np.median(np.abs(standardized)):.3f}"
    )
    print(
        "observations inside posterior predictive 95% HDI: "
        f"{observation_diagnostics['observed_inside_predictive_95_hdi'].mean():.1%}"
    )
    print(
        "median mechanism curve lag-1 residual correlation: "
        f"{curve_residuals['mechanism_lag1_residual_correlation'].median():.3f}"
    )
    print(
        "median conditional curve lag-1 residual correlation: "
        f"{curve_residuals['conditional_lag1_residual_correlation'].median():.3f}"
    )
    print(
        "median standardized conditional lag-1 residual correlation: "
        f"{curve_residuals['standardized_conditional_lag1_residual_correlation'].median():.3f}"
    )
    print(
        "median |mechanism residual slope| per V: "
        f"{curve_residuals['mechanism_residual_slope_per_V'].abs().median():.3f}"
    )
    print(
        "median |conditional residual slope| per V: "
        f"{curve_residuals['conditional_residual_slope_per_V'].abs().median():.3f}"
    )

    if not shared_residuals.empty:
        shared_fraction = shared_residuals["shared_fraction_squared_residual"].median()
        print(
            "median replicate-shared mechanism squared-residual fraction: "
            f"{shared_fraction:.3f}"
        )

    print("\n=== EXPERIMENTAL OBSERVABLES ===")
    print(observable_summary.to_string(index=False))

    print("\n=== PSIS-LOO ===")
    if loo is None:
        print(
            "Skipped for MVN likelihood: observation-wise PSIS-LOO is not valid when "
            "potential points within a sweep are conditionally correlated."
        )
    else:
        print(loo.summary.to_string(index=False))

    print("\n=== LOO-PIT CALIBRATION ===")
    if calibration is None:
        print(
            "Skipped for MVN likelihood: the current LOO-PIT implementation assumes "
            "observation-wise independent likelihood terms."
        )
    else:
        print(calibration.summary.to_string(index=False))

    if likelihood_name == "setup_intercept":
        print(
            "LOO scope: observation-wise conditional prediction; other observations from the same "
            "setup remain available when one observation is held out."
        )

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