"""Postprocess a multi-material AgPd composition posterior fit."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import yaml

from mkm.models.agpd_basic import available_agpd_composition_models
from mkm.observable_maps import build_adjacent_log_order_map, build_alpha_map, build_log_slope_order_map
from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.diagnostics import (
    build_balance_summary,
    build_noise_summary,
    build_parameter_contraction,
    build_physical_summary,
)
from mkm.postprocessing.loo import compute_loo_diagnostics
from mkm.postprocessing.materials import (
    summarize_material_noise,
    summarize_observation_diagnostics_by_material,
    summarize_pointwise_loo_by_material,
    summarize_residual_structure_by_material,
)
from mkm.postprocessing.observable_comparison import (
    ExperimentalObservableComparison,
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
    plot_loo_pit_conditions,
    plot_loo_pit_coverage,
    plot_loo_pit_ecdf,
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pareto_k,
    plot_pointwise_loo,
    plot_pointwise_variable,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import summarize_residual_curves, summarize_shared_replicate_residuals
from mkm.postprocessing.sampling import build_sampling_diagnostics, sampling_parameter_names
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import (
    build_agpd_composition_model_data,
    build_agpd_inputs,
    load_agpd_model_config,
    load_agpd_preprocessing_config,
)


DEFAULT_COMPOSITION_MODEL = "shared"

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
    parser.add_argument("model", choices=available_agpd_composition_models())
    parser.add_argument("--composition-model", default=DEFAULT_COMPOSITION_MODEL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="setup_intercept")
    return parser.parse_args()


def _load_run_metadata(posterior_dir):
    path = posterior_dir / "run_metadata.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Composition posterior metadata not found: {path}. "
            "New composition fits are expected to persist their fitted material set."
        )

    with open(path, "r") as file:
        metadata = yaml.safe_load(file)

    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid run metadata: {path}")

    return metadata


def _load_experimental_observables(paths, materials):
    material_set = set(materials)

    alpha = pd.read_parquet(paths.agpd_summary_path)
    oh = pd.read_parquet(paths.agpd_delta_oh_path)
    co = pd.read_parquet(paths.agpd_delta_co_path)

    alpha = alpha.loc[alpha["material"].isin(material_set)].copy()
    oh = oh.loc[oh["material"].isin(material_set)].copy()
    co = co.loc[co["material"].isin(material_set)].copy()

    return alpha, oh, co


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

    alpha_comparison = build_experimental_observable_comparison(
        pooled=alpha_pooled,
        by_chain=alpha_by_chain,
        experimental=experimental_alpha,
        key_columns=["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index"],
        observed_column="alpha_mean",
        observed_sd_column="alpha_sd",
    )
    oh_comparison = build_experimental_observable_comparison(
        pooled=oh_summary.pooled,
        by_chain=oh_summary.by_chain,
        experimental=experimental_oh,
        key_columns=["material", "CO_mole_fraction", "analysis_grid_index"],
        observed_column="delta_OH",
        observed_sd_column="delta_OH_sd",
    )
    co_comparison = build_experimental_observable_comparison(
        pooled=co_pooled,
        by_chain=co_by_chain,
        experimental=experimental_co,
        key_columns=[
            "material",
            "C_KOH_M",
            "CO_lower_mole_fraction",
            "CO_upper_mole_fraction",
            "analysis_grid_index",
        ],
        observed_column="delta_CO",
        observed_sd_column="delta_CO_sd",
    )

    return {
        "alpha": alpha_comparison,
        "delta_OH": oh_comparison,
        "delta_CO": co_comparison,
    }


def _subset_comparison(comparison, material):
    return ExperimentalObservableComparison(
        pooled=comparison.pooled.loc[comparison.pooled["material"] == material].copy(),
        by_chain=comparison.by_chain.loc[comparison.by_chain["material"] == material].copy(),
        chain_spread=comparison.chain_spread.loc[comparison.chain_spread["material"] == material].copy(),
    )


def _save_observable_comparisons(comparisons, tables_dir, derived_dir, materials):
    for name, result in comparisons.items():
        result.pooled.to_parquet(derived_dir / f"{name}_comparison.parquet", index=False)
        result.by_chain.to_parquet(derived_dir / f"{name}_by_chain.parquet", index=False)
        result.chain_spread.to_parquet(derived_dir / f"{name}_chain_spread.parquet", index=False)

    records = []

    for material in materials:
        for name, comparison in comparisons.items():
            subset = _subset_comparison(comparison, material)
            if subset.pooled.empty:
                continue

            summary = summarize_experimental_observable(name, subset)
            summary["material"] = material
            records.append(summary)

    frame = pd.DataFrame(records)
    frame.to_csv(tables_dir / "experimental_observable_summary_by_material.csv", index=False)
    return frame


def _pointwise_variable_is_applicable(variable_name, material, config):
    composition = config["surface_composition"][material]

    if variable_name in {"theta_OH_Ag", "theta_empty_Ag"}:
        return float(composition["Ag_fraction"]) > 0.0

    if variable_name in {"theta_CO", "theta_OH_Pd", "theta_empty_Pd"}:
        return float(composition["Pd_fraction"]) > 0.0

    return True


def main():
    args = parse_args()
    model_name = args.model
    likelihood_name = args.likelihood
    composition_model = args.composition_model

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    preprocessing_config = load_agpd_preprocessing_config(paths)

    posterior_dir = paths.agpd_composition_posterior_output_dir(
        composition_model=composition_model,
        model_name=model_name,
        likelihood_name=likelihood_name,
    )
    posterior_path = posterior_dir / "posterior.nc"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    metadata = _load_run_metadata(posterior_dir)
    materials = tuple(metadata["materials"])
    prior_material = metadata["prior_profile_material"]

    if metadata.get("model") != model_name:
        raise ValueError(
            f"Run metadata model '{metadata.get('model')}' does not match requested model '{model_name}'."
        )
    if metadata.get("likelihood") != likelihood_name:
        raise ValueError(
            f"Run metadata likelihood '{metadata.get('likelihood')}' does not match requested "
            f"likelihood '{likelihood_name}'."
        )

    model_data = build_agpd_composition_model_data(paths, materials)
    inputs = build_agpd_inputs(model_data, config, likelihood_name)

    output_dir = posterior_dir / "postprocessing"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"
    by_material_dir = figures_dir / "by_material"

    for path in (tables_dir, derived_dir, figures_dir, by_material_dir):
        path.mkdir(parents=True, exist_ok=True)

    idata = az.from_netcdf(posterior_path)
    posterior = idata.posterior
    parameter_specs = config["prior_profiles"][prior_material][model_name]["parameters"]

    parameter_contraction = build_parameter_contraction(
        posterior=posterior,
        config=config,
        material=prior_material,
        model_name=model_name,
    )
    parameter_contraction.insert(0, "prior_profile_material", prior_material)
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

    material_observation_summary = summarize_observation_diagnostics_by_material(observation_diagnostics)

    curve_residuals = summarize_residual_curves(observation_diagnostics)
    curve_residuals.to_csv(tables_dir / "residual_curve_summary.csv", index=False)

    shared_residuals = summarize_shared_replicate_residuals(observation_diagnostics)
    shared_residuals.to_csv(tables_dir / "shared_replicate_residual_summary.csv", index=False)

    residual_structure = summarize_residual_structure_by_material(curve_residuals, shared_residuals)
    material_residual_summary = material_observation_summary.merge(
        residual_structure,
        on="material",
        how="left",
        validate="one_to_one",
    )
    material_residual_summary.to_csv(tables_dir / "residual_summary_by_material.csv", index=False)

    loo = compute_loo_diagnostics(
        inference_data=idata,
        observations=model_data.observations,
        model_name=model_name,
        var_name="ln_rate_observed",
    )
    loo.summary.to_csv(tables_dir / "loo_summary.csv", index=False)
    loo.pointwise.to_parquet(derived_dir / "loo_pointwise.parquet", index=False)

    loo_by_material = summarize_pointwise_loo_by_material(loo.pointwise)
    loo_by_material.to_csv(tables_dir / "loo_contribution_by_material.csv", index=False)

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

    plot_pareto_k(
        loo_result=loo.loo_result,
        model_name=f"{model_name}, shared composition",
        output_path=figures_dir / "pareto_k.png",
    )

    pit = calibration.pointwise["loo_pit"].to_numpy(dtype=float)
    plot_loo_pit_ecdf(
        loo_pit=pit,
        model_name=f"{model_name}, shared composition",
        output_path=figures_dir / "loo_pit_ecdf.png",
    )
    plot_loo_pit_coverage(
        loo_pit=pit,
        model_name=f"{model_name}, shared composition",
        output_path=figures_dir / "loo_pit_coverage.png",
    )

    physical_summary = build_physical_summary(posterior)
    physical_summary.to_csv(tables_dir / "physical_summary.csv", index=False)

    balance_summary = build_balance_summary(posterior)
    balance_summary.to_csv(tables_dir / "balance_summary.csv", index=False)

    pointwise_summaries = {}

    for name in POINTWISE_VARIABLES:
        if name not in posterior:
            continue

        summary = summarize_pointwise_posterior_variable(
            inference_data=idata,
            model_points=model_data.model_points,
            variable_name=name,
        )
        pointwise_summaries[name] = summary
        summary.to_parquet(derived_dir / f"{name}.parquet", index=False)

    experimental_alpha, experimental_oh, experimental_co = _load_experimental_observables(paths, materials)
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
        materials=materials,
    )

    for material in materials:
        material_dir = by_material_dir / material
        material_dir.mkdir(parents=True, exist_ok=True)

        observations = observation_diagnostics.loc[
            observation_diagnostics["material"] == material
        ].copy()

        for distribution in ("mechanism", "conditional", "predictive"):
            plot_observation_grid(
                observations=observations,
                output_path=material_dir / f"posterior_{distribution}_rate_linear.png",
                residual=False,
                y_scale="linear",
                distribution=distribution,
                context_label=material,
            )
            plot_observation_grid(
                observations=observations,
                output_path=material_dir / f"posterior_{distribution}_rate_log.png",
                residual=False,
                y_scale="log",
                distribution=distribution,
                context_label=material,
            )

        plot_observation_grid(
            observations=observations,
            output_path=material_dir / "conditional_log_rate_residuals.png",
            residual=True,
            distribution="conditional",
            context_label=material,
        )

        for variable_name, summary in pointwise_summaries.items():
            if not _pointwise_variable_is_applicable(variable_name, material, config):
                continue

            material_summary = summary.loc[summary["material"] == material].copy()
            if material_summary.empty:
                continue

            plot_pointwise_variable(
                summary=material_summary,
                variable_name=variable_name,
                output_path=material_dir / f"{variable_name}.png",
                context_label=material,
            )

        material_loo = loo.pointwise.loc[loo.pointwise["material"] == material].copy()
        plot_pointwise_loo(
            pointwise=material_loo,
            model_name=f"{model_name}, {material}",
            output_path=material_dir / "pointwise_loo.png",
        )

        material_pit = calibration.pointwise.loc[
            calibration.pointwise["material"] == material
        ].copy()
        plot_loo_pit_conditions(
            pointwise=material_pit,
            model_name=f"{model_name}, {material}",
            output_path=material_dir / "loo_pit_conditions.png",
        )
        plot_loo_pit_ecdf(
            loo_pit=material_pit["loo_pit"].to_numpy(dtype=float),
            model_name=f"{model_name}, {material}",
            output_path=material_dir / "loo_pit_ecdf.png",
        )
        plot_loo_pit_coverage(
            loo_pit=material_pit["loo_pit"].to_numpy(dtype=float),
            model_name=f"{model_name}, {material}",
            output_path=material_dir / "loo_pit_coverage.png",
        )

        alpha = _subset_comparison(observable_comparisons["alpha"], material)
        oh = _subset_comparison(observable_comparisons["delta_OH"], material)
        co = _subset_comparison(observable_comparisons["delta_CO"], material)

        if not alpha.pooled.empty:
            plot_alpha_comparison(alpha.pooled, material_dir, material=material)
        if not oh.pooled.empty:
            plot_delta_oh_comparison(
                oh.pooled,
                material_dir / "delta_OH.png",
                material=material,
            )
        if not co.pooled.empty:
            plot_delta_co_comparison(co.pooled, material_dir, material=material)

    print(f"\nAgPd composition posterior: {model_name}")
    print(f"Composition parameterization: {composition_model}")
    print(f"Likelihood: {likelihood_name}")
    print(f"Materials: {', '.join(materials)}")
    print(f"Prior profile source: {prior_material}")

    print("\n=== SAMPLER ===")
    print(sampling.run_summary.to_string(index=False))

    print("\n=== MATERIAL NOISE ===")
    print(material_noise.to_string(index=False))

    print("\n=== MATERIAL RESIDUALS ===")
    print(material_residual_summary.to_string(index=False))

    print("\n=== MATERIAL OBSERVABLES ===")
    print(observable_summary.to_string(index=False))

    print("\n=== PSIS-LOO ===")
    print(loo.summary.to_string(index=False))
    print(
        "\nMaterial LOO rows below are contributions to observation-wise LOO, not leave-one-material-out validation."
    )
    print(loo_by_material.to_string(index=False))

    print("\n=== PHYSICAL VARIABLES ===")
    print(physical_summary.to_string(index=False))

    print("\n=== BALANCE CHECKS ===")
    print(balance_summary.to_string(index=False))

    print(f"\nSaved composition postprocessing to: {output_dir}")


if __name__ == "__main__":
    main()
