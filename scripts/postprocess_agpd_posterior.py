"""Generate persistent numerical and graphical post-processing products for an AgPd posterior fit.

This script currently contains both reusable calculations and plotting/report assembly.
It is functional and intentionally preserved during housekeeping, with future extraction
to dedicated post-processing modules planned without changing scientific behavior here.
"""

from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import yaml
from mkm.postprocessing.diagnostics import (
    build_balance_summary,
    build_noise_summary,
    build_parameter_contraction,
    build_physical_summary,
    flatten_posterior_samples,
    summarize_samples,
)
from mkm.postprocessing.predictions import (
    build_observation_diagnostics,
)
from mkm.postprocessing.residuals import (
    summarize_residual_curves,
    summarize_shared_replicate_residuals,
)
from mkm.postprocessing.observables import summarize_pointwise_posterior_variable
from mkm.postprocessing.sampling import build_sampling_diagnostics, sampling_parameter_names
from mkm.postprocessing.plotting import (
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_variable,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
)

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.agpd_basic import available_agpd_models


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AgPd_COOx_basic"
    / "analysis"
    / "AgPd_COOx_basic_selected.parquet"
)

CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
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


def main():
    args = parse_args()

    model_name = args.model
    likelihood_name = args.likelihood

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

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
    selected = selected[selected["material"] == MATERIAL].copy()

    model_data = build_model_data(selected_replicates=selected, electrolyte_concentration_column="C_KOH_M")

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

    noise_summary = build_noise_summary(posterior)
    noise_summary.to_csv(tables_dir / "noise_summary.csv", index=False)

    observation_diagnostics = build_observation_diagnostics(
        inference_data=idata,
        model_data=model_data,
        inputs=inputs,
        likelihood_name=likelihood_name,
    )
    observation_diagnostics.to_parquet(derived_dir / "observation_diagnostics.parquet", index=False)

    curve_residuals = summarize_residual_curves(
        observation_diagnostics
    )
    curve_residuals.to_csv(tables_dir / "residual_curve_summary.csv", index=False)

    shared_residuals = summarize_shared_replicate_residuals(
        observation_diagnostics
    )
    shared_residuals.to_csv(tables_dir / "shared_replicate_residual_summary.csv", index=False)

    physical_summary = build_physical_summary(posterior)
    physical_summary.to_csv(tables_dir / "physical_summary.csv", index=False)

    balance_summary = build_balance_summary(posterior)
    balance_summary.to_csv(tables_dir / "balance_summary.csv", index=False)

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

    pointwise_names = [
        "theta_CO",
        "theta_OH_Pd",
        "theta_empty_Pd",
        "theta_OH_Ag",
        "theta_empty_Ag",
        "rate_fraction_BF",
        "rate_fraction_ER",
        "rate_fraction_LH",
    ]

    for name in pointwise_names:
        if name not in posterior:
            continue

        summary = summarize_pointwise_posterior_variable(
            inference_data=idata,
            model_points=model_data.model_points,
            variable_name=name,
        )

        summary.to_parquet(derived_dir / f"{name}.parquet", index=False)
        plot_pointwise_variable(summary=summary, variable_name=name, output_path=figures_dir / f"{name}.png")

    sampling_names = sampling_parameter_names(posterior, parameter_specs)
    sampling = build_sampling_diagnostics(idata, sampling_names)

    sampling.parameter_summary.to_csv(tables_dir / "sampler_parameter_diagnostics.csv", index=False)
    sampling.run_summary.to_csv(tables_dir / "sampler_run_summary.csv", index=False)
    sampling.bfmi_by_chain.to_csv(tables_dir / "sampler_bfmi_by_chain.csv", index=False)

    plot_sampling_trace(idata, sampling_names, figures_dir / "sampler_trace.png")
    plot_sampling_rank(idata, sampling_names, figures_dir / "sampler_rank.png")
    plot_sampling_energy(idata, sampling_names, figures_dir / "sampler_energy.png")
    plot_sampling_pairs(idata, sampling_names, figures_dir / "sampler_pairs.png")

    print(f"\n{MATERIAL}: {model_name}")
    print(f"Likelihood: {likelihood_name}")

    print("\n=== SAMPLER ===")
    print(sampling.run_summary.to_string(index=False))

    print("\n=== PARAMETER CONTRACTION ===")
    print(
        parameter_contraction[
            [
                "parameter",
                "prior_sd",
                "posterior_sd",
                "sd_ratio_posterior_over_prior",
                "posterior_q025",
                "posterior_q50",
                "posterior_q975",
            ]
        ]
        .sort_values("sd_ratio_posterior_over_prior")
        .to_string(index=False)
    )

    print("\n=== NOISE ===")
    print(noise_summary.to_string(index=False))

    print("\n=== RESIDUALS ===")
    residual = observation_diagnostics["residual_conditional"]
    standardized = observation_diagnostics[
        "standardized_residual_conditional"
    ]

    print(
        f"conditional residual mean: "
        f"{residual.mean():.4f}"
    )

    print(
        f"conditional residual RMS: "
        f"{np.sqrt(np.mean(residual.to_numpy() ** 2)):.4f}"
    )

    print(
        f"median |conditional standardized residual|: "
        f"{np.median(np.abs(standardized)):.3f}"
    )

    print(
        f"observations inside posterior predictive 95%: "
        f"{observation_diagnostics['observed_inside_predictive_95'].mean():.1%}"
    )

    print(
        f"median curve lag-1 residual correlation: "
        f"{curve_residuals['lag1_residual_correlation'].median():.3f}"
    )

    print(
        f"median |residual slope| per V: "
        f"{curve_residuals['residual_slope_per_V'].abs().median():.3f}"
    )

    if not shared_residuals.empty:
        print(
            "median shared squared-residual fraction: "
            f"{shared_residuals[
                'shared_fraction_squared_residual'
            ].median():.3f}"
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