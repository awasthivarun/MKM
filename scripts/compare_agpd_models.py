"""Compare fitted AgPd microkinetic models using PSIS-LOO and LOO predictive calibration."""

from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import pandas as pd
import yaml

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.model_comparison import build_loo_model_comparison
from mkm.postprocessing.plotting import (
    plot_loo_comparison,
    plot_loo_pit_conditions,
    plot_loo_pit_coverage,
    plot_loo_pit_ecdf,
    plot_pareto_k,
    plot_pointwise_elpd_difference,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
DATA_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_selected.parquet"
CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
POSTERIOR_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"
LOO_VAR_NAME = "ln_rate_observed"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="setup_intercept")
    parser.add_argument("--models", nargs="+", choices=available_agpd_models(), default=None)
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


def _get_output_dir(likelihood_name):
    return POSTERIOR_ROOT / MATERIAL / likelihood_name / "model_comparison"


def _build_inputs(model_data, config, likelihood_name):
    if likelihood_name == "setup_intercept":
        setup_config = config["likelihood"]["setup_intercept"]
        return build_model_input_arrays(
            model_data,
            setup_group_columns=setup_config["group_columns"],
            setup_zero_sum_columns=setup_config["zero_sum_within"],
        )

    return build_model_input_arrays(model_data)


def main():
    args = parse_args()
    likelihood_name = args.likelihood
    model_names = args.models or list(available_agpd_models())

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    selected = pd.read_parquet(DATA_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )
    inputs = _build_inputs(
        model_data=model_data,
        config=config,
        likelihood_name=likelihood_name,
    )

    inference_data_by_model = {
        model_name: az.from_netcdf(_get_posterior_dir(model_name, likelihood_name) / "posterior.nc")
        for model_name in model_names
    }

    output_dir = _get_output_dir(likelihood_name)
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    comparison = build_loo_model_comparison(
        inference_data_by_model=inference_data_by_model,
        observations=model_data.observations,
        var_name=LOO_VAR_NAME,
    )

    comparison.loo_summary.to_csv(tables_dir / "loo_summary.csv", index=False)
    comparison.compare_table.to_csv(tables_dir / "loo_compare.csv", index=False)
    comparison.difference_summary.to_csv(tables_dir / "pointwise_elpd_difference_summary.csv", index=False)

    comparison.pareto_k.to_parquet(derived_dir / "pareto_k.parquet", index=False)
    comparison.pointwise_elpd.to_parquet(derived_dir / "pointwise_elpd.parquet", index=False)
    comparison.pointwise_differences.to_parquet(
        derived_dir / "pointwise_elpd_differences.parquet",
        index=False,
    )

    plot_loo_comparison(
        compare_table=comparison.compare_table,
        output_path=figures_dir / "loo_comparison.png",
    )

    for model_name, loo_result in comparison.loo_results.items():
        plot_pareto_k(
            loo_result=loo_result,
            model_name=model_name,
            output_path=figures_dir / f"pareto_k_{model_name}.png",
        )

    for row in comparison.difference_summary.itertuples(index=False):
        plot_pointwise_elpd_difference(
            frame=comparison.pointwise_differences,
            numerator_model=row.numerator_model,
            denominator_model=row.denominator_model,
            output_path=figures_dir / f"pointwise_elpd_{row.comparison}.png",
        )

    calibration_pointwise = []
    calibration_summaries = []

    for model_name in model_names:
        calibration = compute_normal_loo_pit(
            inference_data=inference_data_by_model[model_name],
            loo_result=comparison.loo_results[model_name],
            observations=model_data.observations,
            inputs=inputs,
            likelihood_name=likelihood_name,
            var_name=LOO_VAR_NAME,
        )

        pointwise = calibration.pointwise.copy()
        pointwise.insert(0, "model", model_name)
        calibration_pointwise.append(pointwise)

        summary = calibration.summary.copy()
        summary.insert(0, "model", model_name)
        calibration_summaries.append(summary)

        pit = pointwise["loo_pit"].to_numpy(dtype=float)

        plot_loo_pit_ecdf(
            loo_pit=pit,
            model_name=model_name,
            output_path=figures_dir / f"loo_pit_ecdf_{model_name}.png",
        )
        plot_loo_pit_coverage(
            loo_pit=pit,
            model_name=model_name,
            output_path=figures_dir / f"loo_pit_coverage_{model_name}.png",
        )
        plot_loo_pit_conditions(
            pointwise=pointwise,
            model_name=model_name,
            output_path=figures_dir / f"loo_pit_conditions_{model_name}.png",
        )

    loo_pit_pointwise = pd.concat(calibration_pointwise, ignore_index=True)
    loo_pit_summary = pd.concat(calibration_summaries, ignore_index=True)

    loo_pit_pointwise.to_parquet(derived_dir / "loo_pit.parquet", index=False)
    loo_pit_summary.to_csv(tables_dir / "loo_pit_summary.csv", index=False)

    print(f"\n{MATERIAL}")
    print(f"Likelihood: {likelihood_name}")
    print(f"Models: {', '.join(model_names)}")

    print("\n=== PSIS-LOO ===")
    print(comparison.loo_summary.to_string(index=False))

    print("\n=== MODEL COMPARISON ===")
    print(comparison.compare_table.to_string(index=False))

    print("\n=== POINTWISE ELPD DIFFERENCES ===")
    print(comparison.difference_summary.to_string(index=False))

    print("\n=== LOO-PIT CALIBRATION ===")
    print(loo_pit_summary.to_string(index=False))

    print(
        "\nCross-validation scope: observation-wise LOO. For the setup-intercept likelihood, "
        "other observations from the same setup remain available when one observation is held out."
    )
    print("Stacking weights are predictive weights, not posterior probabilities of mechanisms.")

    print(f"\nSaved model comparison to: {output_dir}")


if __name__ == "__main__":
    main()