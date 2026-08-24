"""Compare fitted AgPd microkinetic models using PSIS-LOO."""

from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import pandas as pd

from mkm.model_data import build_model_data
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.model_comparison import build_loo_model_comparison
from mkm.postprocessing.plotting import plot_loo_comparison, plot_pointwise_elpd_difference


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
DATA_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_selected.parquet"
POSTERIOR_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"


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


def main():
    args = parse_args()
    likelihood_name = args.likelihood
    model_names = args.models or list(available_agpd_models())

    selected = pd.read_parquet(DATA_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()
    model_data = build_model_data(selected, electrolyte_concentration_column="C_KOH_M")

    inference_data_by_model = {
        model_name: az.from_netcdf(_get_posterior_dir(model_name, likelihood_name) / "posterior.nc")
        for model_name in model_names
    }

    output_dir = POSTERIOR_ROOT / MATERIAL / likelihood_name / "model_comparison"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    comparison = build_loo_model_comparison(
        inference_data_by_model=inference_data_by_model,
        observations=model_data.observations,
        var_name="ln_rate_observed",
    )

    comparison.compare_table.to_csv(tables_dir / "loo_compare.csv", index=False)
    comparison.difference_summary.to_csv(tables_dir / "pointwise_elpd_difference_summary.csv", index=False)
    comparison.pointwise_differences.to_parquet(
        derived_dir / "pointwise_elpd_differences.parquet",
        index=False,
    )

    plot_loo_comparison(comparison.compare_table, figures_dir / "loo_comparison.png")

    for row in comparison.difference_summary.itertuples(index=False):
        plot_pointwise_elpd_difference(
            frame=comparison.pointwise_differences,
            numerator_model=row.numerator_model,
            denominator_model=row.denominator_model,
            output_path=figures_dir / f"pointwise_elpd_{row.comparison}.png",
        )

    print(f"\n{MATERIAL}")
    print(f"Likelihood: {likelihood_name}")
    print(f"Models: {', '.join(model_names)}")

    print("\n=== MODEL COMPARISON ===")
    print(comparison.compare_table.to_string(index=False))

    print("\n=== POINTWISE ELPD DIFFERENCES ===")
    print(comparison.difference_summary.to_string(index=False))

    print(
        "\nCross-validation scope: observation-wise LOO. For the setup-intercept likelihood, "
        "other observations from the same setup remain available when one observation is held out."
    )
    print("Stacking weights are predictive weights, not posterior probabilities of mechanisms.")
    print(f"\nSaved model comparison to: {output_dir}")


if __name__ == "__main__":
    main()