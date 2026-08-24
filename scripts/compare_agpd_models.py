"""Compare fitted AgPd microkinetic models using PSIS-LOO."""

from argparse import ArgumentParser

import arviz as az

from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.model_comparison import build_loo_model_comparison
from mkm.postprocessing.plotting import plot_loo_comparison, plot_pointwise_elpd_difference
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import build_agpd_model_data, load_agpd_model_config, validate_agpd_material


DEFAULT_MATERIAL = "Ag10Pd90"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="setup_intercept")
    parser.add_argument("--models", nargs="+", choices=available_agpd_models(), default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    material = args.material
    likelihood_name = args.likelihood
    model_names = args.models or list(available_agpd_models())

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    validate_agpd_material(config, material)

    model_data = build_agpd_model_data(paths, material)

    inference_data_by_model = {
        model_name: az.from_netcdf(
            paths.agpd_posterior_dir(material, model_name, likelihood_name) / "posterior.nc"
        )
        for model_name in model_names
    }

    output_dir = paths.agpd_model_comparison_dir(material, likelihood_name)
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

    print(f"\n{material}")
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