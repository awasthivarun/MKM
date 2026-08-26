"""Plot posterior AgPd mechanism parameters as functions of Ag composition."""

from argparse import ArgumentParser

import arviz as az

from mkm.models.agpd_basic import available_agpd_composition_models, available_agpd_composition_parameterizations
from mkm.postprocessing.composition_parameters import (
    build_agpd_composition_parameter_trends,
    build_agpd_material_noise_summary,
    plot_agpd_composition_parameter_overview,
)
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config


DEFAULT_MODEL = "CO_BF_ER_LH"
DEFAULT_COMPOSITION_MODEL = "linear_xAg"
DEFAULT_LIKELIHOOD = "iid"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", nargs="?", choices=available_agpd_composition_models(), default=DEFAULT_MODEL)
    parser.add_argument(
        "--composition-model",
        choices=available_agpd_composition_parameterizations(),
        default=DEFAULT_COMPOSITION_MODEL,
    )
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept", "mvn"], default=DEFAULT_LIKELIHOOD)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)

    posterior_dir = paths.agpd_composition_posterior_output_dir(
        composition_model=args.composition_model,
        model_name=args.model,
        likelihood_name=args.likelihood,
    )
    posterior_path = posterior_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    idata = az.from_netcdf(posterior_path)
    trends = build_agpd_composition_parameter_trends(
        inference_data=idata,
        config=config,
        model_name=args.model,
        composition_model=args.composition_model,
    )
    noise = build_agpd_material_noise_summary(idata, config)

    postprocessing_dir = posterior_dir / "postprocessing"
    tables_dir = postprocessing_dir / "tables"
    figures_dir = postprocessing_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    trends_path = tables_dir / "composition_parameter_trends.csv"
    noise_path = tables_dir / "composition_noise_by_material.csv"
    figure_path = figures_dir / "composition_parameter_trends.png"

    trends.to_csv(trends_path, index=False)
    noise.to_csv(noise_path, index=False)
    plot_agpd_composition_parameter_overview(
        trends=trends,
        noise_summary=noise,
        config=config,
        model_name=args.model,
        composition_model=args.composition_model,
        output_path=figure_path,
    )

    print(f"\nAgPd composition parameter plot: {args.model}")
    print(f"Composition parameterization: {args.composition_model}")
    print(f"Likelihood: {args.likelihood}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved parameter trends: {trends_path}")
    if not noise.empty:
        print(f"Saved material residual scales: {noise_path}")


if __name__ == "__main__":
    main()
