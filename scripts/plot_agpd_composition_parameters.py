"""Plot effective physical-parameter trends from an all-material AgPd posterior."""

from argparse import ArgumentParser

from mkm.models.agpd_basic import available_agpd_all_material_models
from mkm.postprocessing.composition_parameters import (
    build_agpd_composition_parameter_trends,
    plot_agpd_composition_parameter_overview,
)
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import load_agpd_posterior_run


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_all_material_models())
    parser.add_argument("--parameterization", default="linear_xAg")
    parser.add_argument(
        "--error-structure",
        choices=("shared", "material"),
        default="material",
    )
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument("--n-grid", type=int, default=181)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    specification = resolve_agpd_fit_specification(
        config,
        model_name=args.model,
        all_materials=True,
        parameterization=args.parameterization,
        error_structure=args.error_structure,
        prior_material=args.prior_material,
    )
    run = load_agpd_posterior_run(
        paths,
        config,
        specification,
        reconstruct_pointwise=False,
        progressbar=False,
    )

    trends = build_agpd_composition_parameter_trends(
        run.inference_data,
        config,
        model_name=specification.model_name,
        parameterization=specification.parameterization,
        n_grid=args.n_grid,
    )
    trends_path = run.output_dir / "composition_parameter_trends.parquet"
    trends.to_parquet(trends_path, index=False)

    figure_path = run.output_dir / "figures" / "composition_parameter_overview.png"
    plot_agpd_composition_parameter_overview(
        trends,
        config,
        model_name=specification.model_name,
        parameterization=specification.parameterization,
        output_path=figure_path,
    )

    print(f"Composition parameter trends saved to: {trends_path}")
    print(f"Composition parameter figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
