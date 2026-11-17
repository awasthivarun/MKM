"""Plot effective physical parameters across Ru composition for a pooled PtRu posterior."""

from argparse import ArgumentParser

from mkm.models.ptru import available_ptru_models
from mkm.postprocessing.ptru_composition import (
    build_ptru_composition_parameter_trends, plot_ptru_composition_parameter_overview,
)
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import load_ptru_posterior_run


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--parameterization", choices=("shared", "independent", "linear_xRu"), default="linear_xRu")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--exclude-materials", nargs="*", default=())
    parser.add_argument("--n-grid", type=int, default=181)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config, priors = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, args.medium, args.model, all_materials=True, parameterization=args.parameterization,
        error_structure=args.error_structure, excluded_materials=args.exclude_materials,
    )
    run = load_ptru_posterior_run(paths, spec, reconstruct_pointwise=False, progressbar=False)
    composition_dir = paths.fit_composition_dir(run.output_dir)
    composition_dir.mkdir(parents=True, exist_ok=True)
    trends = build_ptru_composition_parameter_trends(
        run.inference_data, config, priors, spec, n_grid=args.n_grid,
    )
    table = composition_dir / "composition_parameter_trends.parquet"
    figure = composition_dir / "composition_parameter_overview.png"
    trends.to_parquet(table, index=False)
    plot_ptru_composition_parameter_overview(trends, spec, figure)
    print(f"Composition parameter trends saved to: {table}")
    print(f"Composition parameter figure saved to: {figure}")


if __name__ == "__main__":
    main()
