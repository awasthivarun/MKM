"""Generate consolidated diagnostics for an AgPd posterior run."""

from argparse import ArgumentParser

from mkm.models.agpd_basic import available_agpd_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import load_agpd_posterior_run
from mkm.workflows.agpd_postprocess import postprocess_agpd_run


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default="Ag10Pd90")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="material")
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--skip-observables", action="store_true")
    parser.add_argument("--second-order-step-v", type=float, default=0.05)
    parser.add_argument("--plot-level", choices=("none", "core", "full"), default="core")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    specification = resolve_agpd_fit_specification(
        config,
        model_name=args.model,
        all_materials=args.all_materials,
        material=args.material,
        parameterization=args.parameterization,
        error_structure=args.error_structure,
        prior_material=args.prior_material,
    )
    run = load_agpd_posterior_run(
        paths,
        config,
        specification,
        reconstruct_pointwise=True,
        progressbar=True,
    )
    overall = postprocess_agpd_run(
        run,
        config,
        paths,
        random_seed=args.random_seed,
        skip_loo=args.skip_loo,
        skip_observables=args.skip_observables,
        second_order_step_v=args.second_order_step_v,
        plot_level=args.plot_level,
    )
    print(f"Postprocessing products saved to: {run.output_dir}")
    print(f"Postprocessing status: {overall}")


if __name__ == "__main__":
    main()
