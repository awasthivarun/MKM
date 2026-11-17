"""Generate consolidated diagnostics for an AgPd posterior run."""

from argparse import ArgumentParser

from mkm.models.agpd_basic import available_agpd_models
from mkm.project_paths import ProjectPaths
from mkm.postprocessing.drc_status import record_drc_status
from mkm.workflows.postprocessing_common import postprocessing_execution_failed
import pandas as pd
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import load_agpd_posterior_run
from mkm.workflows.agpd_postprocess import (
    postprocess_agpd_composition,
    postprocess_agpd_drc,
    postprocess_agpd_run,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default="Ag10Pd90")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument(
        "--exclude-materials",
        nargs="*",
        default=(),
        help="All-material only: load the fit that omitted these material datasets.",
    )
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--skip-observables", action="store_true")
    parser.add_argument("--second-order-step-v", type=float, default=0.05)
    parser.add_argument("--plot-level", choices=("none", "core", "full"), default="core")
    parser.add_argument("--skip-drc", action="store_true")
    parser.add_argument("--drc-step-eV", type=float, default=1e-4)
    parser.add_argument("--check-drc-half-step", action="store_true")
    parser.add_argument("--save-drc-draws", action="store_true")
    parser.add_argument("--skip-composition", action="store_true")
    parser.add_argument("--composition-n-grid", type=int, default=181)
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
        excluded_materials=args.exclude_materials,
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
    if args.skip_drc:
        drc_dir = paths.fit_drc_dir(run.output_dir)
        drc_status = record_drc_status(
            drc_dir, recomputed=False, figures_dir=paths.fit_figures_dir(run.output_dir),
        )
        print(f"DRC: {drc_status['status']} (see {drc_dir / 'drc_status.json'})")
    else:
        drc = postprocess_agpd_drc(
            run,
            config,
            paths,
            step_eV=args.drc_step_eV,
            check_half_step=args.check_drc_half_step,
            save_draws=args.save_drc_draws,
            make_plots=args.plot_level != "none",
        )
        print(drc.checks.to_string(index=False))
    if specification.is_all_materials and not args.skip_composition:
        postprocess_agpd_composition(
            run, config, paths, n_grid=args.composition_n_grid
        )

    print(f"Postprocessing products saved to: {run.output_dir}")
    print(f"Postprocessing status: {overall}")
    status = pd.read_csv(paths.fit_tables_dir(run.output_dir) / "postprocessing_status.csv")
    if postprocessing_execution_failed(status.to_dict(orient="records")):
        raise RuntimeError("AgPd postprocessing encountered an execution error; inspect status CSV.")


if __name__ == "__main__":
    main()
