"""Postprocess PtRu individual or all-material posteriors in one electrolyte medium."""

from argparse import ArgumentParser

import pandas as pd

from mkm.models.ptru import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.postprocessing_common import postprocessing_execution_failed
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import load_ptru_posterior_run
from mkm.workflows.ptru_postprocess import postprocess_ptru_run


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--material", default="Pt100")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", choices=("shared", "independent", "linear_xRu"), default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--exclude-materials", nargs="*", default=())
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--skip-observables", action="store_true")
    parser.add_argument("--second-order-step-v", type=float, default=0.05)
    parser.add_argument("--plot-level", choices=("none", "core", "full"), default="core")
    parser.add_argument("--skip-drc", action="store_true")
    parser.add_argument("--drc-step-eV", type=float, default=1e-4)
    parser.add_argument("--check-drc-half-step", action="store_true")
    parser.add_argument("--save-drc-draws", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, args.medium, args.model, material=args.material,
        all_materials=args.all_materials, parameterization=args.parameterization,
        error_structure=args.error_structure, excluded_materials=args.exclude_materials,
    )
    run = load_ptru_posterior_run(paths, spec, reconstruct_pointwise=True, progressbar=True)
    status = postprocess_ptru_run(
        run, paths, temperature_K=config["temperature_K"], random_seed=args.random_seed,
        skip_loo=args.skip_loo, skip_observables=args.skip_observables,
        second_order_step_v=args.second_order_step_v, plot_level=args.plot_level,
        skip_drc=args.skip_drc, drc_step_eV=args.drc_step_eV,
        check_drc_half_step=args.check_drc_half_step, save_drc_draws=args.save_drc_draws,
    )
    print(f"PtRu posterior: {args.medium} / {args.model} / {spec.fit_scope}")
    print(f"Status: {status}; output: {run.output_dir}")
    diagnostics = pd.read_csv(paths.fit_tables_dir(run.output_dir) / "postprocessing_status.csv")
    if postprocessing_execution_failed(diagnostics.to_dict(orient="records")):
        raise RuntimeError("PtRu postprocessing failed; inspect postprocessing_status.csv.")


if __name__ == "__main__":
    main()
