"""Compute transition-state DRC for individual or pooled PtRu in acidic or basic electrolyte."""

from argparse import ArgumentParser
from mkm.models.ptru import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import load_ptru_posterior_run
from mkm.workflows.ptru_postprocess import postprocess_ptru_drc


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--material", default="Pt100")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", choices=("shared", "independent", "linear_xRu"), default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--exclude-materials", nargs="*", default=())
    parser.add_argument("--step-eV", type=float, default=1e-4)
    parser.add_argument("--check-half-step", action="store_true")
    parser.add_argument("--save-draws", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, args.medium, args.model, material=args.material,
        all_materials=args.all_materials, parameterization=args.parameterization,
        error_structure=args.error_structure, excluded_materials=args.exclude_materials,
    )
    run = load_ptru_posterior_run(paths, spec, progressbar=False)
    result = postprocess_ptru_drc(
        run, paths, temperature_K=config["temperature_K"], step_eV=args.step_eV,
        check_half_step=args.check_half_step, save_draws=args.save_draws,
        make_plots=not args.skip_plots,
    )
    print(result.checks.to_string(index=False))
    print(f"DRC products saved to: {paths.fit_drc_dir(run.output_dir)}")


if __name__ == "__main__":
    main()
