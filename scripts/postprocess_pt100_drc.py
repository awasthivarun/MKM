"""Compute Pt100 transition-state DRC in acidic or basic electrolyte."""

from argparse import ArgumentParser
from mkm.models.ptru_registry import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.pt100_fit import load_pt100_config
from mkm.workflows.ptru_posterior import load_pt100_posterior_run
from mkm.workflows.ptru_postprocess import postprocess_pt100_drc


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--step-eV", type=float, default=1e-4)
    parser.add_argument("--check-half-step", action="store_true")
    parser.add_argument("--save-draws", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    paths = ProjectPaths.discover(__file__)
    config, _ = load_pt100_config(paths)
    run = load_pt100_posterior_run(paths, args.medium, args.model, progressbar=False)
    result = postprocess_pt100_drc(
        run, paths, temperature_K=config["temperature_K"], step_eV=args.step_eV,
        check_half_step=args.check_half_step, save_draws=args.save_draws,
        make_plots=not args.skip_plots,
    )
    print(result.checks.to_string(index=False))
    print(f"DRC products saved to: {paths.fit_drc_dir(run.output_dir)}")


if __name__ == "__main__":
    main()
