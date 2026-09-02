"""Compute transition-state degree-of-rate-control summaries for an AgPd run."""

from argparse import ArgumentParser

from mkm.model_inputs import build_model_point_inputs
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.drc import (
    compare_transition_state_drc_steps,
    compute_composition_transition_state_drc,
    compute_transition_state_drc,
)
from mkm.postprocessing.plotting import plot_transition_state_drc
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import load_agpd_posterior_run


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default="Ag10Pd90")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument(
        "--error-structure",
        choices=("shared", "material"),
        default="material",
    )
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument("--step-eV", type=float, default=1e-4)
    parser.add_argument("--check-half-step", action="store_true")
    parser.add_argument("--save-draws", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _compute(run, config, step_eV):
    point_inputs = build_model_point_inputs(run.inputs)
    kwargs = {
        "inference_data": run.inference_data,
        "model_name": run.specification.model_name,
        "point_inputs": point_inputs,
        "model_points": run.model_data.model_points,
        "config": config,
        "step_eV": step_eV,
    }
    if run.specification.is_all_materials:
        return compute_composition_transition_state_drc(
            **kwargs,
            parameterization=run.specification.parameterization,
        )
    return compute_transition_state_drc(**kwargs)


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
        reconstruct_pointwise=False,
        progressbar=False,
    )

    drc_dir = paths.fit_drc_dir(run.output_dir)
    drc_dir.mkdir(parents=True, exist_ok=True)

    result = _compute(run, config, args.step_eV)
    result.summary.to_parquet(
        drc_dir / "drc_transition_state.parquet",
        index=False,
    )
    result.checks.to_csv(
        drc_dir / "drc_transition_state_checks.csv",
        index=False,
    )
    if args.save_draws:
        result.draws.to_netcdf(drc_dir / "drc_transition_state_draws.nc")

    if args.check_half_step:
        half_step = _compute(run, config, 0.5 * args.step_eV)
        convergence = compare_transition_state_drc_steps(result, half_step)
        convergence.to_csv(
            drc_dir / "drc_transition_state_step_convergence.csv",
            index=False,
        )

    if not args.skip_plots:
        figures_dir = paths.fit_figures_dir(run.output_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        for material in run.inputs.materials:
            summary = result.summary.loc[result.summary["material"] == material]
            if summary.empty:
                continue
            material_dir = figures_dir / material
            material_dir.mkdir(parents=True, exist_ok=True)
            plot_transition_state_drc(
                summary,
                f"{run.specification.model_name}, {material}",
                material_dir / "drc_transition_states.png",
            )

    print(result.checks.to_string(index=False))
    print(f"DRC products saved to: {drc_dir}")


if __name__ == "__main__":
    main()
