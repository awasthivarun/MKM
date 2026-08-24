"""Compute and persist transition-state DRCs for an AgPd posterior fit."""

from argparse import ArgumentParser

import arviz as az

from mkm.model_inputs import build_model_point_inputs
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.drc import compare_transition_state_drc_steps, compute_transition_state_drc
from mkm.postprocessing.plotting import plot_transition_state_drc
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import (
    build_agpd_inputs,
    build_agpd_model_data,
    load_agpd_model_config,
    validate_agpd_material,
)


DEFAULT_MATERIAL = "Ag10Pd90"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="iid")
    parser.add_argument("--step-eV", type=float, default=1e-4)
    parser.add_argument("--check-half-step", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    material = args.material

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    validate_agpd_material(config, material)

    posterior_dir = paths.agpd_posterior_dir(material, args.model, args.likelihood)
    output_dir = posterior_dir / "postprocessing"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    model_data = build_agpd_model_data(paths, material)
    inputs = build_agpd_inputs(model_data, config, args.likelihood)
    point_inputs = build_model_point_inputs(inputs)
    idata = az.from_netcdf(posterior_dir / "posterior.nc")

    result = compute_transition_state_drc(
        inference_data=idata,
        model_name=args.model,
        point_inputs=point_inputs,
        model_points=model_data.model_points,
        config=config,
        step_eV=args.step_eV,
    )

    result.draws.to_netcdf(derived_dir / "drc_transition_state_draws.nc")
    result.summary.to_parquet(derived_dir / "drc_transition_state.parquet", index=False)
    result.checks.to_csv(tables_dir / "drc_transition_state_checks.csv", index=False)

    plot_transition_state_drc(
        summary=result.summary,
        model_name=args.model,
        output_path=figures_dir / "drc_transition_states.png",
    )

    print(f"\n{material}: {args.model}")
    print(f"Likelihood: {args.likelihood}")
    print("\n=== TRANSITION-STATE DRC CHECKS ===")
    print(result.checks.to_string(index=False))

    if args.check_half_step:
        half_step = compute_transition_state_drc(
            inference_data=idata,
            model_name=args.model,
            point_inputs=point_inputs,
            model_points=model_data.model_points,
            config=config,
            step_eV=0.5 * args.step_eV,
        )
        convergence = compare_transition_state_drc_steps(result, half_step)
        convergence.to_csv(tables_dir / "drc_transition_state_step_convergence.csv", index=False)

        print("\n=== FINITE-DIFFERENCE STEP CONVERGENCE ===")
        print(convergence.to_string(index=False))

    print(f"\nSaved transition-state DRC products to: {output_dir}")


if __name__ == "__main__":
    main()
