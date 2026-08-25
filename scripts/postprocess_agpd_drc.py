"""Compute and persist transition-state DRCs for an AgPd posterior fit."""

from argparse import ArgumentParser

import arviz as az
import yaml

from mkm.model_inputs import build_model_point_inputs
from mkm.models.agpd_basic import (
    available_agpd_composition_parameterizations,
    available_agpd_models,
)
from mkm.postprocessing.drc import (
    compare_transition_state_drc_steps,
    compute_composition_transition_state_drc,
    compute_transition_state_drc,
)
from mkm.postprocessing.plotting import plot_transition_state_drc
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import (
    build_agpd_composition_model_data,
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
    parser.add_argument(
        "--composition-model",
        choices=available_agpd_composition_parameterizations(),
        help="Use a saved multi-material composition posterior instead of an individual-material posterior.",
    )
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="iid")
    parser.add_argument("--step-eV", type=float, default=1e-4)
    parser.add_argument("--check-half-step", action="store_true")
    return parser.parse_args()


def _load_composition_run_metadata(posterior_dir):
    path = posterior_dir / "run_metadata.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Composition posterior metadata not found: {path}")

    with open(path, "r") as file:
        metadata = yaml.safe_load(file)

    if not isinstance(metadata, dict) or "materials" not in metadata:
        raise ValueError(f"Invalid composition posterior metadata: {path}")
    return metadata


def main():
    args = parse_args()
    material = args.material

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)

    if args.composition_model is None:
        validate_agpd_material(config, material)
        posterior_dir = paths.agpd_posterior_dir(material, args.model, args.likelihood)
        model_data = build_agpd_model_data(paths, material)
        compute_drc = compute_transition_state_drc
        compute_kwargs = {}
        materials = (material,)
    else:
        posterior_dir = paths.agpd_composition_posterior_output_dir(
            composition_model=args.composition_model,
            model_name=args.model,
            likelihood_name=args.likelihood,
        )
        metadata = _load_composition_run_metadata(posterior_dir)
        materials = tuple(metadata["materials"])
        model_data = build_agpd_composition_model_data(paths, materials)
        compute_drc = compute_composition_transition_state_drc
        compute_kwargs = {"composition_model": args.composition_model}

    output_dir = posterior_dir / "postprocessing"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    inputs = build_agpd_inputs(model_data, config, args.likelihood)
    point_inputs = build_model_point_inputs(inputs)
    idata = az.from_netcdf(posterior_dir / "posterior.nc")

    result = compute_drc(
        inference_data=idata,
        model_name=args.model,
        point_inputs=point_inputs,
        model_points=model_data.model_points,
        config=config,
        step_eV=args.step_eV,
        **compute_kwargs,
    )

    result.draws.to_netcdf(derived_dir / "drc_transition_state_draws.nc")
    result.summary.to_parquet(derived_dir / "drc_transition_state.parquet", index=False)
    result.checks.to_csv(tables_dir / "drc_transition_state_checks.csv", index=False)

    if args.composition_model is None:
        plot_transition_state_drc(
            summary=result.summary,
            model_name=args.model,
            output_path=figures_dir / "drc_transition_states.png",
        )
    else:
        by_material_dir = figures_dir / "drc_transition_states_by_material"
        by_material_dir.mkdir(parents=True, exist_ok=True)

        for current_material in materials:
            material_summary = result.summary.loc[result.summary["material"] == current_material].copy()
            if material_summary.empty:
                continue
            plot_transition_state_drc(
                summary=material_summary,
                model_name=f"{args.model}, {current_material}",
                output_path=by_material_dir / f"{current_material}.png",
            )

    if args.composition_model is None:
        print(f"\n{material}: {args.model}")
    else:
        print(f"\nAgPd composition DRC: {args.model}")
        print(f"Composition parameterization: {args.composition_model}")
        print(f"Materials: {', '.join(materials)}")
    print(f"Likelihood: {args.likelihood}")
    print("\n=== TRANSITION-STATE DRC CHECKS ===")
    print(result.checks.to_string(index=False))

    if args.check_half_step:
        half_step = compute_drc(
            inference_data=idata,
            model_name=args.model,
            point_inputs=point_inputs,
            model_points=model_data.model_points,
            config=config,
            step_eV=0.5 * args.step_eV,
            **compute_kwargs,
        )
        convergence = compare_transition_state_drc_steps(result, half_step)
        convergence.to_csv(tables_dir / "drc_transition_state_step_convergence.csv", index=False)

        print("\n=== FINITE-DIFFERENCE STEP CONVERGENCE ===")
        print(convergence.to_string(index=False))

    print(f"\nSaved transition-state DRC products to: {output_dir}")


if __name__ == "__main__":
    main()
