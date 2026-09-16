"""Run ordinary all-material postprocessing on an AgPd LOCO/LOMO refit."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import postprocess_agpd_drc as drc_postprocess
import postprocess_agpd_posterior as posterior_postprocess

from mkm.inference.posterior import add_log_likelihood, compute_posterior_deterministics, load_inference_data
from mkm.model_data import build_model_data
from mkm.postprocessing.composition_parameters import (
    build_agpd_composition_parameter_trends,
    plot_agpd_composition_parameter_overview,
)
from mkm.postprocessing.diagnostics import build_physical_checks
from mkm.postprocessing.materials import (
    summarize_observation_diagnostics_by_material,
    summarize_residual_structure_by_material,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import summarize_residual_curves, summarize_shared_replicate_residuals
from mkm.project_paths import ProjectPaths
from mkm.provenance import read_run_metadata
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import all_parameter_specs, build_agpd_fit_model, resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import AgPdPosteriorRun
from mkm.workflows.agpd_validation import split_agpd_loco, split_agpd_lomo


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("model")
    parser.add_argument("scheme", choices=("loco", "lomo"))
    parser.add_argument("--holdout-material", required=True)
    parser.add_argument("--koh-M", type=float)
    parser.add_argument("--co-mole-fraction", type=float)
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="material")
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--second-order-step-v", type=float, default=0.05)
    parser.add_argument("--drc-step-eV", type=float, default=1e-4)
    parser.add_argument("--plot-level", choices=("none", "core", "full"), default="full")
    args, _ = parser.parse_known_args()
    return args


def _validation_output_dir(paths, args):
    return paths.agpd_validation_output_dir(
        scheme=args.scheme,
        model_name=args.model,
        parameterization=args.parameterization,
        error_structure=args.error_structure,
        material=args.holdout_material,
        koh_M=args.koh_M if args.scheme == "loco" else None,
        co_mole_fraction=args.co_mole_fraction if args.scheme == "loco" else None,
    )


def _build_training_model_data(paths, args):
    selected = pd.read_parquet(paths.agpd_selected_path)
    if args.scheme == "lomo":
        train, _ = split_agpd_lomo(selected, args.holdout_material)
    else:
        if args.koh_M is None or args.co_mole_fraction is None:
            raise ValueError("LOCO postprocessing requires --koh-M and --co-mole-fraction.")
        train, _ = split_agpd_loco(
            selected,
            args.holdout_material,
            args.koh_M,
            args.co_mole_fraction,
        )
    return build_model_data(train, electrolyte_concentration_column="C_KOH_M")


def _load_validation_run(paths, config, specification, output_dir, model_data):
    posterior_path = output_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Validation posterior not found: {posterior_path}")

    fit = build_agpd_fit_model(specification, model_data, config)
    built = fit.built_model
    inference_data = load_inference_data(posterior_path)

    deterministic_names = ["ln_rate_model", *tuple(built.mechanism_result.pointwise)]
    missing = [name for name in deterministic_names if name not in inference_data.posterior]
    if missing:
        posterior = compute_posterior_deterministics(
            inference_data,
            built,
            var_names=missing,
            backend="numba",
            progressbar=True,
        )
        inference_data = inference_data.copy()
        inference_data.posterior = posterior

    if not hasattr(inference_data, "log_likelihood") or "rate_observed" not in inference_data.log_likelihood:
        inference_data = add_log_likelihood(
            inference_data,
            built,
            backend="numba",
            progressbar=True,
        )

    metadata_path = output_dir / "run_metadata.yaml"
    metadata = read_run_metadata(metadata_path) if metadata_path.exists() else {}
    free_parameter_names = tuple(variable.name for variable in built.model.free_RVs)

    return AgPdPosteriorRun(
        specification=specification,
        metadata=metadata,
        output_dir=output_dir,
        model_data=model_data,
        inputs=fit.inputs,
        built_model=built,
        inference_data=inference_data,
        parameter_specs=all_parameter_specs(specification, config),
        free_parameter_names=free_parameter_names,
    )


def _postprocess_standard_fit(run, config, paths, root, args):
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []

    observation_diagnostics = build_observation_diagnostics(
        run.inference_data,
        run.model_data,
        run.inputs,
        random_seed=args.random_seed,
    )
    observation_diagnostics.to_parquet(tables_dir / "observation_diagnostics.parquet", index=False)

    model_point_summary = posterior_postprocess._summarize_model_points(run)
    model_point_summary.to_parquet(tables_dir / "model_point_diagnostics.parquet", index=False)

    physical_checks = build_physical_checks(run.inference_data.posterior)
    physical_checks.to_csv(tables_dir / "physical_checks.csv", index=False)

    curve_residuals = summarize_residual_curves(observation_diagnostics)
    shared_residuals = summarize_shared_replicate_residuals(observation_diagnostics)
    material_observation = summarize_observation_diagnostics_by_material(observation_diagnostics)
    material_residual = summarize_residual_structure_by_material(curve_residuals, shared_residuals)

    status_rows.append(posterior_postprocess._status_row("core", "complete"))
    posterior_postprocess._write_postprocessing_status(tables_dir, status_rows)

    loo, loo_material = posterior_postprocess._compute_loo_stage(run, tables_dir, status_rows)
    posterior_postprocess._write_postprocessing_status(tables_dir, status_rows)
    calibration = posterior_postprocess._compute_loo_pit_stage(run, loo, tables_dir, status_rows)
    posterior_postprocess._write_postprocessing_status(tables_dir, status_rows)

    material_summary = posterior_postprocess._merge_material_summaries(
        material_observation,
        material_residual,
        loo_material,
    )
    material_summary.to_csv(tables_dir / "material_summary.csv", index=False)

    observable_points = posterior_postprocess._build_observables_stage(
        run,
        config,
        paths,
        tables_dir,
        status_rows,
    )
    second_order_points = posterior_postprocess._build_second_order_stage(
        run,
        config,
        paths,
        tables_dir,
        status_rows,
        observable_points,
        args.second_order_step_v,
    )
    posterior_postprocess._write_postprocessing_status(tables_dir, status_rows)

    posterior_postprocess._make_plots(
        run,
        config,
        observation_diagnostics,
        model_point_summary,
        figures_dir,
        args.plot_level,
        status_rows=status_rows,
        loo=loo,
        calibration=calibration,
        observable_points=observable_points,
        second_order_points=second_order_points,
    )

    overall = posterior_postprocess._overall_postprocessing_status(status_rows)
    status_rows.append(posterior_postprocess._status_row("overall", overall))
    posterior_postprocess._write_postprocessing_status(tables_dir, status_rows)
    return overall


def _postprocess_drc(run, config, root, args):
    drc_dir = root / "drc"
    drc_dir.mkdir(parents=True, exist_ok=True)

    result = drc_postprocess._compute(run, config, args.drc_step_eV)
    result.summary.to_parquet(drc_dir / "drc_transition_state.parquet", index=False)
    result.checks.to_csv(drc_dir / "drc_transition_state_checks.csv", index=False)

    half_step = drc_postprocess._compute(run, config, 0.5 * args.drc_step_eV)
    convergence = drc_postprocess.compare_transition_state_drc_steps(result, half_step)
    convergence.to_csv(drc_dir / "drc_transition_state_step_convergence.csv", index=False)

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for material in run.inputs.materials:
        summary = result.summary.loc[result.summary["material"] == material]
        if summary.empty:
            continue
        material_dir = figures_dir / material
        material_dir.mkdir(parents=True, exist_ok=True)
        drc_postprocess.plot_transition_state_drc(
            summary,
            f"{run.specification.model_name}, {material}",
            material_dir / "drc_transition_states.png",
        )


def _postprocess_composition(run, config, root):
    if not run.specification.is_all_materials or run.specification.parameterization == "shared":
        return

    composition_dir = root / "composition"
    composition_dir.mkdir(parents=True, exist_ok=True)
    trends = build_agpd_composition_parameter_trends(
        run.inference_data,
        config,
        model_name=run.specification.model_name,
        parameterization=run.specification.parameterization,
        n_grid=181,
    )
    trends.to_parquet(composition_dir / "composition_parameter_trends.parquet", index=False)
    plot_agpd_composition_parameter_overview(
        trends,
        config,
        model_name=run.specification.model_name,
        parameterization=run.specification.parameterization,
        output_path=composition_dir / "composition_parameter_overview.png",
    )


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

    output_dir = _validation_output_dir(paths, args)
    model_data = _build_training_model_data(paths, args)
    run = _load_validation_run(paths, config, specification, output_dir, model_data)

    standard_root = output_dir / "postprocessing"
    standard_root.mkdir(parents=True, exist_ok=True)

    overall = _postprocess_standard_fit(run, config, paths, standard_root, args)
    _postprocess_drc(run, config, standard_root, args)
    _postprocess_composition(run, config, standard_root)

    print(f"Validation refit postprocessing saved to: {standard_root}")
    print(f"Standard postprocessing status: {overall}")


if __name__ == "__main__":
    main()
