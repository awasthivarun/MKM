"""Top-level orchestration for AgPd posterior postprocessing."""

from mkm.postprocessing.diagnostics import build_physical_checks
from mkm.postprocessing.materials import (
    summarize_observation_diagnostics_by_material,
    summarize_residual_structure_by_material,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import summarize_residual_curves, summarize_shared_replicate_residuals
from .plots import _make_plots
from .products import (
    _build_observables_stage,
    _build_second_order_stage,
    _compute_loo_pit_stage,
    _compute_loo_stage,
    _merge_material_summaries,
    _overall_postprocessing_status,
    _summarize_model_points,
)
from .status import status_row, write_postprocessing_status
def postprocess_agpd_run(
    run,
    config,
    paths,
    *,
    random_seed=1,
    skip_loo=False,
    skip_observables=False,
    second_order_step_v=0.05,
    plot_level="core",
):
    """Generate the complete postprocessing product set for one loaded AgPd posterior run."""
    if plot_level not in {"none", "core", "full"}:
        raise ValueError("plot_level must be one of: none, core, full.")

    tables_dir = paths.fit_tables_dir(run.output_dir)
    figures_dir = paths.fit_figures_dir(run.output_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []

    try:
        observation_diagnostics = build_observation_diagnostics(
            run.inference_data,
            run.model_data,
            run.inputs,
            random_seed=random_seed,
        )
        observation_diagnostics.to_parquet(tables_dir / "observation_diagnostics.parquet", index=False)

        model_point_summary = _summarize_model_points(run)
        model_point_summary.to_parquet(tables_dir / "model_point_diagnostics.parquet", index=False)

        physical_checks = build_physical_checks(run.inference_data.posterior)
        physical_checks.to_csv(tables_dir / "physical_checks.csv", index=False)

        curve_residuals = summarize_residual_curves(observation_diagnostics)
        shared_residuals = summarize_shared_replicate_residuals(observation_diagnostics)
        material_observation = summarize_observation_diagnostics_by_material(observation_diagnostics)
        material_residual = summarize_residual_structure_by_material(curve_residuals, shared_residuals)
    except Exception as error:
        status_rows.append(status_row("core", "error", f"{type(error).__name__}: {error}"))
        status_rows.append(status_row("overall", "error", "Core postprocessing failed."))
        write_postprocessing_status(tables_dir, status_rows)
        raise

    status_rows.append(status_row("core", "complete"))
    write_postprocessing_status(tables_dir, status_rows)

    loo = None
    calibration = None
    loo_material = None
    if skip_loo:
        status_rows.append(status_row("loo", "skipped_user", "LOO was disabled by the caller."))
        status_rows.append(status_row("loo_pit", "skipped_user", "LOO was disabled by the caller."))
    else:
        loo, loo_material = _compute_loo_stage(run, tables_dir, status_rows)
        write_postprocessing_status(tables_dir, status_rows)
        calibration = _compute_loo_pit_stage(run, loo, tables_dir, status_rows)
    write_postprocessing_status(tables_dir, status_rows)

    material_summary = _merge_material_summaries(material_observation, material_residual, loo_material)
    material_summary.to_csv(tables_dir / "material_summary.csv", index=False)

    observable_points = None
    second_order_points = None
    if skip_observables:
        status_rows.append(status_row("observables", "skipped_user", "Observable postprocessing was disabled."))
        status_rows.append(status_row("second_order", "skipped_user", "Observable postprocessing was disabled."))
    else:
        observable_points = _build_observables_stage(run, config, paths, tables_dir, status_rows)
        second_order_points = _build_second_order_stage(
            run,
            config,
            paths,
            tables_dir,
            status_rows,
            observable_points,
            second_order_step_v,
        )
    write_postprocessing_status(tables_dir, status_rows)

    _make_plots(
        run,
        config,
        observation_diagnostics,
        model_point_summary,
        figures_dir,
        plot_level,
        status_rows=status_rows,
        loo=loo,
        calibration=calibration,
        observable_points=observable_points,
        second_order_points=second_order_points,
    )

    overall = _overall_postprocessing_status(status_rows)
    status_rows.append(status_row("overall", overall))
    write_postprocessing_status(tables_dir, status_rows)
    return overall
