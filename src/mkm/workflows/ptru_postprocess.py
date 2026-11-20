"""PtRu posterior postprocessing workflow and stable public API."""

import numpy as np
from mkm.postprocessing.drc_status import record_drc_status
from mkm.postprocessing.diagnostics import (
    build_parameter_dependence_matrices, build_physical_checks, build_posterior_parameter_summary,
)
from mkm.postprocessing.materials import (
    summarize_observation_diagnostics_by_material, summarize_residual_structure_by_material,
)
from mkm.postprocessing.model_points import summarize_model_points
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import summarize_residual_curves, summarize_shared_replicate_residuals
from mkm.postprocessing.second_order import build_second_order_outputs
from mkm.workflows.postprocessing_common import (
    compute_loo_pit_stage, compute_loo_stage, merge_material_summaries, overall_postprocessing_status,
    status_row, write_postprocessing_status, clear_postprocessing_tables, write_postprocessing_manifest,
)
from mkm.workflows.postprocessing_common import safe_plot
from mkm.postprocessing.drc_status import reset_figures_preserving_drc
from mkm.workflows._ptru_postprocess_products import (
    ptru_ion_name, ptru_electrolyte_label, load_ptru_experimental_observables,
    build_ptru_observable_comparisons,
)
from mkm.workflows._ptru_postprocess_plots import plot_ptru_products as _plot_ptru_products_impl
from mkm.workflows._ptru_postprocess_drc import postprocess_ptru_drc, postprocess_pt100_drc


def plot_ptru_products(
    run, observations, points, figures_dir, plot_level, status_rows, *,
    observable_points=None, second_order_points=None, loo=None, calibration=None,
):
    """Compatibility wrapper retaining existing plot callbacks and monkeypatch contracts."""
    return _plot_ptru_products_impl(
        run, observations, points, figures_dir, plot_level, status_rows,
        observable_points=observable_points, second_order_points=second_order_points,
        loo=loo, calibration=calibration,
        safe_plot_callback=safe_plot, reset_callback=reset_figures_preserving_drc,
    )



def postprocess_ptru_run(
    run,
    paths,
    *,
    temperature_K,
    random_seed=1,
    skip_loo=False,
    skip_observables=False,
    second_order_step_v=0.05,
    plot_level="core",
    skip_drc=True,
    drc_step_eV=1e-4,
    check_drc_half_step=False,
    save_drc_draws=False,
):
    """Generate PtRu postprocessing products with shared MKM infrastructure."""
    if plot_level not in {"none", "core", "full"}:
        raise ValueError("plot_level must be one of: none, core, full.")

    tables_dir = paths.fit_tables_dir(run.output_dir)
    figures_dir = paths.fit_figures_dir(run.output_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    clear_postprocessing_tables(tables_dir)
    options = {
        "random_seed": random_seed, "skip_loo": skip_loo,
        "skip_observables": skip_observables, "second_order_step_v": second_order_step_v,
        "plot_level": plot_level, "skip_drc": skip_drc, "drc_step_eV": drc_step_eV,
        "check_drc_half_step": check_drc_half_step, "save_drc_draws": save_drc_draws,
    }
    write_postprocessing_manifest(run.output_dir, run, options=options)

    try:
        observations = build_observation_diagnostics(
            run.inference_data, run.model_data, run.inputs,
            random_seed=random_seed,
        )
        observations.to_parquet(tables_dir / "observation_diagnostics.parquet", index=False)

        points = summarize_model_points(
            run.inference_data,
            run.model_data.model_points,
            tuple(run.built_model.mechanism_result.pointwise),
        )
        points.to_parquet(tables_dir / "model_point_diagnostics.parquet", index=False)

        physical_checks = build_physical_checks(run.inference_data.posterior)
        physical_checks.to_csv(tables_dir / "physical_checks.csv", index=False)

        parameters = build_posterior_parameter_summary(
            run.inference_data, run.parameter_specs,
        )
        parameters.to_csv(tables_dir / "posterior_parameter_diagnostics.csv", index=False)

        covariance, correlation = build_parameter_dependence_matrices(
            run.inference_data.posterior, run.free_parameter_names,
        )
        covariance.to_csv(tables_dir / "posterior_covariance_matrix.csv")
        correlation.to_csv(tables_dir / "posterior_correlation_matrix.csv")

        curve_residuals = summarize_residual_curves(observations)
        shared_residuals = summarize_shared_replicate_residuals(observations)
        curve_residuals.to_csv(tables_dir / "residual_curves.csv", index=False)
        shared_residuals.to_csv(tables_dir / "shared_replicate_residuals.csv", index=False)

        material_observations = summarize_observation_diagnostics_by_material(observations)
        material_residuals = summarize_residual_structure_by_material(
            curve_residuals, shared_residuals,
        )

        # Preserve the established PtRu diagnostic filenames.
        points.to_csv(tables_dir / "posterior_model_points.csv", index=False)
        observations.to_csv(tables_dir / "posterior_observations.csv", index=False)

        condition_summary = observations.groupby(
            ["material", "electrolyte_concentration_M", "CO_mole_fraction"], as_index=False,
        ).agg(
            n=("standardized_residual", "size"),
            mean_z=("standardized_residual", "mean"),
            rmse_z=("standardized_residual", lambda x: np.sqrt(np.mean(x**2))),
            mean_residual=("residual", "mean"),
        )
        condition_summary.to_csv(tables_dir / "posterior_condition_summary.csv", index=False)

    except Exception as error:
        status_rows.append(status_row("core", "error", f"{type(error).__name__}: {error}"))
        status_rows.append(status_row("overall", "error", "Core postprocessing failed."))
        write_postprocessing_status(tables_dir, status_rows)
        write_postprocessing_manifest(run.output_dir, run, options=options, status="error")
        raise

    status_rows.append(status_row("core", "complete"))
    write_postprocessing_status(tables_dir, status_rows)

    loo = None
    calibration = None
    loo_material = None

    if skip_loo:
        status_rows.append(status_row("loo", "skipped_user", "LOO disabled by caller."))
        status_rows.append(status_row("loo_pit", "skipped_user", "LOO disabled by caller."))
    else:
        loo, loo_material = compute_loo_stage(run, tables_dir, status_rows)
        write_postprocessing_status(tables_dir, status_rows)
        calibration = compute_loo_pit_stage(run, loo, tables_dir, status_rows)

    material_summary = merge_material_summaries(
        material_observations, material_residuals, loo_material,
    )
    material_summary.to_csv(tables_dir / "material_summary.csv", index=False)

    observable_points = None
    second_order_points = None

    if skip_observables:
        status_rows.append(status_row("observables", "skipped_user", "Observables disabled by caller."))
        status_rows.append(status_row("second_order", "skipped_user", "Observables disabled by caller."))
    else:
        try:
            observable_points, observable_summary = build_ptru_observable_comparisons(
                run, paths, temperature_K,
            )
            observable_points.to_parquet(
                tables_dir / "experimental_observable_comparisons.parquet", index=False,
            )
            observable_summary.to_csv(
                tables_dir / "experimental_observable_summary.csv", index=False,
            )
            status_rows.append(status_row("observables", "complete"))
        except Exception as error:
            status_rows.append(status_row(
                "observables", "error", f"{type(error).__name__}: {error}",
            ))

        if observable_points is None:
            status_rows.append(status_row(
                "second_order", "skipped_observables_unavailable",
                "First-order observable comparisons unavailable.",
            ))
        else:
            try:
                model_points = run.model_data.model_points
                result = build_second_order_outputs(
                    inference_data=run.inference_data,
                    model_points=model_points,
                    conditions=run.model_data.conditions,
                    observable_points=observable_points,
                    temperature_K=temperature_K,
                    koh_values=sorted(model_points["electrolyte_concentration_M"].unique()),
                    co_values=sorted(model_points["CO_mole_fraction"].unique()),
                    potential_step_V=second_order_step_v,
                    concentration_column="C_electrolyte_M",
                    ion_name=ptru_ion_name(run.specification.medium),
                )
                second_order_points = result.points
                result.points.to_csv(tables_dir / "second_order_observables.csv", index=False)
                result.summary.to_csv(tables_dir / "second_order_summary.csv", index=False)
                status_rows.append(status_row("second_order", "complete"))
            except Exception as error:
                status_rows.append(status_row(
                    "second_order", "error", f"{type(error).__name__}: {error}",
                ))

    write_postprocessing_status(tables_dir, status_rows)

    plot_ptru_products(
        run, observations, points, figures_dir, plot_level, status_rows,
        observable_points=observable_points,
        second_order_points=second_order_points,
        loo=loo,
        calibration=calibration,
    )

    if skip_drc:
        drc_dir = paths.fit_drc_dir(run.output_dir)
        drc_status = record_drc_status(
            drc_dir, recomputed=False, figures_dir=paths.fit_figures_dir(run.output_dir),
        )
        status_rows.append(status_row("drc", "skipped_user", (
            f"DRC not recomputed: {drc_status['status']}; "
            f"previous calculation timestamp: {drc_status['last_computed_at_utc'] or 'unknown'}; "
            f"see {drc_dir / 'drc_status.json'}"
        )))
    else:
        try:
            postprocess_ptru_drc(
                run, paths, temperature_K=temperature_K, step_eV=drc_step_eV,
                check_half_step=check_drc_half_step, save_draws=save_drc_draws,
                make_plots=plot_level != "none",
            )
            status_rows.append(status_row("drc", "complete"))
        except Exception as error:
            status_rows.append(status_row("drc", "error", f"{type(error).__name__}: {error}"))
    write_postprocessing_status(tables_dir, status_rows)

    overall = overall_postprocessing_status(status_rows)
    status_rows.append(status_row("overall", overall))
    write_postprocessing_status(tables_dir, status_rows)
    write_postprocessing_manifest(run.output_dir, run, options=options, status=overall)
    return overall
