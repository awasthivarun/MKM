
"""Posterior postprocessing for Pt100 in acidic and basic electrolyte."""

import numpy as np
import pandas as pd

from mkm.observable_maps import (
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
)
from mkm.postprocessing.diagnostics import (
    build_parameter_dependence_matrices,
    build_physical_checks,
    build_posterior_parameter_summary,
)
from mkm.postprocessing.materials import (
    summarize_observation_diagnostics_by_material,
    summarize_residual_structure_by_material,
)
from mkm.postprocessing.model_points import summarize_model_points
from mkm.postprocessing.observable_comparison import (
    build_experimental_observable_comparison,
    summarize_experimental_observable,
)
from mkm.postprocessing.observables import summarize_posterior_linear_observable
from mkm.postprocessing.plotting import (
    COVERAGE_COLORS,
    PATHWAY_COLORS,
    POINTWISE_LABELS,
    plot_alpha_comparison,
    plot_delta_co_comparison,
    plot_delta_oh_comparison,
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_variables,
    plot_sampling_correlations,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
    plot_second_order_difference,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import (
    summarize_residual_curves,
    summarize_shared_replicate_residuals,
)
from mkm.postprocessing.second_order import build_second_order_outputs
from mkm.workflows.agpd_postprocess.products import (
    _compute_loo_pit_stage,
    _compute_loo_stage,
    _merge_material_summaries,
    _overall_postprocessing_status,
)
from mkm.workflows.agpd_postprocess.status import (
    safe_plot,
    status_row,
    write_postprocessing_status,
)


def ptru_ion_name(medium):
    if medium == "acidic":
        return "delta_Hplus"
    if medium == "basic":
        return "delta_OH"
    raise ValueError(f"Unsupported PtRu medium: {medium}")


def ptru_electrolyte_label(medium):
    return "HClO4" if medium == "acidic" else "KOH"



def load_ptru_experimental_observables(paths, medium):
    """Load only selected Todd observables for the fitted Pt100 dataset."""
    ion_name = ptru_ion_name(medium)
    names = {
        "alpha": "alpha_summary",
        ion_name: ion_name,
        "delta_CO": "delta_CO",
    }
    frames = {}

    for observable, product in names.items():
        path = paths.ptru_observable_path(medium, product, selected=True)
        data = pd.read_parquet(path)
        data = data.loc[data["material"] == "Pt100"].copy()

        if "medium" in data.columns and not data["medium"].eq(medium).all():
            raise ValueError(f"Experimental {observable} table contains another medium.")

        if data.empty:
            raise ValueError(f"No selected Pt100 {observable} observations found: {path}")

        frames[observable] = data

    return frames


def build_ptru_observable_comparisons(run, paths, temperature_K):
    """Build Pt100 posterior/experimental comparisons using common observable maps."""
    medium = run.specification.medium
    ion_name = ptru_ion_name(medium)
    model_points = run.model_data.model_points
    conditions = run.model_data.conditions

    concentrations = sorted(model_points["electrolyte_concentration_M"].unique())
    co_fractions = sorted(model_points["CO_mole_fraction"].unique())

    alpha_map = build_alpha_map(model_points, temperature_K)
    ion_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=concentrations,
        group_columns=["material", "CO_mole_fraction"],
    )
    co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=co_fractions,
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    alpha = summarize_posterior_linear_observable(run.inference_data, alpha_map)
    ion = summarize_posterior_linear_observable(run.inference_data, ion_map)
    co = summarize_posterior_linear_observable(run.inference_data, co_map)

    condition_metadata = conditions[
        ["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]
    ].rename(columns={"electrolyte_concentration_M": "C_electrolyte_M"})

    alpha_pooled = alpha.pooled.merge(
        condition_metadata, on="condition_id", how="left", validate="many_to_one",
    )
    alpha_chain = alpha.by_chain.merge(
        condition_metadata, on="condition_id", how="left", validate="many_to_one",
    )

    rename_co = {
        "electrolyte_concentration_M": "C_electrolyte_M",
        "lower_CO_mole_fraction": "CO_lower_mole_fraction",
        "upper_CO_mole_fraction": "CO_upper_mole_fraction",
    }
    experimental = load_ptru_experimental_observables(paths, medium)

    comparison_specs = {
        "alpha": (
            alpha_pooled, alpha_chain,
            ["material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index"],
            "alpha_mean", "alpha_sd",
        ),
        ion_name: (
            ion.pooled, ion.by_chain,
            ["material", "CO_mole_fraction", "analysis_grid_index"],
            ion_name, f"{ion_name}_sd",
        ),
        "delta_CO": (
            co.pooled.rename(columns=rename_co),
            co.by_chain.rename(columns=rename_co),
            [
                "material", "C_electrolyte_M", "CO_lower_mole_fraction",
                "CO_upper_mole_fraction", "analysis_grid_index",
            ],
            "delta_CO", "delta_CO_sd",
        ),
    }

    point_frames = []
    summary_rows = []

    for name, (pooled, by_chain, keys, value_column, sd_column) in comparison_specs.items():
        measured = experimental[name].copy()

        if sd_column not in measured:
            measured[sd_column] = np.nan

        comparison = build_experimental_observable_comparison(
            pooled=pooled,
            by_chain=by_chain,
            experimental=measured,
            key_columns=keys,
            observed_column=value_column,
            observed_sd_column=sd_column,
        )

        frame = comparison.pooled.copy()
        frame.insert(0, "observable", name)
        point_frames.append(frame)
        summary_rows.append(summarize_experimental_observable(name, comparison))

    return (
        pd.concat(point_frames, ignore_index=True, sort=False),
        pd.DataFrame(summary_rows),
    )


def plot_ptru_products(
    run,
    observations,
    points,
    figures_dir,
    plot_level,
    status_rows,
    *,
    observable_points=None,
    second_order_points=None,
):
    """Use the existing shared plots with PtRu-specific presentation parameters."""
    if plot_level == "none":
        status_rows.append(status_row("plots", "skipped_user", "Plot level is none."))
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    attempts = 0
    failures = 0

    def plot(name, function, *args, **kwargs):
        nonlocal attempts, failures
        attempts += 1
        if not safe_plot(status_rows, name, function, *args, **kwargs):
            failures += 1

    medium = run.specification.medium
    electrolyte_label = ptru_electrolyte_label(medium)
    ion_name = ptru_ion_name(medium)
    ion_label = "H+" if medium == "acidic" else "OH"
    context = f"Pt100 / {medium}"

    plot(
        "posterior_parameters", plot_parameter_posteriors,
        run.inference_data.posterior, run.parameter_specs,
        figures_dir / "posterior_parameters.png",
    )
    
    plot(
        "posterior_correlations", plot_sampling_correlations, 
        run.inference_data, run.free_parameter_names,
        figures_dir / "posterior_correlations.png",
    )

    material_dir = figures_dir / "Pt100"
    material_dir.mkdir(parents=True, exist_ok=True)

    plot(
        "Pt100/rates_model", plot_observation_grid,
        observations, material_dir / "rates_model.png",
        distribution="model", context_label=context,
        electrolyte_label=electrolyte_label,
    )
    plot(
        "Pt100/rates_predictive", plot_observation_grid,
        observations, material_dir / "rates_predictive.png",
        distribution="predictive", y_scale="linear", context_label=context,
        electrolyte_label=electrolyte_label,
    )
    plot(
        "Pt100/residuals", plot_observation_grid,
        observations, material_dir / "residuals.png",
        residual=True, y_scale="linear", context_label=context,
        electrolyte_label=electrolyte_label,
    )

    if plot_level == "full":
        plot(
            "Pt100/coverages", plot_pointwise_variables,
            points, ("theta_empty", "theta_CO", "theta_OH"),
            material_dir / "coverages.png",
            title="Pt site coverages", ylabel="Pt-site coverage",
            colors=COVERAGE_COLORS, labels=POINTWISE_LABELS,
            context_label=context, electrolyte_label=electrolyte_label,
        )

        pathways = tuple(
            name for name in ("rate_fraction_ER", "rate_fraction_LH")
            if f"{name}_median" in points
        )
        if pathways:
            pathway_colors = {
                "rate_fraction_ER": PATHWAY_COLORS["ER"],
                "rate_fraction_LH": PATHWAY_COLORS["LH"],
            }
            plot(
                "Pt100/rate_fractions", plot_pointwise_variables,
                points, pathways, material_dir / "rate_fractions.png",
                title="Pathway rate fractions", ylabel="Rate fraction",
                colors=pathway_colors, labels=POINTWISE_LABELS,
                context_label=context, observed_rates=observations,
                electrolyte_label=electrolyte_label,
            )

        if observable_points is not None and not observable_points.empty:
            alpha = observable_points.loc[observable_points["observable"] == "alpha"]
            ion = observable_points.loc[observable_points["observable"] == ion_name]
            co = observable_points.loc[observable_points["observable"] == "delta_CO"]

            if not alpha.empty:
                plot(
                    "Pt100/alpha", plot_alpha_comparison,
                    alpha, material_dir / "alpha.png", "Pt100",
                    electrolyte_label=electrolyte_label,
                )
            if not ion.empty:
                plot(
                    f"Pt100/{ion_name}", plot_delta_oh_comparison,
                    ion, material_dir / f"{ion_name}.png", "Pt100",
                    ion_name=ion_name, ion_label=ion_label,
                )
            if not co.empty:
                plot(
                    "Pt100/delta_CO", plot_delta_co_comparison,
                    co, material_dir / "delta_CO.png", "Pt100",
                    electrolyte_label=electrolyte_label,
                )

        if second_order_points is not None and not second_order_points.empty:
            delta2 = second_order_points.loc[second_order_points["observable"] == "delta2"]
            if not delta2.empty:
                plot(
                    "Pt100/second_order_difference", plot_second_order_difference,
                    delta2, material_dir / "second_order_difference.png", "Pt100",
                    electrolyte_column="C_electrolyte_M",
                    electrolyte_label=electrolyte_label,
                    ion_label=ion_label,
                )

        plot(
            "sampling_trace", plot_sampling_trace,
            run.inference_data, run.free_parameter_names,
            figures_dir / "sampling_trace.png",
        )
        plot(
            "sampling_rank", plot_sampling_rank,
            run.inference_data, run.free_parameter_names,
            figures_dir / "sampling_rank.png",
        )
        plot(
            "sampling_energy", plot_sampling_energy,
            run.inference_data, run.free_parameter_names,
            figures_dir / "sampling_energy.png",
        )

        if len(run.free_parameter_names) <= 30:
            plot(
                "sampling_pairs", plot_sampling_pairs,
                run.inference_data, run.free_parameter_names,
                figures_dir / "sampling_pairs.png",
            )
        else:
            status_rows.append(status_row(
                "plot:sampling_pairs", "skipped_large_parameter_set",
                "More than 30 parameters.",
            ))

    status_rows.append(status_row(
        "plots",
        "partial" if failures else "complete",
        f"{attempts - failures}/{attempts} figures generated.",
        plot_attempts=attempts,
        plot_failures=failures,
    ))



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
):
    """Generate Pt100 postprocessing products with shared MKM infrastructure."""
    if plot_level not in {"none", "core", "full"}:
        raise ValueError("plot_level must be one of: none, core, full.")

    tables_dir = paths.fit_tables_dir(run.output_dir)
    figures_dir = paths.fit_figures_dir(run.output_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []

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

        # Retain old Pt100 CSV filenames with the new common summary schema.
        points.to_csv(tables_dir / "posterior_model_points.csv", index=False)
        observations.to_csv(tables_dir / "posterior_observations.csv", index=False)

        condition_summary = observations.groupby(
            ["electrolyte_concentration_M", "CO_mole_fraction"], as_index=False,
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
        loo, loo_material = _compute_loo_stage(run, tables_dir, status_rows)
        write_postprocessing_status(tables_dir, status_rows)
        calibration = _compute_loo_pit_stage(run, loo, tables_dir, status_rows)

    material_summary = _merge_material_summaries(
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
    )

    overall = _overall_postprocessing_status(status_rows)
    status_rows.append(status_row("overall", overall))
    write_postprocessing_status(tables_dir, status_rows)
    return overall

