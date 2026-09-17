"""Plot orchestration for AgPd posterior postprocessing."""

import shutil

from mkm.postprocessing.plotting import (
    COVERAGE_COLORS,
    PATHWAY_COLORS,
    POINTWISE_LABELS,
    plot_alpha_comparison,
    plot_alpha_overlay_koh,
    plot_alpha_overlay_pco,
    plot_delta_co_comparison,
    plot_delta_co_overlay_koh,
    plot_delta_co_overlay_pco,
    plot_delta_oh_comparison,
    plot_delta_oh_overlay_pco,
    plot_rate_overlay_koh,
    plot_rate_overlay_pco,
    plot_loo_diagnostics,
    plot_pointwise_loo_pit,
    plot_loo_pit_summary,
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_loo,
    plot_pointwise_variables,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
    plot_second_order_difference,
)
from mkm.postprocessing.diagnostics import build_parameter_dependence_matrices
from mkm.postprocessing.sampling import sampling_parameter_names
from .status import safe_plot, status_row


PAIR_PLOT_PARAMETER_LIMIT = 30
def _make_plots(
    run,
    config,
    observation_diagnostics,
    model_point_summary,
    figures_dir,
    level,
    *,
    status_rows,
    loo=None,
    calibration=None,
    observable_points=None,
    second_order_points=None,
):
    if level == "none":
        status_rows.append(status_row("plots", "skipped_user", "plot level is none"))
        return

    try:
        if figures_dir.exists():
            shutil.rmtree(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        status_rows.append(
            status_row("plots", "error", f"Could not prepare figures directory: {type(error).__name__}: {error}")
        )
        return

    attempts = 0
    failures = 0

    def plot(name, function, *args, **kwargs):
        nonlocal attempts, failures
        attempts += 1
        if not safe_plot(status_rows, name, function, *args, **kwargs):
            failures += 1

    plot(
        "posterior_parameters",
        plot_parameter_posteriors,
        run.inference_data.posterior,
        run.parameter_specs,
        figures_dir / "posterior_parameters.png",
    )

    pathway_colors = {
        "rate_fraction_BF": PATHWAY_COLORS["BF"],
        "rate_fraction_ER": PATHWAY_COLORS["ER"],
        "rate_fraction_LH": PATHWAY_COLORS["LH"],
    }

    for material in run.inputs.materials:
        material_dir = figures_dir / material
        material_dir.mkdir(parents=True, exist_ok=True)
        observations = observation_diagnostics.loc[observation_diagnostics["material"] == material]
        points = model_point_summary.loc[model_point_summary["material"] == material]

        if run.specification.is_all_materials:
            material_context = (
                f"{material} / {run.specification.parameterization} / "
                f"{run.specification.error_structure}"
            )
        else:
            material_context = material

        plot(
            f"{material}/rates_model",
            plot_observation_grid,
            observations,
            material_dir / "rates_model.png",
            distribution="model",
            context_label=material_context,
        )
        plot(
            f"{material}/rates_predictive",
            plot_observation_grid,
            observations,
            material_dir / "rates_predictive.png",
            distribution="predictive",
            y_scale="linear",
            context_label=material_context,
        )
        plot(
            f"{material}/residuals",
            plot_observation_grid,
            observations,
            material_dir / "residuals.png",
            residual=True,
            y_scale="linear",
            context_label=material_context,
        )

        if level == "full":
            plot(
                f"{material}/coverages_Pd",
                plot_pointwise_variables,
                points,
                ("theta_CO", "theta_OH_Pd", "theta_empty_Pd"),
                material_dir / "coverages_Pd.png",
                title="Pd site coverages",
                ylabel="Pd-site coverage",
                colors=COVERAGE_COLORS,
                labels=POINTWISE_LABELS,
                context_label=material_context,
            )

            ag_fraction = float(
                config.get("surface_composition", {}).get(material, {}).get("Ag_fraction", 0.0)
            )
            if ag_fraction > 0.0:
                plot(
                    f"{material}/coverages_Ag",
                    plot_pointwise_variables,
                    points,
                    ("theta_OH_Ag", "theta_empty_Ag"),
                    material_dir / "coverages_Ag.png",
                    title="Ag site coverages",
                    ylabel="Ag-site coverage",
                    colors=COVERAGE_COLORS,
                    labels=POINTWISE_LABELS,
                    context_label=material_context,
                )

            plot(
                f"{material}/rate_fractions",
                plot_pointwise_variables,
                points,
                ("rate_fraction_BF", "rate_fraction_ER", "rate_fraction_LH"),
                material_dir / "rate_fractions.png",
                title="Pathway rate fractions",
                ylabel="rate fraction",
                colors=pathway_colors,
                labels=POINTWISE_LABELS,
                context_label=material_context,
                observed_rates=observations,
            )

            material_loo = None
            if loo is not None:
                material_loo = loo.pointwise.loc[loo.pointwise["material"] == material]

            material_calibration = None
            if calibration is not None:
                material_calibration = calibration.pointwise.loc[
                    calibration.pointwise["material"] == material
                ]
                if not material_calibration.empty:
                    material_pit = material_calibration["loo_pit"].to_numpy(dtype=float)
                    plot(
                        f"{material}/loo_pit",
                        plot_loo_pit_summary,
                        material_pit,
                        run.specification.model_name,
                        material_dir / "loo_pit.png",
                        context_label=material,
                    )

            if material_loo is not None and not material_loo.empty:
                if material_calibration is not None and not material_calibration.empty:
                    plot(
                        f"{material}/loo_pointwise_pit",
                        plot_pointwise_loo_pit,
                        material_loo,
                        material_calibration,
                        material_dir / "loo_pointwise_pit.png",
                        context_label=material,
                    )
                else:
                    plot(
                        f"{material}/loo_pointwise",
                        plot_pointwise_loo,
                        material_loo,
                        run.specification.model_name,
                        material_dir / "loo_pointwise.png",
                        context_label=material,
                    )

            if observable_points is not None and not observable_points.empty:
                material_points = observable_points.loc[observable_points["material"] == material]

                alpha = material_points.loc[material_points["observable"] == "alpha"]
                if not alpha.empty:
                    plot(f"{material}/alpha", plot_alpha_comparison, alpha, material_dir / "alpha.png", material)

                delta_oh = material_points.loc[material_points["observable"] == "delta_OH"]
                if not delta_oh.empty:
                    plot(
                        f"{material}/delta_OH",
                        plot_delta_oh_comparison,
                        delta_oh,
                        material_dir / "delta_OH.png",
                        material,
                    )

                delta_co = material_points.loc[material_points["observable"] == "delta_CO"]
                if not delta_co.empty:
                    plot(
                        f"{material}/delta_CO",
                        plot_delta_co_comparison,
                        delta_co,
                        material_dir / "delta_CO.png",
                        material,
                    )

            if second_order_points is not None and not second_order_points.empty:
                material_second_order = second_order_points.loc[second_order_points["material"] == material]
                delta2 = material_second_order.loc[material_second_order["observable"] == "delta2"]
                if not delta2.empty:
                    plot(
                        f"{material}/second_order_difference",
                        plot_second_order_difference,
                        delta2,
                        material_dir / "second_order_difference.png",
                        material,
                    )

    if level == "full" and observable_points is not None and not observable_points.empty:
        alpha_all = observable_points.loc[observable_points["observable"] == "alpha"]
        delta_oh_all = observable_points.loc[observable_points["observable"] == "delta_OH"]
        delta_co_all = observable_points.loc[observable_points["observable"] == "delta_CO"]

        if not alpha_all.empty:
            plot(
                "alpha_overlay_pco",
                plot_alpha_overlay_pco,
                alpha_all,
                figures_dir / "alpha_overlay_PCO.png",
            )
            plot(
                "alpha_overlay_koh",
                plot_alpha_overlay_koh,
                alpha_all,
                figures_dir / "alpha_overlay_KOH.png",
            )

        if not delta_co_all.empty:
            plot(
                "delta_CO_overlay_pco",
                plot_delta_co_overlay_pco,
                delta_co_all,
                figures_dir / "delta_CO_overlay_PCO.png",
            )
            plot(
                "delta_CO_overlay_koh",
                plot_delta_co_overlay_koh,
                delta_co_all,
                figures_dir / "delta_CO_overlay_KOH.png",
            )

        if not delta_oh_all.empty:
            plot(
                "delta_OH_overlay_pco",
                plot_delta_oh_overlay_pco,
                delta_oh_all,
                figures_dir / "delta_OH_overlay_PCO.png",
            )

    if level == "full" and observation_diagnostics is not None and not observation_diagnostics.empty:
        plot(
            "rate_overlay_pco",
            plot_rate_overlay_pco,
            observation_diagnostics,
            figures_dir / "rate_overlay_PCO.png",
        )
        plot(
            "rate_overlay_koh",
            plot_rate_overlay_koh,
            observation_diagnostics,
            figures_dir / "rate_overlay_KOH.png",
        )

    if level == "full":
        plot(
            "sampling_trace",
            plot_sampling_trace,
            run.inference_data,
            run.free_parameter_names,
            figures_dir / "sampling_trace.png",
        )
        plot(
            "sampling_rank",
            plot_sampling_rank,
            run.inference_data,
            run.free_parameter_names,
            figures_dir / "sampling_rank.png",
        )
        plot(
            "sampling_energy",
            plot_sampling_energy,
            run.inference_data,
            run.free_parameter_names,
            figures_dir / "sampling_energy.png",
        )
        try:
            pair_names = sampling_parameter_names(run.inference_data.posterior, run.parameter_specs, exclude=())
        except Exception as error:
            attempts += 1
            failures += 1
            status_rows.append(
                status_row(
                    "plot:sampling_pairs",
                    "error",
                    f"Could not resolve sampling pair parameters: {type(error).__name__}: {error}",
                )
            )
        else:
            try:
                covariance, correlation = build_parameter_dependence_matrices(
                    run.inference_data.posterior, pair_names
                )
                tables_dir = figures_dir.parent / "tables"
                tables_dir.mkdir(parents=True, exist_ok=True)
                covariance.to_csv(tables_dir / "posterior_covariance_matrix.csv")
                correlation.to_csv(tables_dir / "posterior_correlation_matrix.csv")
            except Exception as error:
                status_rows.append(
                    status_row(
                        "sampling_dependence_matrices",
                        "error",
                        f"{type(error).__name__}: {error}",
                    )
                )
            else:
                n_parameters = int(covariance.shape[0])
                status_rows.append(
                    status_row(
                        "sampling_dependence_matrices",
                        "complete",
                        f"Saved numeric covariance and correlation matrices for {n_parameters} parameters.",
                    )
                )
                if n_parameters <= PAIR_PLOT_PARAMETER_LIMIT:
                    plot(
                        "sampling_pairs",
                        plot_sampling_pairs,
                        run.inference_data,
                        pair_names,
                        figures_dir / "sampling_pairs.png",
                    )
                else:
                    status_rows.append(
                        status_row(
                            "plot:sampling_pairs",
                            "skipped_large_parameter_set",
                            f"Skipped pair plot for {n_parameters} parameters; limit is "
                            f"{PAIR_PLOT_PARAMETER_LIMIT}.",
                        )
                    )

        if loo is not None:
            loo_pit_values = (
                None
                if calibration is None
                else calibration.pointwise["loo_pit"].to_numpy(dtype=float)
            )
            context = "all materials" if run.specification.is_all_materials else None
            plot(
                "loo_diagnostics",
                plot_loo_diagnostics,
                loo.loo_result,
                loo_pit_values,
                run.specification.model_name,
                figures_dir / "loo_diagnostics.png",
                context_label=context,
            )

    plot_status = "complete" if failures == 0 else "partial"
    message = "" if failures == 0 else f"{failures} of {attempts} plot call(s) failed; see plot:* rows."
    status_rows.append(
        status_row(
            "plots",
            plot_status,
            message,
            plot_attempts=attempts,
            plot_failures=failures,
        )
    )


