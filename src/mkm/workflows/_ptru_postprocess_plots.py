"""PtRu figure orchestration built from shared plot primitives."""

from mkm.postprocessing.plotting import (
    COVERAGE_COLORS, PATHWAY_COLORS, POINTWISE_LABELS, plot_alpha_comparison, plot_alpha_overlay_koh,
    plot_alpha_overlay_pco, plot_delta_co_comparison, plot_delta_co_overlay_koh, plot_delta_co_overlay_pco,
    plot_delta_oh_comparison, plot_delta_oh_overlay_pco, plot_loo_diagnostics, plot_loo_pit_summary,
    plot_pointwise_loo, plot_pointwise_loo_pit, plot_observation_grid, plot_parameter_posteriors,
    plot_rate_overlay_koh, plot_rate_overlay_pco, plot_pointwise_variables, plot_sampling_correlations,
    plot_sampling_energy, plot_sampling_pairs, plot_sampling_rank, plot_sampling_trace,
    plot_second_order_difference,
)
from mkm.workflows.postprocessing_common import status_row
from mkm.workflows._ptru_postprocess_products import ptru_electrolyte_label, ptru_ion_name


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
    loo=None,
    calibration=None,
    safe_plot_callback,
    reset_callback,
):
    """Use the existing shared plots with PtRu-specific presentation parameters."""
    if plot_level == "none":
        status_rows.append(status_row("plots", "skipped_user", "Plot level is none."))
        return

    try:
        reset_callback(figures_dir)
    except Exception as error:
        status_rows.append(status_row(
            "plots", "error", f"Could not prepare figures directory: {type(error).__name__}: {error}",
        ))
        return

    attempts = 0
    failures = 0

    def plot(name, function, *args, **kwargs):
        nonlocal attempts, failures
        attempts += 1
        if not safe_plot_callback(status_rows, name, function, *args, **kwargs):
            failures += 1

    medium = run.specification.medium
    electrolyte_label = ptru_electrolyte_label(medium)
    ion_name = ptru_ion_name(medium)
    ion_label = "H+" if medium == "acidic" else "OH"
    context = f"{medium} / {run.specification.fit_scope}"

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

    for material in run.inputs.materials:
        context = f"{material} / {medium}"
        material_observations = observations.loc[observations["material"] == material]
        material_points = points.loc[points["material"] == material]
        material_observable_points = (
            None if observable_points is None else observable_points.loc[observable_points["material"] == material]
        )
        material_second_order_points = (
            None if second_order_points is None else second_order_points.loc[second_order_points["material"] == material]
        )
        material_dir = figures_dir / material
        material_dir.mkdir(parents=True, exist_ok=True)

        plot(
            f"{material}/rates_model", plot_observation_grid,
            material_observations, material_dir / "rates_model.png",
            distribution="model", context_label=context,
            electrolyte_label=electrolyte_label,
        )
        plot(
            f"{material}/rates_predictive", plot_observation_grid,
            material_observations, material_dir / "rates_predictive.png",
            distribution="predictive", y_scale="linear", context_label=context,
            electrolyte_label=electrolyte_label,
        )
        plot(
            f"{material}/residuals", plot_observation_grid,
            material_observations, material_dir / "residuals.png",
            residual=True, y_scale="linear", context_label=context,
            electrolyte_label=electrolyte_label,
        )

        if plot_level == "full":
            plot(
                f"{material}/coverages", plot_pointwise_variables,
                material_points, ("theta_empty", "theta_CO", "theta_OH"),
                material_dir / "coverages.png",
                title="Pt site coverages", ylabel="Pt-site coverage",
                colors=COVERAGE_COLORS, labels=POINTWISE_LABELS,
                context_label=context, electrolyte_label=electrolyte_label,
            )
            if material != "Pt100" and "theta_OH_Ru_median" in material_points:
                plot(
                    f"{material}/ru_coverages", plot_pointwise_variables,
                    material_points, ("theta_empty_Ru", "theta_OH_Ru"),
                    material_dir / "ru_coverages.png",
                    title="Ru site coverages", ylabel="Ru-site coverage",
                    colors={"theta_empty_Ru": "#777777", "theta_OH_Ru": "#009E73"},
                    labels={"theta_empty_Ru": "empty Ru", "theta_OH_Ru": "OH on Ru"},
                    context_label=context, electrolyte_label=electrolyte_label,
                )

            pathways = tuple(
                name for name in ("rate_fraction_BF", "rate_fraction_ER", "rate_fraction_LH")
                if f"{name}_median" in material_points
            )
            if pathways:
                pathway_colors = {
                    "rate_fraction_BF": PATHWAY_COLORS["BF"],
                    "rate_fraction_ER": PATHWAY_COLORS["ER"],
                    "rate_fraction_LH": PATHWAY_COLORS["LH"],
                }
                plot(
                    f"{material}/rate_fractions", plot_pointwise_variables,
                    material_points, pathways, material_dir / "rate_fractions.png",
                    title="Pathway rate fractions", ylabel="Rate fraction",
                    colors=pathway_colors, labels=POINTWISE_LABELS,
                    context_label=context, observed_rates=material_observations,
                    electrolyte_label=electrolyte_label,
                )

            material_loo = None if loo is None else loo.pointwise.loc[loo.pointwise["material"] == material]
            material_pit = None
            if calibration is not None:
                material_pit = calibration.pointwise.loc[
                    calibration.pointwise["material"] == material
                ]
                if not material_pit.empty:
                    plot(
                        f"{material}/loo_pit", plot_loo_pit_summary,
                        material_pit["loo_pit"].to_numpy(dtype=float), run.specification.model_name,
                        material_dir / "loo_pit.png", context_label=context,
                    )
            if material_loo is not None and not material_loo.empty:
                if material_pit is not None and not material_pit.empty:
                    plot(
                        f"{material}/loo_pointwise_pit", plot_pointwise_loo_pit,
                        material_loo, material_pit, material_dir / "loo_pointwise_pit.png",
                        context_label=context, electrolyte_label=electrolyte_label,
                    )
                else:
                    plot(
                        f"{material}/loo_pointwise", plot_pointwise_loo,
                        material_loo, run.specification.model_name, material_dir / "loo_pointwise.png",
                        context_label=context, electrolyte_label=electrolyte_label,
                    )

            if material_observable_points is not None and not material_observable_points.empty:
                alpha = material_observable_points.loc[material_observable_points["observable"] == "alpha"]
                ion = material_observable_points.loc[material_observable_points["observable"] == ion_name]
                co = material_observable_points.loc[material_observable_points["observable"] == "delta_CO"]

                if not alpha.empty:
                    plot(
                        f"{material}/alpha", plot_alpha_comparison,
                        alpha, material_dir / "alpha.png", material,
                        electrolyte_label=electrolyte_label,
                        electrolyte_column="C_electrolyte_M",
                    )
                if not ion.empty:
                    plot(
                        f"{material}/{ion_name}", plot_delta_oh_comparison,
                        ion, material_dir / f"{ion_name}.png", material,
                        ion_name=ion_name, ion_label=ion_label,
                    )
                if not co.empty:
                    plot(
                        f"{material}/delta_CO", plot_delta_co_comparison,
                        co, material_dir / "delta_CO.png", material,
                        electrolyte_label=electrolyte_label,
                        electrolyte_column="C_electrolyte_M",
                    )

            if material_second_order_points is not None and not material_second_order_points.empty:
                delta2 = material_second_order_points.loc[material_second_order_points["observable"] == "delta2"]
                if not delta2.empty:
                    plot(
                        f"{material}/second_order_difference", plot_second_order_difference,
                        delta2, material_dir / "second_order_difference.png", material,
                        electrolyte_column="C_electrolyte_M",
                        electrolyte_label=electrolyte_label,
                        ion_label=ion_label,
                    )

    if plot_level == "full":
        context = f"{medium} / {run.specification.fit_scope}"
        if observable_points is not None and not observable_points.empty:
            alpha_all = observable_points.loc[observable_points["observable"] == "alpha"]
            ion_all = observable_points.loc[observable_points["observable"] == ion_name]
            co_all = observable_points.loc[observable_points["observable"] == "delta_CO"]
            overlay_kwargs = {"electrolyte_column": "C_electrolyte_M", "electrolyte_label": electrolyte_label}
            if not alpha_all.empty:
                plot("alpha_overlay_pco", plot_alpha_overlay_pco, alpha_all,
                     figures_dir / "alpha_overlay_PCO.png", **overlay_kwargs)
                plot("alpha_overlay_electrolyte", plot_alpha_overlay_koh, alpha_all,
                     figures_dir / f"alpha_overlay_{electrolyte_label}.png", **overlay_kwargs)
            if not co_all.empty:
                plot("delta_CO_overlay_pco", plot_delta_co_overlay_pco, co_all,
                     figures_dir / "delta_CO_overlay_PCO.png", **overlay_kwargs)
                plot("delta_CO_overlay_electrolyte", plot_delta_co_overlay_koh, co_all,
                     figures_dir / f"delta_CO_overlay_{electrolyte_label}.png", **overlay_kwargs)
            if not ion_all.empty:
                plot(f"{ion_name}_overlay_pco", plot_delta_oh_overlay_pco, ion_all,
                     figures_dir / f"{ion_name}_overlay_PCO.png", ion_name=ion_name, ion_label=ion_label)

        if observations is not None and not observations.empty:
            plot("rate_overlay_pco", plot_rate_overlay_pco, observations,
                 figures_dir / "rate_overlay_PCO.png", electrolyte_label=electrolyte_label)
            plot("rate_overlay_electrolyte", plot_rate_overlay_koh, observations,
                 figures_dir / f"rate_overlay_{electrolyte_label}.png", electrolyte_label=electrolyte_label)

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

        if loo is not None:
            loo_pit = None if calibration is None else calibration.pointwise["loo_pit"].to_numpy(dtype=float)
            plot(
                "loo_diagnostics", plot_loo_diagnostics,
                loo.loo_result, loo_pit, run.specification.model_name,
                figures_dir / "loo_diagnostics.png", context_label=context,
            )

    status_rows.append(status_row(
        "plots",
        "partial" if failures else "complete",
        f"{attempts - failures}/{attempts} figures generated.",
        plot_attempts=attempts,
        plot_failures=failures,
    ))
