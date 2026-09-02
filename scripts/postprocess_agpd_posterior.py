"""Generate consolidated diagnostics for an AgPd posterior run."""

from argparse import ArgumentParser
import shutil

import numpy as np
import pandas as pd

from mkm.models.agpd_basic import available_agpd_models
from mkm.observable_maps import (
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
)
from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.diagnostics import (
    build_physical_checks,
    flatten_posterior_samples,
    summarize_samples,
)
from mkm.postprocessing.loo import compute_loo_diagnostics
from mkm.postprocessing.materials import (
    summarize_observation_diagnostics_by_material,
    summarize_pointwise_loo_by_material,
    summarize_residual_structure_by_material,
)
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
    plot_loo_pit_conditions,
    plot_loo_pit_coverage,
    plot_loo_pit_ecdf,
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pareto_k,
    plot_pointwise_loo,
    plot_pointwise_variables,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
)
from mkm.postprocessing.predictions import build_observation_diagnostics
from mkm.postprocessing.residuals import (
    summarize_residual_curves,
    summarize_shared_replicate_residuals,
)
from mkm.postprocessing.sampling import sampling_parameter_names
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import (
    load_agpd_model_config,
    load_agpd_preprocessing_config,
)
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import load_agpd_posterior_run


POINTWISE_VARIABLES = (
    "theta_CO",
    "theta_OH_Pd",
    "theta_empty_Pd",
    "theta_OH_Ag",
    "theta_empty_Ag",
    "rate_fraction_BF",
    "rate_fraction_ER",
    "rate_fraction_LH",
)


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
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--skip-observables", action="store_true")
    parser.add_argument(
        "--plot-level",
        choices=("none", "core", "full"),
        default="core",
    )
    return parser.parse_args()


def _summarize_model_points(run):
    result = run.model_data.model_points.copy()
    posterior = run.inference_data.posterior

    ln_rate = flatten_posterior_samples(posterior["ln_rate_model"])
    for prefix, values in (("ln_rate_model", ln_rate), ("rate_model", np.exp(ln_rate))):
        summary = summarize_samples(values)
        for statistic, statistic_values in summary.items():
            result[f"{prefix}_{statistic}"] = statistic_values

    for variable in POINTWISE_VARIABLES:
        if variable not in posterior:
            continue
        summary = summarize_samples(flatten_posterior_samples(posterior[variable]))
        for statistic, statistic_values in summary.items():
            result[f"{variable}_{statistic}"] = statistic_values

    return result


def _merge_material_summaries(*frames):
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    result = frames[0].copy()
    for frame in frames[1:]:
        frame = frame.copy()
        overlap = sorted((set(result.columns) & set(frame.columns)) - {"material"})
        if overlap:
            comparison = result[["material", *overlap]].merge(
                frame[["material", *overlap]],
                on="material",
                how="outer",
                suffixes=("_left", "_right"),
                indicator=True,
                validate="one_to_one",
            )
            if not comparison["_merge"].eq("both").all():
                raise ValueError(
                    "Material summary tables with duplicate columns must contain "
                    "the same materials."
                )

            for column in overlap:
                left = comparison[f"{column}_left"]
                right = comparison[f"{column}_right"]
                if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
                    right
                ):
                    equal = np.allclose(
                        left.to_numpy(dtype=float),
                        right.to_numpy(dtype=float),
                        equal_nan=True,
                    )
                else:
                    left = left.astype("object").where(left.notna(), None)
                    right = right.astype("object").where(right.notna(), None)
                    equal = left.equals(right)
                if not equal:
                    raise ValueError(
                        f"Conflicting material-summary column '{column}'."
                    )
            frame = frame.drop(columns=overlap)

        result = result.merge(
            frame,
            on="material",
            how="outer",
            validate="one_to_one",
        )
    return result


def _load_experimental_observables(paths, materials):
    materials = set(materials)
    alpha = pd.read_parquet(paths.agpd_summary_path)
    oh = pd.read_parquet(paths.agpd_delta_oh_path)
    co = pd.read_parquet(paths.agpd_delta_co_path)
    return (
        alpha.loc[alpha["material"].isin(materials)].copy(),
        oh.loc[oh["material"].isin(materials)].copy(),
        co.loc[co["material"].isin(materials)].copy(),
    )


def _build_observable_outputs(run, config, preprocessing_config, paths):
    model_points = run.model_data.model_points
    alpha_map = build_alpha_map(model_points, config["temperature_K"])
    oh_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=preprocessing_config["KOH_concentrations_M"],
        group_columns=["material", "CO_mole_fraction"],
    )
    co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=preprocessing_config["CO_mole_fractions"],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    alpha = summarize_posterior_linear_observable(run.inference_data, alpha_map)
    oh = summarize_posterior_linear_observable(run.inference_data, oh_map)
    co = summarize_posterior_linear_observable(run.inference_data, co_map)

    condition_metadata = run.model_data.conditions[
        ["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]
    ].rename(columns={"electrolyte_concentration_M": "C_KOH_M"})
    alpha_pooled = alpha.pooled.merge(
        condition_metadata,
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    alpha_chain = alpha.by_chain.merge(
        condition_metadata,
        on="condition_id",
        how="left",
        validate="many_to_one",
    )

    rename_co = {
        "electrolyte_concentration_M": "C_KOH_M",
        "lower_CO_mole_fraction": "CO_lower_mole_fraction",
        "upper_CO_mole_fraction": "CO_upper_mole_fraction",
    }
    co_pooled = co.pooled.rename(columns=rename_co)
    co_chain = co.by_chain.rename(columns=rename_co)

    experimental_alpha, experimental_oh, experimental_co = _load_experimental_observables(
        paths,
        run.inputs.materials,
    )
    comparisons = {
        "alpha": build_experimental_observable_comparison(
            pooled=alpha_pooled,
            by_chain=alpha_chain,
            experimental=experimental_alpha,
            key_columns=["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index"],
            observed_column="alpha_mean",
            observed_sd_column="alpha_sd",
        ),
        "delta_OH": build_experimental_observable_comparison(
            pooled=oh.pooled,
            by_chain=oh.by_chain,
            experimental=experimental_oh,
            key_columns=["material", "CO_mole_fraction", "analysis_grid_index"],
            observed_column="delta_OH",
            observed_sd_column="delta_OH_sd",
        ),
        "delta_CO": build_experimental_observable_comparison(
            pooled=co_pooled,
            by_chain=co_chain,
            experimental=experimental_co,
            key_columns=[
                "material",
                "C_KOH_M",
                "CO_lower_mole_fraction",
                "CO_upper_mole_fraction",
                "analysis_grid_index",
            ],
            observed_column="delta_CO",
            observed_sd_column="delta_CO_sd",
        ),
    }

    pooled = []
    summary = []
    for name, comparison in comparisons.items():
        frame = comparison.pooled.copy()
        frame.insert(0, "observable", name)
        pooled.append(frame)
        summary.append(summarize_experimental_observable(name, comparison))
    return pd.concat(pooled, ignore_index=True, sort=False), pd.DataFrame(summary)


def _make_plots(
    run,
    config,
    observation_diagnostics,
    model_point_summary,
    figures_dir,
    level,
    *,
    loo=None,
    calibration=None,
    observable_points=None,
):
    if level == "none":
        return

    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_parameter_posteriors(
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
        observations = observation_diagnostics.loc[
            observation_diagnostics["material"] == material
        ]
        points = model_point_summary.loc[model_point_summary["material"] == material]

        if run.specification.is_all_materials:
            material_context = (
                f"{material} / {run.specification.parameterization} / "
                f"{run.specification.error_structure}"
            )
        else:
            material_context = material

        plot_observation_grid(
            observations,
            material_dir / "rates_model.png",
            distribution="model",
            context_label=material_context,
        )
        plot_observation_grid(
            observations,
            material_dir / "rates_predictive.png",
            distribution="predictive",
            y_scale="linear",
            context_label=material_context,
        )
        plot_observation_grid(
            observations,
            material_dir / "residuals.png",
            residual=True,
            y_scale="linear",
            context_label=material_context,
        )

        if level == "full":
            plot_pointwise_variables(
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
                config.get("surface_composition", {})
                .get(material, {})
                .get("Ag_fraction", 0.0)
            )
            if ag_fraction > 0.0:
                plot_pointwise_variables(
                    points,
                    ("theta_OH_Ag", "theta_empty_Ag"),
                    material_dir / "coverages_Ag.png",
                    title="Ag site coverages",
                    ylabel="Ag-site coverage",
                    colors=COVERAGE_COLORS,
                    labels=POINTWISE_LABELS,
                    context_label=material_context,
                )

            plot_pointwise_variables(
                points,
                ("rate_fraction_BF", "rate_fraction_ER", "rate_fraction_LH"),
                material_dir / "rate_fractions.png",
                title="Pathway rate fractions",
                ylabel="rate fraction",
                colors=pathway_colors,
                labels=POINTWISE_LABELS,
                context_label=material_context,
            )

            if loo is not None:
                material_loo = loo.pointwise.loc[loo.pointwise["material"] == material]
                if not material_loo.empty:
                    plot_pointwise_loo(
                        material_loo,
                        run.specification.model_name,
                        material_dir / "loo_pointwise.png",
                        context_label=material,
                    )

            if calibration is not None:
                material_calibration = calibration.pointwise.loc[
                    calibration.pointwise["material"] == material
                ]
                if not material_calibration.empty:
                    material_pit = material_calibration["loo_pit"].to_numpy(dtype=float)
                    plot_loo_pit_ecdf(
                        material_pit,
                        run.specification.model_name,
                        material_dir / "loo_pit_ecdf.png",
                        context_label=material,
                    )
                    plot_loo_pit_coverage(
                        material_pit,
                        run.specification.model_name,
                        material_dir / "loo_pit_coverage.png",
                        context_label=material,
                    )
                    plot_loo_pit_conditions(
                        material_calibration,
                        run.specification.model_name,
                        material_dir / "loo_pit_conditions.png",
                        context_label=material,
                    )

            if observable_points is not None and not observable_points.empty:
                material_points = observable_points.loc[
                    observable_points["material"] == material
                ]

                alpha = material_points.loc[material_points["observable"] == "alpha"]
                if not alpha.empty:
                    plot_alpha_comparison(
                        alpha,
                        material_dir / "alpha.png",
                        material,
                    )

                delta_oh = material_points.loc[
                    material_points["observable"] == "delta_OH"
                ]
                if not delta_oh.empty:
                    plot_delta_oh_comparison(
                        delta_oh,
                        material_dir / "delta_OH.png",
                        material,
                    )

                delta_co = material_points.loc[
                    material_points["observable"] == "delta_CO"
                ]
                if not delta_co.empty:
                    plot_delta_co_comparison(
                        delta_co,
                        material_dir / "delta_CO.png",
                        material,
                    )

    if level == "full":
        plot_sampling_trace(
            run.inference_data,
            run.free_parameter_names,
            figures_dir / "sampling_trace.png",
        )
        plot_sampling_rank(
            run.inference_data,
            run.free_parameter_names,
            figures_dir / "sampling_rank.png",
        )
        plot_sampling_energy(
            run.inference_data,
            run.free_parameter_names,
            figures_dir / "sampling_energy.png",
        )
        pair_names = sampling_parameter_names(
            run.inference_data.posterior,
            run.parameter_specs,
        )
        plot_sampling_pairs(
            run.inference_data,
            pair_names,
            figures_dir / "sampling_pairs.png",
        )

        if loo is not None:
            plot_pareto_k(
                loo.loo_result,
                run.specification.model_name,
                figures_dir / "loo_pareto_k.png",
            )

        if calibration is not None:
            loo_pit_values = calibration.pointwise["loo_pit"].to_numpy(dtype=float)
            plot_loo_pit_ecdf(
                loo_pit_values,
                run.specification.model_name,
                figures_dir / "loo_pit_ecdf.png",
                context_label="all materials" if run.specification.is_all_materials else None,
            )
            plot_loo_pit_coverage(
                loo_pit_values,
                run.specification.model_name,
                figures_dir / "loo_pit_coverage.png",
                context_label="all materials" if run.specification.is_all_materials else None,
            )

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
        reconstruct_pointwise=True,
        progressbar=True,
    )

    tables_dir = paths.fit_tables_dir(run.output_dir)
    figures_dir = paths.fit_figures_dir(run.output_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    observation_diagnostics = build_observation_diagnostics(
        run.inference_data,
        run.model_data,
        run.inputs,
        random_seed=args.random_seed,
    )
    observation_diagnostics.to_parquet(
        tables_dir / "observation_diagnostics.parquet",
        index=False,
    )

    model_point_summary = _summarize_model_points(run)
    model_point_summary.to_parquet(
        tables_dir / "model_point_diagnostics.parquet",
        index=False,
    )

    physical_checks = build_physical_checks(run.inference_data.posterior)
    physical_checks.to_csv(tables_dir / "physical_checks.csv", index=False)

    curve_residuals = summarize_residual_curves(observation_diagnostics)
    shared_residuals = summarize_shared_replicate_residuals(observation_diagnostics)
    material_observation = summarize_observation_diagnostics_by_material(
        observation_diagnostics
    )
    material_residual = summarize_residual_structure_by_material(
        curve_residuals,
        shared_residuals,
    )

    loo = None
    calibration = None
    loo_material = None
    if not args.skip_loo:
        loo = compute_loo_diagnostics(
            run.inference_data,
            run.model_data.observations,
            model_name=run.specification.model_name,
        )
        loo.summary.to_csv(tables_dir / "loo_summary.csv", index=False)
        loo.pointwise.to_parquet(tables_dir / "loo_pointwise.parquet", index=False)
        loo_material = summarize_pointwise_loo_by_material(loo.pointwise)

        calibration = compute_normal_loo_pit(
            run.inference_data,
            loo.loo_result,
            run.model_data.observations,
            run.inputs,
        )
        calibration.summary.to_csv(
            tables_dir / "loo_pit_summary.csv",
            index=False,
        )
        calibration.pointwise.to_parquet(
            tables_dir / "loo_pit_pointwise.parquet",
            index=False,
        )

    material_summary = _merge_material_summaries(
        material_observation,
        material_residual,
        loo_material,
    )
    material_summary.to_csv(tables_dir / "material_summary.csv", index=False)

    observable_points = None
    if not args.skip_observables:
        preprocessing_config = load_agpd_preprocessing_config(paths)
        observable_points, observable_summary = _build_observable_outputs(
            run,
            config,
            preprocessing_config,
            paths,
        )
        observable_points.to_parquet(
            tables_dir / "experimental_observable_comparisons.parquet",
            index=False,
        )
        observable_summary.to_csv(
            tables_dir / "experimental_observable_summary.csv",
            index=False,
        )

    _make_plots(
        run,
        config,
        observation_diagnostics,
        model_point_summary,
        figures_dir,
        args.plot_level,
        loo=loo,
        calibration=calibration,
        observable_points=observable_points,
    )

    print(f"Postprocessing products saved to: {run.output_dir}")


if __name__ == "__main__":
    main()
