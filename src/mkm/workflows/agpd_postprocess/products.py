"""Tabular and diagnostic product builders for AgPd postprocessing."""

import pandas as pd

from mkm.observable_maps import build_adjacent_log_order_map, build_alpha_map, build_log_slope_order_map
from mkm.postprocessing.observable_comparison import (
    build_experimental_observable_comparison,
    summarize_experimental_observable,
)
from mkm.postprocessing.observables import summarize_posterior_linear_observable
from mkm.postprocessing.second_order import build_second_order_outputs
from mkm.workflows.agpd_basic import load_agpd_preprocessing_config
from mkm.workflows.postprocessing_common import (
    compute_loo_pit_stage as _compute_loo_pit_stage,
    compute_loo_stage as _compute_loo_stage,
    merge_material_summaries as _merge_material_summaries,
    overall_postprocessing_status as _overall_postprocessing_status,
    status_row,
)
from mkm.postprocessing.model_points import summarize_model_points

POINTWISE_VARIABLES = (
    "theta_CO",
    "theta_COOH",
    "theta_OH_Pd",
    "theta_empty_Pd",
    "theta_OH_Ag",
    "theta_empty_Ag",
    "rate_fraction_BF",
    "rate_fraction_ER",
    "rate_fraction_LH",
    "net_pathway_fraction_BF",
    "net_pathway_fraction_ER",
    "net_pathway_fraction_LH",
    "ssa_relative_residual_COOH",
)


def _summarize_model_points(run):
    return summarize_model_points(
        run.inference_data,
        run.model_data.model_points,
        pointwise_variables=POINTWISE_VARIABLES,
    )


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


def _build_observables_stage(run, config, paths, tables_dir, status_rows):
    try:
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
    except Exception as error:
        status_rows.append(status_row("observables", "error", f"{type(error).__name__}: {error}"))
        return None

    status_rows.append(status_row("observables", "complete"))
    return observable_points


def _build_second_order_stage(
    run,
    config,
    paths,
    tables_dir,
    status_rows,
    observable_points,
    potential_step_V,
):
    if observable_points is None:
        status_rows.append(
            status_row(
                "second_order",
                "skipped_observables_unavailable",
                "First-order observable products were unavailable.",
            )
        )
        return None

    try:
        preprocessing_config = load_agpd_preprocessing_config(paths)
        result = build_second_order_outputs(
            inference_data=run.inference_data,
            model_points=run.model_data.model_points,
            conditions=run.model_data.conditions,
            observable_points=observable_points,
            temperature_K=config["temperature_K"],
            koh_values=preprocessing_config["KOH_concentrations_M"],
            co_values=preprocessing_config["CO_mole_fractions"],
            potential_step_V=potential_step_V,
        )
        result.points.to_csv(tables_dir / "second_order_observables.csv", index=False)
        result.summary.to_csv(tables_dir / "second_order_summary.csv", index=False)
    except Exception as error:
        status_rows.append(status_row("second_order", "error", f"{type(error).__name__}: {error}"))
        return None

    status_rows.append(
        status_row(
            "second_order",
            "complete",
            f"Symmetric potential derivative step = {float(potential_step_V):g} V.",
        )
    )
    return result.points


