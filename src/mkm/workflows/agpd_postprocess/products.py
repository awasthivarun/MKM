"""Tabular and diagnostic product builders for AgPd postprocessing."""

import warnings

import numpy as np
import pandas as pd

from mkm.observable_maps import build_adjacent_log_order_map, build_alpha_map, build_log_slope_order_map
from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.diagnostics import flatten_posterior_samples, summarize_samples
from mkm.postprocessing.loo import compute_loo_diagnostics
from mkm.postprocessing.materials import summarize_pointwise_loo_by_material
from mkm.postprocessing.observable_comparison import (
    build_experimental_observable_comparison,
    summarize_experimental_observable,
)
from mkm.postprocessing.observables import summarize_posterior_linear_observable
from mkm.postprocessing.second_order import build_second_order_outputs
from mkm.workflows.agpd_basic import load_agpd_preprocessing_config
from .status import LOO_PIT_MAX_PARETO_K, status_row, warning_text

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


def _compute_loo_stage(run, tables_dir, status_rows):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loo = compute_loo_diagnostics(
                run.inference_data,
                run.model_data.observations,
                model_name=run.specification.model_name,
            )
    except Exception as error:
        captured_warnings = warning_text(caught) if "caught" in locals() else ""
        message = f"{type(error).__name__}: {error}"
        if captured_warnings:
            message = f"{message}; warnings: {captured_warnings}"
        status_rows.append(status_row("loo", "error", message))
        return None, None

    loo.summary.to_csv(tables_dir / "loo_summary.csv", index=False)
    loo.pointwise.to_parquet(tables_dir / "loo_pointwise.parquet", index=False)
    loo_material = summarize_pointwise_loo_by_material(loo.pointwise)

    row = loo.summary.iloc[0]
    good_k = float(row["good_k"])
    max_pareto_k = float(row["max_pareto_k"])
    n_bad = int(row["n_pareto_k_above_good_k"])
    captured_warnings = warning_text(caught)
    has_warning = bool(row["warning"]) or n_bad > 0 or bool(captured_warnings)
    status_rows.append(
        status_row(
            "loo",
            "warning" if has_warning else "complete",
            "PSIS-LOO completed with reliability warning(s)." if has_warning else "",
            warning_count=len(caught),
            warnings=captured_warnings,
            good_k=good_k,
            max_pareto_k=max_pareto_k,
            n_pareto_k_above_good_k=n_bad,
        )
    )
    return loo, loo_material


def _compute_loo_pit_stage(run, loo, tables_dir, status_rows):
    if loo is None:
        status_rows.append(status_row("loo_pit", "skipped_loo_unavailable", "PSIS-LOO did not complete."))
        return None

    max_pareto_k = float(loo.summary.iloc[0]["max_pareto_k"])
    if not np.isfinite(max_pareto_k):
        status_rows.append(
            status_row("loo_pit", "skipped_unreliable_psis", "max Pareto-k is non-finite.", max_pareto_k=max_pareto_k)
        )
        return None
    if max_pareto_k >= LOO_PIT_MAX_PARETO_K:
        status_rows.append(
            status_row(
                "loo_pit",
                "skipped_unreliable_psis",
                f"max Pareto-k={max_pareto_k:.6g} >= {LOO_PIT_MAX_PARETO_K:g}; LOO-PIT not attempted.",
                max_pareto_k=max_pareto_k,
            )
        )
        return None

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibration = compute_normal_loo_pit(
                run.inference_data,
                loo.loo_result,
                run.model_data.observations,
                run.inputs,
            )
    except Exception as error:
        captured_warnings = warning_text(caught) if "caught" in locals() else ""
        message = f"{type(error).__name__}: {error}"
        if captured_warnings:
            message = f"{message}; warnings: {captured_warnings}"
        status_rows.append(status_row("loo_pit", "error", message, max_pareto_k=max_pareto_k))
        return None

    calibration.summary.to_csv(tables_dir / "loo_pit_summary.csv", index=False)
    calibration.pointwise.to_parquet(tables_dir / "loo_pit_pointwise.parquet", index=False)
    captured_warnings = warning_text(caught)
    status_rows.append(
        status_row(
            "loo_pit",
            "warning" if captured_warnings else "complete",
            "LOO-PIT completed with numerical warning(s)." if captured_warnings else "",
            warning_count=len(caught),
            warnings=captured_warnings,
            max_pareto_k=max_pareto_k,
        )
    )
    return calibration


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


def _overall_postprocessing_status(status_rows):
    aggregate_stages = {"core", "loo", "loo_pit", "observables", "second_order", "plots"}
    aggregate = {
        row["stage"]: row["status"]
        for row in status_rows
        if row["stage"] in aggregate_stages
    }
    if aggregate.get("core") == "error":
        return "error"
    partial_statuses = {
        "error",
        "partial",
        "skipped_loo_unavailable",
        "skipped_unreliable_psis",
    }
    if any(status in partial_statuses for stage, status in aggregate.items() if stage != "core"):
        return "partial"
    if any(status == "warning" for status in aggregate.values()):
        return "complete_with_warnings"
    return "complete"




