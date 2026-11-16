"""Common posterior workflow status, LOO orchestration, and material-summary merging."""

import warnings

import numpy as np
import pandas as pd

from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.loo import compute_loo_diagnostics
from mkm.postprocessing.materials import summarize_pointwise_loo_by_material

POSTPROCESSING_STATUS_COLUMNS = (
    "stage",
    "status",
    "message",
    "warning_count",
    "warnings",
    "good_k",
    "max_pareto_k",
    "n_pareto_k_above_good_k",
    "plot_attempts",
    "plot_failures",
)
LOO_PIT_MAX_PARETO_K = 1.0


def clean_message(value):
    return " ".join(str(value).split())


def warning_text(caught_warnings):
    messages = []
    for caught in caught_warnings:
        message = clean_message(caught.message)
        if message not in messages:
            messages.append(message)
    return "; ".join(messages)


def status_row(stage, status, message="", **values):
    row = {column: "" for column in POSTPROCESSING_STATUS_COLUMNS}
    row.update({"stage": stage, "status": status, "message": clean_message(message)})
    row.update(values)
    return row


def write_postprocessing_status(tables_dir, status_rows):
    pd.DataFrame(status_rows, columns=POSTPROCESSING_STATUS_COLUMNS).to_csv(
        tables_dir / "postprocessing_status.csv",
        index=False,
    )


def safe_plot(status_rows, name, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:
        status_rows.append(status_row(f"plot:{name}", "error", f"{type(error).__name__}: {error}"))
        return False
    status_rows.append(status_row(f"plot:{name}", "complete"))
    return True


def merge_material_summaries(*frames):
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


def compute_loo_stage(run, tables_dir, status_rows):
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


def compute_loo_pit_stage(run, loo, tables_dir, status_rows):
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


def overall_postprocessing_status(status_rows):
    aggregate_stages = {"core", "loo", "loo_pit", "observables", "second_order", "plots", "drc"}
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


# Deliberately limited to owned generated TABLES, not independently produced checkpoints.
POSTPROCESSING_STAGE_FILES = {
    "core": (
        "observation_diagnostics.parquet", "model_point_diagnostics.parquet", "physical_checks.csv",
        "material_summary.csv", "posterior_parameter_diagnostics.csv", "posterior_covariance_matrix.csv",
        "posterior_correlation_matrix.csv", "residual_curves.csv", "shared_replicate_residuals.csv",
        "posterior_model_points.csv", "posterior_observations.csv", "posterior_condition_summary.csv",
    ),
    "loo": ("loo_summary.csv", "loo_pointwise.parquet"),
    "loo_pit": ("loo_pit_summary.csv", "loo_pit_pointwise.parquet"),
    "observables": ("experimental_observable_comparisons.parquet", "experimental_observable_summary.csv"),
    "second_order": ("second_order_observables.csv", "second_order_summary.csv"),
}


def clear_postprocessing_tables(tables_dir, stages=tuple(POSTPROCESSING_STAGE_FILES)):
    """Remove only previously generated tables from specified stages before a new invocation."""
    for stage in stages:
        for filename in POSTPROCESSING_STAGE_FILES[stage]:
            (tables_dir / filename).unlink(missing_ok=True)


def postprocessing_execution_failed(status_rows):
    return any(row["status"] == "error" for row in status_rows)


def write_postprocessing_manifest(output_dir, run, *, options, status="running"):
    from datetime import datetime, timezone
    from mkm.provenance import write_run_metadata
    path = output_dir / "postprocessing_manifest.yaml"
    metadata = getattr(run, "metadata", {}) or {}
    tables_dir = output_dir / "tables"
    write_run_metadata({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "current_tables": sorted(p.name for p in tables_dir.iterdir() if p.is_file())
        if tables_dir.exists() else [],
        "posterior_run": {
            "model": metadata.get("model"), "materials": metadata.get("materials"),
            "resolved_fit_sha256": metadata.get("resolved_fit_sha256"),
            "inputs": metadata.get("inputs"),
        },
        "options": dict(options),
        "retained_figures_when_plot_level_none": options.get("plot_level") == "none",
    }, path)
