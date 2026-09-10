"""Run the complete AgPd individual-material finite-rate CO model grid."""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import yaml


ALLOY_MATERIALS = ("Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10")
ALLOY_MODELS = ("CO_LH", "CO_ER", "CO_BF", "CO_ER_LH", "CO_BF_LH", "CO_BF_ER", "CO_BF_ER_LH")
PD_MODELS = ("CO_LH", "CO_ER", "CO_ER_LH")
TARGET_ACCEPT_SEQUENCE = (0.90, 0.95, 0.99)

DIAGNOSTIC_FIELDS = (
    "grid_started_utc",
    "material",
    "model",
    "attempt",
    "target_accept",
    "terminal_attempt",
    "fit_returncode",
    "fit_wall_seconds",
    "run_created_utc",
    "git_commit",
    "data_sha256",
    "model_config_sha256",
    "metadata_status",
    "n_divergent",
    "n_max_treedepth",
    "fraction_max_treedepth",
    "min_bfmi",
    "max_rhat",
    "max_rhat_parameter",
    "n_rhat_gt_1_01",
    "n_rhat_gt_1_05",
    "n_rhat_gt_1_10",
    "rhat_gt_1_05_parameters",
    "min_ess_bulk",
    "min_ess_bulk_parameter",
    "min_ess_tail",
    "min_ess_tail_parameter",
    "divergence_class",
    "treedepth_class",
    "bfmi_class",
    "rhat_class",
    "ess_bulk_class",
    "ess_tail_class",
    "sampling_status",
    "status_reasons",
    "diagnostics_error",
    "postprocess_returncode",
    "postprocess_wall_seconds",
    "postprocess_overall_status",
    "postprocess_core_status",
    "postprocess_loo_status",
    "postprocess_loo_message",
    "loo_good_k",
    "loo_max_pareto_k",
    "loo_n_pareto_k_above_good_k",
    "loo_warning_count",
    "loo_warnings",
    "postprocess_loo_pit_status",
    "postprocess_loo_pit_message",
    "postprocess_observables_status",
    "postprocess_plots_status",
    "postprocess_plot_attempts",
    "postprocess_plot_failures",
    "postprocess_optional_failures",
    "postprocess_status_error",
    "drc_returncode",
    "drc_wall_seconds",
    "comparison_included",
    "comparison_excluded_reason",
    "comparison_name",
    "comparison_models",
    "comparison_returncode",
    "comparison_wall_seconds",
    "comparison_status",
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing posterior directories on the first attempt for each fit.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Rebuild per-material model comparisons from the existing grid diagnostics without rerunning fits.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if args.compare_only and args.overwrite:
        parser.error("--overwrite cannot be used with --compare-only.")
    return args


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _run(command):
    print("\n> " + " ".join(str(part) for part in command), flush=True)
    start = perf_counter()
    returncode = subprocess.run(command, check=False).returncode
    return returncode, perf_counter() - start


def _individual_root(root):
    return root / "results" / "AgPd_COOx_basic" / "posterior" / "individual"


def _fit_dir(root, material, model):
    return _individual_root(root) / material / model


def _fit_command(root, model, material, target_accept, overwrite):
    command = [
        sys.executable,
        str(root / "scripts" / "fit_agpd_posterior.py"),
        model,
        "--material",
        material,
        "--error-structure",
        "material",
        "--target-accept",
        f"{target_accept:.2f}",
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _postprocess_command(root, model, material):
    return [
        sys.executable,
        str(root / "scripts" / "postprocess_agpd_posterior.py"),
        model,
        "--material",
        material,
        "--error-structure",
        "material",
        "--plot-level",
        "full",
    ]


def _drc_command(root, model, material):
    return [
        sys.executable,
        str(root / "scripts" / "postprocess_agpd_drc.py"),
        model,
        "--material",
        material,
        "--error-structure",
        "material",
        "--check-half-step",
    ]


def _comparison_name(material):
    return f"{material}_all_models"


def _comparison_command(root, material, models):
    command = [
        sys.executable,
        str(root / "scripts" / "compare_agpd_models.py"),
        "--comparison-name",
        _comparison_name(material),
        "--material",
        material,
    ]
    for model in models:
        command.extend(["--run", f"{model},{model}"])
    return command


def _as_float(value):
    if value in (None, ""):
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _as_int(value):
    result = _as_float(value)
    return int(result) if math.isfinite(result) else None


def _format_float(value):
    return value if value is not None and math.isfinite(_as_float(value)) else ""


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_diagnostics(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Grid diagnostics do not exist: {path}. Run the full individual grid before using --compare-only."
        )

    with open(path, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    return [
        {field: row.get(field, "") for field in DIAGNOSTIC_FIELDS}
        for row in rows
    ]


def _read_metadata(run_dir):
    with open(run_dir / "run_metadata.yaml", "r") as file:
        metadata = yaml.safe_load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Run metadata must contain a mapping: {run_dir / 'run_metadata.yaml'}")
    return metadata


def _parameter_diagnostics(run_dir):
    path = run_dir / "posterior_parameters.csv"
    with open(path, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    def finite_rows(column):
        return [(row, _as_float(row.get(column))) for row in rows if math.isfinite(_as_float(row.get(column)))]

    rhat_rows = finite_rows("rhat")
    bulk_rows = finite_rows("ess_bulk")
    tail_rows = finite_rows("ess_tail")

    max_rhat_row, max_rhat = max(rhat_rows, key=lambda item: item[1]) if rhat_rows else ({}, math.nan)
    min_bulk_row, min_bulk = min(bulk_rows, key=lambda item: item[1]) if bulk_rows else ({}, math.nan)
    min_tail_row, min_tail = min(tail_rows, key=lambda item: item[1]) if tail_rows else ({}, math.nan)

    def label(row):
        return row.get("parameter") or row.get("variable") or ""

    over_101 = [label(row) for row, value in rhat_rows if value > 1.01]
    over_105 = [label(row) for row, value in rhat_rows if value > 1.05]
    over_110 = [label(row) for row, value in rhat_rows if value > 1.10]

    return {
        "max_rhat_from_table": max_rhat,
        "max_rhat_parameter": label(max_rhat_row),
        "n_rhat_gt_1_01": len(over_101),
        "n_rhat_gt_1_05": len(over_105),
        "n_rhat_gt_1_10": len(over_110),
        "rhat_gt_1_05_parameters": ";".join(over_105),
        "min_ess_bulk_from_table": min_bulk,
        "min_ess_bulk_parameter": label(min_bulk_row),
        "min_ess_tail_from_table": min_tail,
        "min_ess_tail_parameter": label(min_tail_row),
    }


def _classify_rhat(value):
    if not math.isfinite(value):
        return "unavailable"
    if value <= 1.01:
        return "clean"
    if value <= 1.05:
        return "mild"
    if value <= 1.10:
        return "high"
    return "severe"


def _classify_ess(value):
    if not math.isfinite(value):
        return "unavailable"
    if value >= 400:
        return "clean"
    if value >= 100:
        return "caution"
    return "low"


def _classify_bfmi(value):
    if not math.isfinite(value):
        return "unavailable"
    if value >= 0.30:
        return "clean"
    if value >= 0.10:
        return "low"
    return "severe"


def _classify_treedepth(n_max_treedepth, fraction):
    if n_max_treedepth == 0:
        return "clean"
    if not math.isfinite(fraction):
        return "unavailable"
    if fraction < 0.01:
        return "mild"
    if fraction < 0.05:
        return "high"
    return "severe"


def _classify_divergences(n_divergent, target_accept):
    if n_divergent == 0:
        return "clean"
    if target_accept < TARGET_ACCEPT_SEQUENCE[-1]:
        return "retry"
    return "persistent_at_0.99"


def _sampling_status(record):
    if record["diagnostics_error"]:
        return "FAIL_DIAGNOSTICS", "sampling diagnostics could not be read"

    hard_failures = []
    cautions = []

    if record["divergence_class"] == "persistent_at_0.99":
        hard_failures.append("divergences persisted at target_accept=0.99")
    if record["bfmi_class"] == "severe":
        hard_failures.append("min BFMI < 0.10")
    if record["treedepth_class"] == "severe":
        hard_failures.append("max-treedepth fraction >= 5%")

    # R-hat and ESS are intentionally warnings, not automatic failures. Broad or numerically
    # redundant kinetic parameters can mix poorly while leaving the model predictions effectively unchanged.
    if record["rhat_class"] in {"mild", "high", "severe"}:
        cautions.append(f"R-hat {record['rhat_class']}")
    if record["ess_bulk_class"] in {"caution", "low"}:
        cautions.append(f"bulk ESS {record['ess_bulk_class']}")
    if record["ess_tail_class"] in {"caution", "low"}:
        cautions.append(f"tail ESS {record['ess_tail_class']}")
    if record["bfmi_class"] == "low":
        cautions.append("min BFMI 0.10-0.30")
    if record["treedepth_class"] in {"mild", "high"}:
        cautions.append(f"max treedepth {record['treedepth_class']}")

    if hard_failures:
        return "FAIL_SAMPLER_HEALTH", "; ".join(hard_failures + cautions)
    if cautions:
        return "CAUTION", "; ".join(cautions)
    return "PASS", ""


def _attempt_record(grid_started_utc, material, model, attempt, target_accept):
    return {field: "" for field in DIAGNOSTIC_FIELDS} | {
        "grid_started_utc": grid_started_utc,
        "material": material,
        "model": model,
        "attempt": attempt,
        "target_accept": target_accept,
        "terminal_attempt": False,
        "comparison_included": False,
    }


def _populate_sampling_diagnostics(record, run_dir):
    try:
        metadata = _read_metadata(run_dir)
        health = metadata.get("sampling_health", {})
        parameter_diagnostics = _parameter_diagnostics(run_dir)

        record.update(
            {
                "run_created_utc": metadata.get("created_utc", ""),
                "git_commit": metadata.get("git_commit", ""),
                "data_sha256": metadata.get("inputs", {}).get("data_sha256", ""),
                "model_config_sha256": metadata.get("inputs", {}).get("model_config_sha256", ""),
                "metadata_status": metadata.get("status", ""),
                "n_divergent": _as_int(health.get("n_divergent")),
                "n_max_treedepth": _as_int(health.get("n_max_treedepth")),
                "fraction_max_treedepth": _format_float(health.get("fraction_max_treedepth")),
                "min_bfmi": _format_float(health.get("min_bfmi")),
                "max_rhat": _format_float(health.get("max_rhat")),
                "min_ess_bulk": _format_float(health.get("min_ess_bulk")),
                "min_ess_tail": _format_float(health.get("min_ess_tail")),
                "max_rhat_parameter": parameter_diagnostics["max_rhat_parameter"],
                "n_rhat_gt_1_01": parameter_diagnostics["n_rhat_gt_1_01"],
                "n_rhat_gt_1_05": parameter_diagnostics["n_rhat_gt_1_05"],
                "n_rhat_gt_1_10": parameter_diagnostics["n_rhat_gt_1_10"],
                "rhat_gt_1_05_parameters": parameter_diagnostics["rhat_gt_1_05_parameters"],
                "min_ess_bulk_parameter": parameter_diagnostics["min_ess_bulk_parameter"],
                "min_ess_tail_parameter": parameter_diagnostics["min_ess_tail_parameter"],
            }
        )

        max_rhat = _as_float(record["max_rhat"])
        min_ess_bulk = _as_float(record["min_ess_bulk"])
        min_ess_tail = _as_float(record["min_ess_tail"])
        min_bfmi = _as_float(record["min_bfmi"])
        fraction_max_treedepth = _as_float(record["fraction_max_treedepth"])
        n_divergent = record["n_divergent"]
        n_max_treedepth = record["n_max_treedepth"]

        record["divergence_class"] = _classify_divergences(n_divergent, float(record["target_accept"]))
        record["treedepth_class"] = _classify_treedepth(n_max_treedepth, fraction_max_treedepth)
        record["bfmi_class"] = _classify_bfmi(min_bfmi)
        record["rhat_class"] = _classify_rhat(max_rhat)
        record["ess_bulk_class"] = _classify_ess(min_ess_bulk)
        record["ess_tail_class"] = _classify_ess(min_ess_tail)
    except Exception as error:
        record["diagnostics_error"] = f"{type(error).__name__}: {error}"


def _postprocessing_status_rows(run_dir):
    path = run_dir / "tables" / "postprocessing_status.csv"
    with open(path, "r", newline="") as file:
        rows = list(csv.DictReader(file))
    return {row.get("stage", ""): row for row in rows if row.get("stage")}


def _populate_postprocessing_status(record, run_dir):
    try:
        rows = _postprocessing_status_rows(run_dir)
        core = rows.get("core", {})
        loo = rows.get("loo", {})
        loo_pit = rows.get("loo_pit", {})
        observables = rows.get("observables", {})
        plots = rows.get("plots", {})
        overall = rows.get("overall", {})

        record.update(
            {
                "postprocess_overall_status": overall.get("status", ""),
                "postprocess_core_status": core.get("status", ""),
                "postprocess_loo_status": loo.get("status", ""),
                "postprocess_loo_message": loo.get("message", ""),
                "loo_good_k": loo.get("good_k", ""),
                "loo_max_pareto_k": loo.get("max_pareto_k", ""),
                "loo_n_pareto_k_above_good_k": loo.get("n_pareto_k_above_good_k", ""),
                "loo_warning_count": loo.get("warning_count", ""),
                "loo_warnings": loo.get("warnings", ""),
                "postprocess_loo_pit_status": loo_pit.get("status", ""),
                "postprocess_loo_pit_message": loo_pit.get("message", ""),
                "postprocess_observables_status": observables.get("status", ""),
                "postprocess_plots_status": plots.get("status", ""),
                "postprocess_plot_attempts": plots.get("plot_attempts", ""),
                "postprocess_plot_failures": plots.get("plot_failures", ""),
            }
        )

        optional_failures = []
        for stage in ("loo", "loo_pit", "observables", "plots"):
            status = rows.get(stage, {}).get("status", "")
            if status in {"error", "partial", "skipped_loo_unavailable", "skipped_unreliable_psis"}:
                optional_failures.append(f"{stage}:{status}")
        record["postprocess_optional_failures"] = ";".join(optional_failures)
    except Exception as error:
        record["postprocess_status_error"] = f"{type(error).__name__}: {error}"


def _exclude_from_comparison(record, reason):
    record["comparison_included"] = False
    existing = str(record.get("comparison_excluded_reason", "")).strip()
    record["comparison_excluded_reason"] = f"{existing}; {reason}".strip("; ") if existing else reason


def _write_diagnostics(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _run_one_fit(root, args, grid_started_utc, diagnostics_path, records, material, model):
    run_dir = _fit_dir(root, material, model)

    for attempt, target_accept in enumerate(TARGET_ACCEPT_SEQUENCE, start=1):
        record = _attempt_record(grid_started_utc, material, model, attempt, target_accept)
        overwrite = args.overwrite if attempt == 1 else True

        fit_returncode, fit_seconds = _run(_fit_command(root, model, material, target_accept, overwrite))
        record["fit_returncode"] = fit_returncode
        record["fit_wall_seconds"] = f"{fit_seconds:.3f}"

        if fit_returncode != 0:
            record["terminal_attempt"] = True
            record["sampling_status"] = "FIT_ERROR"
            record["status_reasons"] = f"fit command returned {fit_returncode}"
            record["comparison_excluded_reason"] = "fit command failed"
            records.append(record)
            _write_diagnostics(diagnostics_path, records)
            return record

        _populate_sampling_diagnostics(record, run_dir)
        n_divergent = record["n_divergent"]

        if record["diagnostics_error"]:
            record["terminal_attempt"] = True
        elif n_divergent and target_accept < TARGET_ACCEPT_SEQUENCE[-1]:
            record["sampling_status"] = "RETRY_DIVERGENCES"
            record["status_reasons"] = (
                f"{n_divergent} divergence(s); retrying at the next target_accept"
            )
            records.append(record)
            _write_diagnostics(diagnostics_path, records)
            continue
        else:
            record["terminal_attempt"] = True

        if not record["sampling_status"]:
            record["sampling_status"], record["status_reasons"] = _sampling_status(record)

        hard_failure = record["sampling_status"] in {"FIT_ERROR", "FAIL_DIAGNOSTICS", "FAIL_SAMPLER_HEALTH"}
        record["comparison_included"] = not hard_failure
        if hard_failure:
            record["comparison_excluded_reason"] = record["status_reasons"]

        records.append(record)
        _write_diagnostics(diagnostics_path, records)

        post_returncode, post_seconds = _run(_postprocess_command(root, model, material))
        record["postprocess_returncode"] = post_returncode
        record["postprocess_wall_seconds"] = f"{post_seconds:.3f}"
        _populate_postprocessing_status(record, run_dir)
        if record.get("postprocess_loo_status") == "error":
            _exclude_from_comparison(record, "PSIS-LOO failed during postprocessing")
        _write_diagnostics(diagnostics_path, records)

        drc_returncode, drc_seconds = _run(_drc_command(root, model, material))
        record["drc_returncode"] = drc_returncode
        record["drc_wall_seconds"] = f"{drc_seconds:.3f}"
        _write_diagnostics(diagnostics_path, records)

        return record

    raise RuntimeError("Target-accept retry sequence terminated unexpectedly.")


def _run_material_comparison(root, diagnostics_path, records, material, final_records):
    eligible_models = [
        model
        for model, record in final_records.items()
        if record.get("comparison_included") is True
    ]
    comparison_name = _comparison_name(material)
    comparison_models = ";".join(eligible_models)

    if len(eligible_models) < 2:
        status = "SKIPPED_FEWER_THAN_TWO_ELIGIBLE_MODELS"
        returncode = ""
        seconds = ""
    else:
        returncode, elapsed = _run(_comparison_command(root, material, eligible_models))
        seconds = f"{elapsed:.3f}"
        status = "COMPLETE" if returncode == 0 else "COMPARISON_ERROR"

    for record in final_records.values():
        record["comparison_name"] = comparison_name
        record["comparison_models"] = comparison_models
        record["comparison_returncode"] = returncode
        record["comparison_wall_seconds"] = seconds
        record["comparison_status"] = status

    _write_diagnostics(diagnostics_path, records)
    return returncode


def _terminal_records_by_model(records, material):
    final_records = {}
    valid_models = set(_material_models(material))
    for record in records:
        if record.get("material") != material:
            continue
        if record.get("model") not in valid_models:
            continue
        if not _as_bool(record.get("terminal_attempt")):
            continue
        final_records[record["model"]] = record
    return final_records


def _run_compare_only(root, args, diagnostics_path):
    records = _read_diagnostics(diagnostics_path)
    materials = (*ALLOY_MATERIALS, "Pd100")
    comparison_failures = []
    n_materials = 0

    print(f"Rebuilding model comparisons from: {diagnostics_path}", flush=True)

    for material in materials:
        final_records = _terminal_records_by_model(records, material)
        if not final_records:
            print(f"Skipping {material}: no terminal fit records in diagnostics.", flush=True)
            continue

        n_materials += 1
        eligible_models = [
            model
            for model, record in final_records.items()
            if _as_bool(record.get("comparison_included"))
        ]
        print(
            f"\n{material}: rebuilding {_comparison_name(material)} with "
            f"{len(eligible_models)}/{len(final_records)} eligible model(s).",
            flush=True,
        )

        returncode = _run_material_comparison(
            root,
            diagnostics_path,
            records,
            material,
            final_records,
        )
        if returncode not in ("", None, 0):
            comparison_failures.append((material, returncode))
            if args.stop_on_error:
                break

    print("\n" + "=" * 80)
    print(f"Diagnostics updated at: {diagnostics_path}")
    if n_materials == 0:
        print("No material comparisons were rebuilt.")
        raise SystemExit(1)
    if comparison_failures:
        print(f"Comparison failures recorded: {len(comparison_failures)}")
        for material, returncode in comparison_failures:
            print(f"  {material}: exit {returncode}")
        raise SystemExit(1)

    print(f"Rebuilt comparisons for {n_materials} material(s).")


def _material_models(material):
    return PD_MODELS if material == "Pd100" else ALLOY_MODELS


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    individual_root = _individual_root(root)
    diagnostics_path = individual_root / "individual_grid_diagnostics.csv"

    if args.compare_only:
        _run_compare_only(root, args, diagnostics_path)
        return

    grid_started_utc = _utc_now()
    records = []
    _write_diagnostics(diagnostics_path, records)

    materials = (*ALLOY_MATERIALS, "Pd100")
    total_jobs = sum(len(_material_models(material)) for material in materials)
    print(
        f"Queued {total_jobs} individual fits. Divergences trigger target_accept retries at "
        f"{TARGET_ACCEPT_SEQUENCE}; terminal fits get full postprocessing and half-step DRC checks.",
        flush=True,
    )
    print(f"Persistent diagnostics: {diagnostics_path}", flush=True)

    completed_jobs = 0
    hard_failures = []
    command_failures = []
    stop_requested = False

    for material in materials:
        final_records = {}
        print(f"\n{'#' * 80}\nMATERIAL: {material}\n{'#' * 80}", flush=True)

        for model in _material_models(material):
            completed_jobs += 1
            print(f"\n{'=' * 80}\n[{completed_jobs}/{total_jobs}] {material} / {model}\n{'=' * 80}", flush=True)
            record = _run_one_fit(
                root,
                args,
                grid_started_utc,
                diagnostics_path,
                records,
                material,
                model,
            )
            final_records[model] = record

            if record["sampling_status"] in {"FIT_ERROR", "FAIL_DIAGNOSTICS", "FAIL_SAMPLER_HEALTH"}:
                hard_failures.append((material, model, record["sampling_status"]))

            for stage in ("fit", "postprocess", "drc"):
                returncode = record.get(f"{stage}_returncode")
                if returncode not in ("", None, 0):
                    command_failures.append((material, model, stage, returncode))
                    if args.stop_on_error:
                        stop_requested = True

            if stop_requested:
                break

        if not stop_requested:
            comparison_returncode = _run_material_comparison(
                root,
                diagnostics_path,
                records,
                material,
                final_records,
            )
            if comparison_returncode not in ("", None, 0):
                command_failures.append((material, "", "comparison", comparison_returncode))
                if args.stop_on_error:
                    stop_requested = True

        if stop_requested:
            break

    print("\n" + "=" * 80)
    print(f"Diagnostics saved to: {diagnostics_path}")
    if hard_failures:
        print(f"Hard sampler/fit failures recorded: {len(hard_failures)}")
    if command_failures:
        print(f"Command-stage failures recorded: {len(command_failures)}")
    if hard_failures or command_failures:
        raise SystemExit(1)

    print(f"Completed all {total_jobs} individual fits, postprocessing/DRC, and material comparisons.")


if __name__ == "__main__":
    main()
