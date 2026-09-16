"""Run the complete AgPd individual-material and all-material finite-rate CO fit grid."""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import yaml


ALLOY_MATERIALS = ("Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10")
ALLOY_MODELS = ("CO_LH", "CO_ER", "CO_BF", "CO_ER_LH", "CO_BF_LH", "CO_BF_ER", "CO_BF_ER_LH")
PD_MODELS = ("CO_LH", "CO_ER", "CO_ER_LH")

ALL_MATERIAL_PARAMETERIZATION = "linear_xAg"
ALL_MATERIAL_ERROR_STRUCTURE = "shared"
ALL_MATERIAL_PRIOR_MATERIAL = "Ag10Pd90"
ALL_MATERIAL_RUNS = (
    ("Full", "CO_BF_ER_LH"),
    ("Full_q1", "CO_BF_ER_LH_q1"),
    ("Full_neg", "CO_BF_ER_LH_neg"),
    ("Ag10_no_BF", "CO_BF_ER_LH_Ag10_no_BF"),
    ("Ag10_no_ER", "CO_BF_ER_LH_Ag10_no_ER"),
    ("Ag10_no_BF_q1", "CO_BF_ER_LH_Ag10_no_BF_q1"),
    ("Ag10_no_BF_neg", "CO_BF_ER_LH_Ag10_no_BF_neg"),
)
ALL_MATERIAL_COMPARISON_NAME = "all_materials_shared_models"

TARGET_ACCEPT_SEQUENCE = (0.90, 0.95, 0.99)
AGPD_DATASET_NAMES = {"base": "AgPd_COOx_basic", "maxtof": "AgPd_COOx_basic_maxtof"}
AGPD_DATA_VARIANT_ENV = "MKM_AGPD_DATA_VARIANT"

DIAGNOSTIC_FIELDS = (
    "grid_started_utc",
    "data_variant",
    "fit_scope",
    "material",
    "model",
    "display_name",
    "parameterization",
    "error_structure",
    "prior_material",
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
    "composition_returncode",
    "composition_wall_seconds",
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
        help="Rebuild comparisons from the diagnostics CSV for the selected run scope.",
    )
    parser.add_argument(
        "--all-materials-only",
        action="store_true",
        help=(
            "Run only the configured all-material linear_xAg fits. "
            "Their diagnostics are written separately under the all-material shared-error directory."
        ),
    )
    parser.add_argument(
        "--data-variant",
        choices=tuple(AGPD_DATASET_NAMES),
        default="base",
        help=(
            "AgPd processed dataset/result namespace. The default 'base' uses AgPd_COOx_basic; "
            "'maxtof' uses AgPd_COOx_basic_maxtof and is propagated to all subprocesses."
        ),
    )
    parser.add_argument(
        "--all-material-model",
        choices=tuple(display_name for display_name, _ in ALL_MATERIAL_RUNS),
        help=(
            "With --all-materials-only, run only the selected all-material fit while preserving the other "
            "all-material diagnostic records. The final comparison is rebuilt from all available terminal records."
        ),
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if args.compare_only and args.overwrite:
        parser.error("--overwrite cannot be used with --compare-only.")
    if args.all_material_model and not args.all_materials_only:
        parser.error("--all-material-model requires --all-materials-only.")
    if args.compare_only and args.all_material_model:
        parser.error("--all-material-model cannot be used with --compare-only.")
    return args


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _run(command):
    print("\n> " + " ".join(str(part) for part in command), flush=True)
    start = perf_counter()
    returncode = subprocess.run(command, check=False).returncode
    return returncode, perf_counter() - start


def _dataset_name():
    variant = os.environ.get(AGPD_DATA_VARIANT_ENV, "base")
    try:
        return AGPD_DATASET_NAMES[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown AgPd data variant '{variant}'.") from exc


def _posterior_root(root):
    return root / "results" / _dataset_name() / "posterior"


def _individual_root(root):
    return _posterior_root(root) / "individual"


def _all_material_root(root):
    return (
        _posterior_root(root)
        / "all_materials"
        / ALL_MATERIAL_PARAMETERIZATION
        / ALL_MATERIAL_ERROR_STRUCTURE
    )


def _diagnostics_path(root, all_materials_only):
    if all_materials_only:
        return _all_material_root(root) / "all_material_grid_diagnostics.csv"
    return _individual_root(root) / "individual_grid_diagnostics.csv"


def _fit_dir(root, job):
    if job["fit_scope"] == "individual":
        return _individual_root(root) / job["material"] / job["model"]
    return (
        _posterior_root(root)
        / "all_materials"
        / job["parameterization"]
        / job["error_structure"]
        / job["model"]
    )


def _fit_command(root, job, target_accept, overwrite):
    command = [
        sys.executable,
        str(root / "scripts" / "fit_agpd_posterior.py"),
        job["model"],
    ]
    if job["fit_scope"] == "individual":
        command.extend(
            [
                "--material",
                job["material"],
                "--error-structure",
                "material",
            ]
        )
    else:
        command.extend(
            [
                "--all-materials",
                "--parameterization",
                job["parameterization"],
                "--error-structure",
                job["error_structure"],
                "--prior-material",
                job["prior_material"],
            ]
        )
    command.extend(["--target-accept", f"{target_accept:.2f}"])
    if overwrite:
        command.append("--overwrite")
    return command


def _postprocess_command(root, job):
    command = [
        sys.executable,
        str(root / "scripts" / "postprocess_agpd_posterior.py"),
        job["model"],
    ]
    if job["fit_scope"] == "individual":
        command.extend(
            [
                "--material",
                job["material"],
                "--error-structure",
                "material",
            ]
        )
    else:
        command.extend(
            [
                "--all-materials",
                "--parameterization",
                job["parameterization"],
                "--error-structure",
                job["error_structure"],
                "--prior-material",
                job["prior_material"],
            ]
        )
    command.extend(["--plot-level", "full"])
    return command


def _drc_command(root, job):
    command = [
        sys.executable,
        str(root / "scripts" / "postprocess_agpd_drc.py"),
        job["model"],
    ]
    if job["fit_scope"] == "individual":
        command.extend(
            [
                "--material",
                job["material"],
                "--error-structure",
                "material",
            ]
        )
    else:
        command.extend(
            [
                "--all-materials",
                "--parameterization",
                job["parameterization"],
                "--error-structure",
                job["error_structure"],
                "--prior-material",
                job["prior_material"],
            ]
        )
    command.append("--check-half-step")
    return command


def _composition_command(root, job):
    if job["fit_scope"] != "all_materials":
        raise ValueError("Composition-parameter plots are only defined for all-material fits.")
    return [
        sys.executable,
        str(root / "scripts" / "plot_agpd_composition_parameters.py"),
        job["model"],
        "--parameterization",
        job["parameterization"],
        "--error-structure",
        job["error_structure"],
        "--prior-material",
        job["prior_material"],
    ]


def _comparison_name(material):
    return f"{material}_all_models"


def _individual_comparison_command(root, material, models):
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


def _all_material_comparison_command(root, jobs):
    command = [
        sys.executable,
        str(root / "scripts" / "compare_agpd_models.py"),
        "--comparison-name",
        ALL_MATERIAL_COMPARISON_NAME,
        "--all-materials",
        "--prior-material",
        ALL_MATERIAL_PRIOR_MATERIAL,
    ]
    for job in jobs:
        command.extend(
            [
                "--run",
                ",".join(
                    (
                        job["display_name"],
                        job["model"],
                        job["parameterization"],
                        job["error_structure"],
                    )
                ),
            ]
        )
    return command


def _individual_job(material, model):
    return {
        "fit_scope": "individual",
        "material": material,
        "model": model,
        "display_name": model,
        "parameterization": "",
        "error_structure": "material",
        "prior_material": material,
    }


def _all_material_job(display_name, model):
    return {
        "fit_scope": "all_materials",
        "material": "",
        "model": model,
        "display_name": display_name,
        "parameterization": ALL_MATERIAL_PARAMETERIZATION,
        "error_structure": ALL_MATERIAL_ERROR_STRUCTURE,
        "prior_material": ALL_MATERIAL_PRIOR_MATERIAL,
    }


def _all_material_jobs(display_name=None):
    jobs = tuple(_all_material_job(name, model) for name, model in ALL_MATERIAL_RUNS)
    if display_name is None:
        return jobs
    return tuple(job for job in jobs if job["display_name"] == display_name)


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
            f"Grid diagnostics do not exist: {path}. Run the full fit grid before using --compare-only."
        )
    with open(path, "r", newline="") as file:
        rows = list(csv.DictReader(file))
    return [{field: row.get(field, "") for field in DIAGNOSTIC_FIELDS} for row in rows]


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

    # R-hat and ESS remain warnings rather than automatic failures, matching the previous grid runner.
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


def _attempt_record(grid_started_utc, job, attempt, target_accept):
    return {field: "" for field in DIAGNOSTIC_FIELDS} | {
        "grid_started_utc": grid_started_utc,
        "data_variant": os.environ.get(AGPD_DATA_VARIANT_ENV, "base"),
        "fit_scope": job["fit_scope"],
        "material": job["material"],
        "model": job["model"],
        "display_name": job["display_name"],
        "parameterization": job["parameterization"],
        "error_structure": job["error_structure"],
        "prior_material": job["prior_material"],
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


def _run_one_fit(root, args, grid_started_utc, diagnostics_path, records, job):
    run_dir = _fit_dir(root, job)

    for attempt, target_accept in enumerate(TARGET_ACCEPT_SEQUENCE, start=1):
        record = _attempt_record(grid_started_utc, job, attempt, target_accept)
        overwrite = args.overwrite if attempt == 1 else True

        fit_returncode, fit_seconds = _run(_fit_command(root, job, target_accept, overwrite))
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
            record["status_reasons"] = f"{n_divergent} divergence(s); retrying at the next target_accept"
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

        post_returncode, post_seconds = _run(_postprocess_command(root, job))
        record["postprocess_returncode"] = post_returncode
        record["postprocess_wall_seconds"] = f"{post_seconds:.3f}"
        _populate_postprocessing_status(record, run_dir)
        if record.get("postprocess_loo_status") == "error":
            _exclude_from_comparison(record, "PSIS-LOO failed during postprocessing")
        _write_diagnostics(diagnostics_path, records)

        drc_returncode, drc_seconds = _run(_drc_command(root, job))
        record["drc_returncode"] = drc_returncode
        record["drc_wall_seconds"] = f"{drc_seconds:.3f}"
        _write_diagnostics(diagnostics_path, records)

        if job["fit_scope"] == "all_materials":
            composition_returncode, composition_seconds = _run(_composition_command(root, job))
            record["composition_returncode"] = composition_returncode
            record["composition_wall_seconds"] = f"{composition_seconds:.3f}"
            _write_diagnostics(diagnostics_path, records)

        return record

    raise RuntimeError("Target-accept retry sequence terminated unexpectedly.")


def _run_material_comparison(root, diagnostics_path, records, material, final_records):
    eligible_models = [
        model for model, record in final_records.items() if record.get("comparison_included") is True
    ]
    comparison_name = _comparison_name(material)
    comparison_models = ";".join(eligible_models)

    if len(eligible_models) < 2:
        status = "SKIPPED_FEWER_THAN_TWO_ELIGIBLE_MODELS"
        returncode = ""
        seconds = ""
    else:
        returncode, elapsed = _run(_individual_comparison_command(root, material, eligible_models))
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


def _run_all_material_comparison(root, diagnostics_path, records, final_records):
    jobs_by_model = {job["model"]: job for job in _all_material_jobs()}
    eligible_jobs = [
        jobs_by_model[model]
        for model, record in final_records.items()
        if model in jobs_by_model and record.get("comparison_included") is True
    ]
    comparison_models = ";".join(f"{job['display_name']}={job['model']}" for job in eligible_jobs)

    if len(eligible_jobs) < 2:
        status = "SKIPPED_FEWER_THAN_TWO_ELIGIBLE_MODELS"
        returncode = ""
        seconds = ""
    else:
        returncode, elapsed = _run(_all_material_comparison_command(root, eligible_jobs))
        seconds = f"{elapsed:.3f}"
        status = "COMPLETE" if returncode == 0 else "COMPARISON_ERROR"

    for record in final_records.values():
        record["comparison_name"] = ALL_MATERIAL_COMPARISON_NAME
        record["comparison_models"] = comparison_models
        record["comparison_returncode"] = returncode
        record["comparison_wall_seconds"] = seconds
        record["comparison_status"] = status

    _write_diagnostics(diagnostics_path, records)
    return returncode


def _record_scope(record):
    scope = str(record.get("fit_scope", "")).strip()
    return scope or "individual"


def _terminal_records_by_model(records, material):
    final_records = {}
    valid_models = set(_material_models(material))
    for record in records:
        if _record_scope(record) != "individual":
            continue
        if record.get("material") != material:
            continue
        if record.get("model") not in valid_models:
            continue
        if not _as_bool(record.get("terminal_attempt")):
            continue
        final_records[record["model"]] = record
    return final_records


def _terminal_all_material_records_by_model(records):
    final_records = {}
    valid_models = {model for _, model in ALL_MATERIAL_RUNS}
    for record in records:
        if _record_scope(record) != "all_materials":
            continue
        if record.get("model") not in valid_models:
            continue
        if not _as_bool(record.get("terminal_attempt")):
            continue
        final_records[record["model"]] = record
    return final_records


def _run_compare_only(root, args, diagnostics_path):
    records = _read_diagnostics(diagnostics_path)
    comparison_failures = []
    n_comparisons = 0

    print(f"Rebuilding model comparisons from: {diagnostics_path}", flush=True)

    for material in (*ALLOY_MATERIALS, "Pd100"):
        final_records = _terminal_records_by_model(records, material)
        if not final_records:
            print(f"Skipping {material}: no terminal fit records in diagnostics.", flush=True)
            continue

        eligible_models = [
            model for model, record in final_records.items() if _as_bool(record.get("comparison_included"))
        ]
        print(
            f"\n{material}: rebuilding {_comparison_name(material)} with "
            f"{len(eligible_models)}/{len(final_records)} eligible model(s).",
            flush=True,
        )
        n_comparisons += 1
        returncode = _run_material_comparison(root, diagnostics_path, records, material, final_records)
        if returncode not in ("", None, 0):
            comparison_failures.append((material, returncode))
            if args.stop_on_error:
                break

    if not (args.stop_on_error and comparison_failures):
        final_records = _terminal_all_material_records_by_model(records)
        if final_records:
            eligible_models = [
                model for model, record in final_records.items() if _as_bool(record.get("comparison_included"))
            ]
            print(
                f"\nAll materials: rebuilding {ALL_MATERIAL_COMPARISON_NAME} with "
                f"{len(eligible_models)}/{len(final_records)} eligible model(s).",
                flush=True,
            )
            n_comparisons += 1
            returncode = _run_all_material_comparison(root, diagnostics_path, records, final_records)
            if returncode not in ("", None, 0):
                comparison_failures.append(("all_materials", returncode))
        else:
            print("Skipping all-material comparison: no terminal all-material fit records in diagnostics.", flush=True)

    print("\n" + "=" * 80)
    print(f"Diagnostics updated at: {diagnostics_path}")
    if n_comparisons == 0:
        print("No model comparisons were rebuilt.")
        raise SystemExit(1)
    if comparison_failures:
        print(f"Comparison failures recorded: {len(comparison_failures)}")
        for name, returncode in comparison_failures:
            print(f"  {name}: exit {returncode}")
        raise SystemExit(1)
    print(f"Rebuilt {n_comparisons} comparison(s).")


def _material_models(material):
    return PD_MODELS if material == "Pd100" else ALLOY_MODELS


def _collect_command_failures(record):
    failures = []
    stages = ["fit", "postprocess", "drc"]
    if record.get("fit_scope") == "all_materials":
        stages.append("composition")
    for stage in stages:
        returncode = record.get(f"{stage}_returncode")
        if returncode not in ("", None, 0):
            failures.append((stage, returncode))
    return failures


def main():
    args = parse_args()
    os.environ[AGPD_DATA_VARIANT_ENV] = args.data_variant
    root = Path(__file__).resolve().parents[1]
    diagnostics_path = _diagnostics_path(root, args.all_materials_only)

    if args.compare_only:
        _run_compare_only(root, args, diagnostics_path)
        return

    grid_started_utc = _utc_now()
    selected_all_material_jobs = _all_material_jobs(args.all_material_model)

    if args.all_materials_only and args.all_material_model and diagnostics_path.exists():
        records = _read_diagnostics(diagnostics_path)
        selected_models = {job["model"] for job in selected_all_material_jobs}
        records = [
            record
            for record in records
            if not (
                _record_scope(record) == "all_materials"
                and record.get("model") in selected_models
            )
        ]
    else:
        records = []
    _write_diagnostics(diagnostics_path, records)

    materials = () if args.all_materials_only else (*ALLOY_MATERIALS, "Pd100")
    individual_jobs = sum(len(_material_models(material)) for material in materials)
    all_material_jobs = len(selected_all_material_jobs)
    total_jobs = individual_jobs + all_material_jobs
    print(
        f"Queued {individual_jobs} individual fits plus {all_material_jobs} all-material fits "
        f"({total_jobs} total). Divergences trigger target_accept retries at {TARGET_ACCEPT_SEQUENCE}; "
        "terminal fits get full postprocessing and half-step DRC checks; all-material fits also get "
        "composition-parameter plots.",
        flush=True,
    )
    print(f"Data variant: {args.data_variant} ({_dataset_name()})", flush=True)
    print(f"Persistent diagnostics: {diagnostics_path}", flush=True)

    completed_jobs = 0
    hard_failures = []
    command_failures = []
    stop_requested = False

    for material in materials:
        final_records = {}
        print(f"\n{'#' * 80}\nINDIVIDUAL MATERIAL: {material}\n{'#' * 80}", flush=True)

        for model in _material_models(material):
            completed_jobs += 1
            job = _individual_job(material, model)
            print(
                f"\n{'=' * 80}\n[{completed_jobs}/{total_jobs}] {material} / {model}\n{'=' * 80}",
                flush=True,
            )
            record = _run_one_fit(root, args, grid_started_utc, diagnostics_path, records, job)
            final_records[model] = record

            if record["sampling_status"] in {"FIT_ERROR", "FAIL_DIAGNOSTICS", "FAIL_SAMPLER_HEALTH"}:
                hard_failures.append(("individual", material, model, record["sampling_status"]))

            for stage, returncode in _collect_command_failures(record):
                command_failures.append(("individual", material, model, stage, returncode))
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
                command_failures.append(("individual", material, "", "comparison", comparison_returncode))
                if args.stop_on_error:
                    stop_requested = True

        if stop_requested:
            break

    all_material_final_records = {}
    if not stop_requested:
        print(f"\n{'#' * 80}\nALL-MATERIAL FITS\n{'#' * 80}", flush=True)
        for job in selected_all_material_jobs:
            completed_jobs += 1
            print(
                f"\n{'=' * 80}\n[{completed_jobs}/{total_jobs}] "
                f"{job['display_name']} ({job['model']})\n{'=' * 80}",
                flush=True,
            )
            record = _run_one_fit(root, args, grid_started_utc, diagnostics_path, records, job)
            all_material_final_records[job["model"]] = record

            if record["sampling_status"] in {"FIT_ERROR", "FAIL_DIAGNOSTICS", "FAIL_SAMPLER_HEALTH"}:
                hard_failures.append(("all_materials", "", job["display_name"], record["sampling_status"]))

            for stage, returncode in _collect_command_failures(record):
                command_failures.append(("all_materials", "", job["display_name"], stage, returncode))
                if args.stop_on_error:
                    stop_requested = True

            if stop_requested:
                break

    if not stop_requested and all_material_final_records:
        comparison_records = all_material_final_records
        if args.all_materials_only and args.all_material_model:
            comparison_records = _terminal_all_material_records_by_model(records)
        comparison_returncode = _run_all_material_comparison(
            root,
            diagnostics_path,
            records,
            comparison_records,
        )
        if comparison_returncode not in ("", None, 0):
            command_failures.append(("all_materials", "", "", "comparison", comparison_returncode))

    print("\n" + "=" * 80)
    print(f"Diagnostics saved to: {diagnostics_path}")
    if hard_failures:
        print(f"Hard sampler/fit failures recorded: {len(hard_failures)}")
    if command_failures:
        print(f"Command-stage failures recorded: {len(command_failures)}")
    if hard_failures or command_failures:
        raise SystemExit(1)

    if args.all_materials_only and args.all_material_model:
        print(
            f"Completed selected all-material fit {args.all_material_model} with shared error structure, "
            "including full postprocessing, DRC, composition plots, and comparison rebuild from available runs."
        )
    elif args.all_materials_only:
        print(
            f"Completed all {all_material_jobs} all-material fits with shared error structure, "
            "including full postprocessing, DRC, composition plots, and model comparison."
        )
    else:
        print(
            f"Completed all {individual_jobs} individual fits and {all_material_jobs} all-material fits, "
            "including postprocessing, DRC, requested comparisons, and all-material composition plots."
        )


if __name__ == "__main__":
    main()
