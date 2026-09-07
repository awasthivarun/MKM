"""Run the complete AgPd individual-material finite-rate CO model grid."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


ALLOY_MATERIALS = ("Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10")
ALLOY_MODELS = ("CO_LH", "CO_ER", "CO_BF", "CO_ER_LH", "CO_BF_LH", "CO_BF_ER", "CO_BF_ER_LH")
PD_MODELS = ("CO_LH", "CO_ER", "CO_ER_LH")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _run(command):
    print("\n> " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, check=False).returncode


def _commands(root, model, material, overwrite):
    fit = [
        sys.executable,
        str(root / "scripts" / "fit_agpd_posterior.py"),
        model,
        "--material",
        material,
        "--error-structure",
        "material",
    ]
    if overwrite:
        fit.append("--overwrite")

    postprocess = [
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
    drc = [
        sys.executable,
        str(root / "scripts" / "postprocess_agpd_drc.py"),
        model,
        "--material",
        material,
        "--error-structure",
        "material",
    ]
    return (("fit", fit), ("postprocess", postprocess), ("drc", drc))


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = [(material, model) for material in ALLOY_MATERIALS for model in ALLOY_MODELS]
    jobs.extend(("Pd100", model) for model in PD_MODELS)

    print(f"Queued {len(jobs)} individual fits; each successful fit gets full postprocessing and DRC.")
    failures = []

    for index, (material, model) in enumerate(jobs, start=1):
        print(f"\n{'=' * 80}\n[{index}/{len(jobs)}] {material} / {model}\n{'=' * 80}", flush=True)
        fit_failed = False
        for stage, command in _commands(root, model, material, args.overwrite):
            if fit_failed:
                break
            returncode = _run(command)
            if returncode == 0:
                continue

            failures.append((material, model, stage, returncode))
            print(f"FAILED: {material} / {model} / {stage} returned {returncode}.", flush=True)
            if stage == "fit":
                fit_failed = True
            if args.stop_on_error:
                break

        if failures and args.stop_on_error:
            break

    print("\n" + "=" * 80)
    if failures:
        print(f"Completed with {len(failures)} failed stage(s):")
        for material, model, stage, returncode in failures:
            print(f"  {material} / {model} / {stage}: exit {returncode}")
        raise SystemExit(1)

    print(f"Completed all {len(jobs)} individual fits and postprocessing stages successfully.")


if __name__ == "__main__":
    main()
