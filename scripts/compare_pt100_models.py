"""Compare completed Pt100 models on the same observations using PSIS-LOO."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from mkm.models.ptru_registry import available_ptru_models
from mkm.postprocessing.model_comparison import build_loo_model_comparison
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_posterior import load_pt100_posterior_run


def parse_run(value):
    fields = [part.strip() for part in value.split(",")]
    if len(fields) != 2 or not all(fields):
        raise ValueError("Each --run must be LABEL,MODEL (for example, LH,CO_LH).")
    label, model = fields
    if model not in available_ptru_models():
        raise ValueError(f"Unsupported Pt100 model: {model}")
    return label, model


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL,MODEL")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = [parse_run(value) for value in args.run]
    if len(candidates) < 2 or len({label for label, _ in candidates}) != len(candidates):
        raise ValueError("Comparison requires at least two runs with distinct labels.")
    if len({model for _, model in candidates}) != len(candidates):
        raise ValueError("Comparison requires distinct model mechanisms.")

    paths = ProjectPaths.discover(__file__)
    inference_data = {}
    observations = None
    for label, model in candidates:
        run = load_pt100_posterior_run(
            paths, args.medium, model, reconstruct_pointwise=False, progressbar=False,
        )
        current = run.model_data.observations.reset_index(drop=True)
        if observations is None:
            observations = current
        elif not observations.equals(current):
            raise ValueError(f"Run '{label}' does not have the same ordered observations.")
        inference_data[label] = run.inference_data

    comparison = build_loo_model_comparison(inference_data, observations)
    output_dir = paths.ptru_model_comparison_dir(args.medium, args.comparison_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.compare_table.to_csv(output_dir / "loo_model_comparison.csv", index=False)
    comparison.difference_summary.to_csv(output_dir / "loo_difference_summary.csv", index=False)
    comparison.pointwise_differences.to_parquet(output_dir / "loo_pointwise_differences.parquet", index=False)
    pd.DataFrame(candidates, columns=["label", "model"]).to_csv(output_dir / "comparison_runs.csv", index=False)

    diagnostic_rows = []
    for label, loo in comparison.loo_results.items():
        pareto_k = np.asarray(loo.pareto_k, dtype=float).reshape(-1)
        threshold = float(loo.good_k)
        diagnostic_rows.append({
            "label": label,
            "good_k": threshold,
            "max_pareto_k": float(np.max(pareto_k)),
            "n_above_good_k": int(np.sum(pareto_k > threshold)),
            "warning": bool(loo.warning),
        })
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "loo_diagnostics.csv", index=False)

    print(comparison.compare_table.to_string(index=False))
    print(f"Comparison saved to: {output_dir}")
    print("PSIS-LOO here holds out individual observations, not entire correlated potential curves.")


if __name__ == "__main__":
    main()
