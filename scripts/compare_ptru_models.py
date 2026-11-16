"""Compare PtRu model posteriors from the same material set and electrolyte medium."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from mkm.models.ptru import available_ptru_models
from mkm.postprocessing.model_comparison import build_loo_model_comparison
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import load_ptru_posterior_run


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--material", default="Pt100")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", choices=("shared", "independent", "linear_xRu"), default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--exclude-materials", nargs="*", default=())
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL,MODEL")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = []
    for value in args.run:
        parts = tuple(part.strip() for part in value.split(","))
        if len(parts) != 2 or not all(parts) or parts[1] not in available_ptru_models():
            raise ValueError("Each --run requires LABEL,MODEL (e.g., LH,CO_LH).")
        candidates.append(parts)
    if len(candidates) < 2 or len({x[0] for x in candidates}) != len(candidates):
        raise ValueError("Model comparison requires two or more distinct labels.")
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    inference_data = {}
    observations = None
    for label, model_name in candidates:
        spec = resolve_ptru_fit_specification(
            config, args.medium, model_name, material=args.material,
            all_materials=args.all_materials, parameterization=args.parameterization,
            error_structure=args.error_structure, excluded_materials=args.exclude_materials,
        )
        run = load_ptru_posterior_run(paths, spec, reconstruct_pointwise=False, progressbar=False)
        current = run.model_data.observations.reset_index(drop=True)
        if observations is None:
            observations = current
        elif not observations.equals(current):
            raise ValueError(f"Run '{label}' has a different ordered observation dataset.")
        inference_data[label] = run.inference_data
    comparison = build_loo_model_comparison(inference_data, observations)
    output_dir = paths.ptru_model_comparison_dir(
        args.medium, args.comparison_name,
        material=args.material, fit_scope="all_materials" if args.all_materials else "individual",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.compare_table.to_csv(output_dir / "loo_model_comparison.csv", index=False)
    comparison.difference_summary.to_csv(output_dir / "loo_difference_summary.csv", index=False)
    comparison.pointwise_differences.to_parquet(output_dir / "loo_pointwise_differences.parquet", index=False)
    pd.DataFrame(candidates, columns=["label", "model"]).to_csv(output_dir / "comparison_runs.csv", index=False)
    rows = []
    for label, loo in comparison.loo_results.items():
        pareto_k = np.asarray(loo.pareto_k, dtype=float).reshape(-1)
        threshold = float(loo.good_k)
        rows.append({
            "label": label, "good_k": threshold,
            "max_pareto_k": float(np.max(pareto_k)),
            "n_above_good_k": int(np.sum(pareto_k > threshold)),
            "warning": bool(loo.warning),
        })
    pd.DataFrame(rows).to_csv(output_dir / "loo_diagnostics.csv", index=False)
    print(comparison.compare_table.to_string(index=False))
    print(f"Comparison saved to: {output_dir}")
    print("PSIS-LOO here holds out individual observations, not entire correlated potential curves.")


if __name__ == "__main__":
    main()
