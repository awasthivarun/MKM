"""Compare PtRu posterior runs with identical observations but potentially different model specifications."""

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
    parser.add_argument("--exclude-materials", nargs="*", default=(),
                        help="All-material only: the same exclusions must apply to every candidate.")
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL,MODEL[,PARAMETERIZATION,ERROR_STRUCTURE]",
                        help="Repeat per candidate. Four-field form overrides the global defaults for that run.")
    return parser.parse_args()


def parse_candidates(values, parameterization="shared", error_structure="shared", all_materials=False):
    candidates = []
    for value in values:
        fields = tuple(part.strip() for part in value.split(","))
        if len(fields) == 2:
            label, model_name = fields
            candidate_parameterization, candidate_errors = parameterization, error_structure
        elif len(fields) == 4 and all_materials:
            label, model_name, candidate_parameterization, candidate_errors = fields
        else:
            raise ValueError("Use LABEL,MODEL or pooled LABEL,MODEL,PARAMETERIZATION,ERROR_STRUCTURE.")
        if (not all(fields) or model_name not in available_ptru_models()
                or candidate_parameterization not in ("shared", "independent", "linear_xRu")
                or candidate_errors not in ("shared", "material")):
            raise ValueError(f"Invalid PtRu run specification: {value}")
        candidates.append((label, model_name, candidate_parameterization, candidate_errors))
    if len(candidates) < 2 or len({entry[0] for entry in candidates}) != len(candidates):
        raise ValueError("Model comparison requires two or more unique run labels.")
    return candidates


def main():
    args = parse_args()
    candidates = parse_candidates(args.run, args.parameterization, args.error_structure, args.all_materials)
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    inference_data = {}
    observations = None
    for label, model_name, parameterization, error_structure in candidates:
        spec = resolve_ptru_fit_specification(
            config, args.medium, model_name, material=args.material,
            all_materials=args.all_materials, parameterization=parameterization,
            error_structure=error_structure, excluded_materials=args.exclude_materials,
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
        args.medium, args.comparison_name, material=args.material,
        fit_scope="all_materials" if args.all_materials else "individual",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.compare_table.to_csv(output_dir / "loo_model_comparison.csv", index=False)
    comparison.difference_summary.to_csv(output_dir / "loo_difference_summary.csv", index=False)
    comparison.pointwise_differences.to_parquet(output_dir / "loo_pointwise_differences.parquet", index=False)
    pd.DataFrame([{
        "label": label, "model": model, "parameterization": parameterization if args.all_materials else None,
        "error_structure": errors, "excluded_materials": ",".join(args.exclude_materials),
    } for label, model, parameterization, errors in candidates]).to_csv(
        output_dir / "comparison_runs.csv", index=False,
    )
    rows = []
    for label, loo in comparison.loo_results.items():
        pareto_k = np.asarray(loo.pareto_k, dtype=float).reshape(-1)
        threshold = float(loo.good_k)
        rows.append({
            "label": label, "good_k": threshold, "max_pareto_k": float(np.max(pareto_k)),
            "n_above_good_k": int(np.sum(pareto_k > threshold)), "warning": bool(loo.warning),
        })
    pd.DataFrame(rows).to_csv(output_dir / "loo_diagnostics.csv", index=False)
    print(comparison.compare_table.to_string(index=False))
    print(f"Comparison saved to: {output_dir}")
    print("PSIS-LOO holds out individual observations, not entire correlated potential curves.")


if __name__ == "__main__":
    main()
