"""Compare completed AgPd posterior runs with PSIS-LOO."""

from argparse import ArgumentParser

import pandas as pd

from mkm.postprocessing.model_comparison import build_loo_model_comparison
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification
from mkm.workflows.agpd_posterior import load_agpd_posterior_run


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--material", default="Ag10Pd90")
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SPEC",
        help=(
            "Repeat for each candidate. Individual syntax: LABEL,MODEL. "
            "All-material syntax: LABEL,MODEL,PARAMETERIZATION,ERROR_STRUCTURE."
        ),
    )
    return parser.parse_args()


def _parse_run(specification, all_materials):
    fields = [field.strip() for field in specification.split(",")]
    expected = 4 if all_materials else 2
    if len(fields) != expected or any(not field for field in fields):
        raise ValueError(
            f"Run specification '{specification}' must contain {expected} comma-separated fields."
        )

    if all_materials:
        label, model, parameterization, error_structure = fields
    else:
        label, model = fields
        parameterization = "shared"
        error_structure = "material"
    return label, model, parameterization, error_structure


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)

    parsed = [_parse_run(value, args.all_materials) for value in args.run]
    labels = [value[0] for value in parsed]
    if len(parsed) < 2:
        raise ValueError("At least two posterior runs are required for comparison.")
    if len(set(labels)) != len(labels):
        raise ValueError("Comparison run labels must be unique.")

    runs = {}
    reference_observations = None
    for label, model, parameterization, error_structure in parsed:
        fit_specification = resolve_agpd_fit_specification(
            config,
            model_name=model,
            all_materials=args.all_materials,
            material=args.material,
            parameterization=parameterization,
            error_structure=error_structure,
            prior_material=args.prior_material,
        )
        run = load_agpd_posterior_run(
            paths,
            config,
            fit_specification,
            reconstruct_pointwise=False,
            progressbar=False,
        )

        observations = run.model_data.observations.reset_index(drop=True)
        if reference_observations is None:
            reference_observations = observations
        elif not reference_observations.equals(observations):
            raise ValueError(
                f"Run '{label}' does not contain the same ordered observations as the first run."
            )
        runs[label] = run.inference_data

    comparison = build_loo_model_comparison(runs, reference_observations)
    output_dir = paths.agpd_model_comparison_dir(
        fit_scope="all_materials" if args.all_materials else "individual",
        material=None if args.all_materials else args.material,
        comparison_name=args.comparison_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.compare_table.to_csv(output_dir / "loo_model_comparison.csv", index=False)
    comparison.difference_summary.to_csv(
        output_dir / "loo_difference_summary.csv",
        index=False,
    )
    comparison.pointwise_differences.to_parquet(
        output_dir / "loo_pointwise_differences.parquet",
        index=False,
    )

    run_manifest = pd.DataFrame(
        [
            {
                "label": label,
                "model": model,
                "parameterization": parameterization if args.all_materials else None,
                "error_structure": error_structure,
            }
            for label, model, parameterization, error_structure in parsed
        ]
    )
    run_manifest.to_csv(output_dir / "comparison_runs.csv", index=False)
    print(comparison.compare_table.to_string(index=False))
    print(f"Comparison saved to: {output_dir}")


if __name__ == "__main__":
    main()
