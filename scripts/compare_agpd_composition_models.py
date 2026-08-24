"""Compare saved postprocessing products from two AgPd composition posterior models."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from mkm.models.agpd_basic import available_agpd_composition_models
from mkm.project_paths import ProjectPaths


DEFAULT_REFERENCE_MODEL = "BF_LH"
DEFAULT_CANDIDATE_MODEL = "CO_BF_ER_LH"
DEFAULT_COMPOSITION_MODEL = "shared"
DEFAULT_LIKELIHOOD = "setup_intercept"

_ALIGNMENT_CANDIDATES = (
    "observation_id",
    "material",
    "electrolyte_concentration_M",
    "CO_mole_fraction",
    "replicate",
    "analysis_grid_index",
    "E_V_SHE",
)


def parse_args():
    models = available_agpd_composition_models()

    parser = ArgumentParser()
    parser.add_argument("--reference-model", choices=models, default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--candidate-model", choices=models, default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--composition-model", default=DEFAULT_COMPOSITION_MODEL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default=DEFAULT_LIKELIHOOD)
    return parser.parse_args()


def _postprocessing_dir(paths, composition_model, likelihood_name, model_name):
    posterior_dir = paths.agpd_composition_posterior_output_dir(
        composition_model=composition_model,
        model_name=model_name,
        likelihood_name=likelihood_name,
    )
    return posterior_dir / "postprocessing"


def _require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required postprocessing product not found: {path}")
    return path


def _load_products(paths, composition_model, likelihood_name, model_name):
    root = _postprocessing_dir(paths, composition_model, likelihood_name, model_name)

    return {
        "root": root,
        "loo": pd.read_parquet(_require_file(root / "derived" / "loo_pointwise.parquet")),
        "residual": pd.read_csv(_require_file(root / "tables" / "residual_summary_by_material.csv")),
        "noise": pd.read_csv(_require_file(root / "tables" / "noise_summary_by_material.csv")),
        "observable": pd.read_csv(
            _require_file(root / "tables" / "experimental_observable_summary_by_material.csv")
        ),
    }


def _alignment_columns(reference, candidate):
    if "observation_id" in reference.columns and "observation_id" in candidate.columns:
        if reference["observation_id"].is_unique and candidate["observation_id"].is_unique:
            return ["observation_id"]

    columns = [
        column
        for column in _ALIGNMENT_CANDIDATES
        if column != "observation_id" and column in reference.columns and column in candidate.columns
    ]

    if not columns:
        raise ValueError("Could not identify common observation metadata columns for paired LOO comparison.")

    return columns


def build_paired_loo(reference, candidate, reference_model, candidate_model):
    key_columns = _alignment_columns(reference, candidate)

    reference_columns = key_columns + ["elpd_loo"]
    candidate_columns = key_columns + ["elpd_loo"]

    ref = reference[reference_columns].copy()
    cand = candidate[candidate_columns].copy()

    if len(ref) != len(cand):
        raise ValueError(
            f"Paired LOO requires equal observation counts; got {len(ref)} and {len(cand)}."
        )

    merged = ref.merge(
        cand,
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )

    if len(merged) != len(ref):
        raise ValueError(
            "Paired LOO observations do not align one-to-one between models."
        )

    merged["reference_model"] = reference_model
    merged["candidate_model"] = candidate_model
    merged["delta_elpd"] = (
        merged["elpd_loo_candidate"] - merged["elpd_loo_reference"]
    )

    if "material" not in merged.columns and "material" in reference.columns:
        material_map = reference[key_columns + ["material"]].drop_duplicates()
        merged = merged.merge(material_map, on=key_columns, how="left", validate="one_to_one")

    return merged


def _paired_elpd_summary(frame, reference_model, candidate_model, scope, material=None):
    delta = frame["delta_elpd"].to_numpy(dtype=float)
    n = len(delta)

    if n < 2:
        se_delta = np.nan
    else:
        se_delta = float(np.sqrt(n * np.var(delta, ddof=1)))

    reference_elpd = float(frame["elpd_loo_reference"].sum())
    candidate_elpd = float(frame["elpd_loo_candidate"].sum())
    delta_elpd = float(delta.sum())

    return {
        "scope": scope,
        "material": material,
        "reference_model": reference_model,
        "candidate_model": candidate_model,
        "n_observations": n,
        "reference_elpd": reference_elpd,
        "candidate_elpd": candidate_elpd,
        "delta_elpd_candidate_minus_reference": delta_elpd,
        "se_delta_elpd": se_delta,
        "delta_elpd_over_se": delta_elpd / se_delta if np.isfinite(se_delta) and se_delta > 0 else np.nan,
        "mean_pointwise_delta_elpd": float(np.mean(delta)),
        "fraction_points_favoring_candidate": float(np.mean(delta > 0)),
    }


def summarize_paired_loo(paired, reference_model, candidate_model):
    records = [
        _paired_elpd_summary(
            paired,
            reference_model=reference_model,
            candidate_model=candidate_model,
            scope="overall",
        )
    ]

    if "material" in paired.columns:
        for material, frame in paired.groupby("material", sort=False):
            records.append(
                _paired_elpd_summary(
                    frame,
                    reference_model=reference_model,
                    candidate_model=candidate_model,
                    scope="material",
                    material=material,
                )
            )

    return pd.DataFrame(records)


def compare_material_table(reference, candidate, key_columns, metrics):
    ref_columns = key_columns + [metric for metric in metrics if metric in reference.columns]
    cand_columns = key_columns + [metric for metric in metrics if metric in candidate.columns]

    shared_metrics = [
        metric for metric in metrics if metric in ref_columns and metric in cand_columns
    ]
    if not shared_metrics:
        raise ValueError(f"No requested comparison metrics are shared for keys {key_columns}.")

    merged = reference[ref_columns].merge(
        candidate[cand_columns],
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )

    for metric in shared_metrics:
        ref_name = f"{metric}_reference"
        cand_name = f"{metric}_candidate"
        merged[f"{metric}_difference_candidate_minus_reference"] = (
            merged[cand_name] - merged[ref_name]
        )

        ref_values = merged[ref_name].to_numpy(dtype=float)
        cand_values = merged[cand_name].to_numpy(dtype=float)
        ratio = np.full(len(merged), np.nan, dtype=float)
        np.divide(cand_values, ref_values, out=ratio, where=ref_values != 0)
        merged[f"{metric}_ratio_candidate_over_reference"] = ratio

    return merged


def main():
    args = parse_args()

    if args.reference_model == args.candidate_model:
        raise ValueError("Reference and candidate models must be different.")

    paths = ProjectPaths.discover(__file__)

    reference = _load_products(
        paths,
        composition_model=args.composition_model,
        likelihood_name=args.likelihood,
        model_name=args.reference_model,
    )
    candidate = _load_products(
        paths,
        composition_model=args.composition_model,
        likelihood_name=args.likelihood,
        model_name=args.candidate_model,
    )

    paired_loo = build_paired_loo(
        reference=reference["loo"],
        candidate=candidate["loo"],
        reference_model=args.reference_model,
        candidate_model=args.candidate_model,
    )
    loo_summary = summarize_paired_loo(
        paired_loo,
        reference_model=args.reference_model,
        candidate_model=args.candidate_model,
    )

    residual_metrics = (
        "mechanism_residual_mean",
        "mechanism_residual_rms",
        "conditional_residual_mean",
        "conditional_residual_rms",
        "median_abs_standardized_residual",
        "predictive_95_hdi_coverage",
        "median_lag1_residual_correlation",
        "median_abs_residual_slope_per_V",
        "median_shared_squared_residual_fraction",
    )
    residual_comparison = compare_material_table(
        reference["residual"],
        candidate["residual"],
        key_columns=["material"],
        metrics=residual_metrics,
    )

    noise_comparison = compare_material_table(
        reference["noise"],
        candidate["noise"],
        key_columns=["material", "variable"],
        metrics=("mean", "sd", "median", "hdi95_lower", "hdi95_upper"),
    )

    observable_metrics = (
        "median_abs_residual",
        "median_abs_standardized_residual",
        "posterior_95_hdi_contains_experimental_mean",
    )
    observable_comparison = compare_material_table(
        reference["observable"],
        candidate["observable"],
        key_columns=["material", "observable"],
        metrics=observable_metrics,
    )

    reference_posterior_dir = reference["root"].parent
    candidate_posterior_dir = candidate["root"].parent

    if reference_posterior_dir.parent != candidate_posterior_dir.parent:
        raise ValueError("Composition model posteriors do not share the same comparison parent directory.")

    output_dir = reference_posterior_dir.parent / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_loo.to_parquet(output_dir / "paired_loo_pointwise.parquet", index=False)
    loo_summary.to_csv(output_dir / "paired_loo_summary.csv", index=False)
    residual_comparison.to_csv(output_dir / "residual_comparison_by_material.csv", index=False)
    noise_comparison.to_csv(output_dir / "noise_comparison_by_material.csv", index=False)
    observable_comparison.to_csv(output_dir / "observable_comparison_by_material.csv", index=False)

    print(
        f"\nAgPd composition model comparison: "
        f"{args.candidate_model} vs {args.reference_model}"
    )
    print(f"Composition parameterization: {args.composition_model}")
    print(f"Likelihood: {args.likelihood}")

    print("\n=== PAIRED PSIS-LOO ===")
    print(loo_summary.to_string(index=False))

    print("\n=== CONDITIONAL RESIDUAL RMS ===")
    columns = [
        "material",
        "conditional_residual_rms_reference",
        "conditional_residual_rms_candidate",
        "conditional_residual_rms_difference_candidate_minus_reference",
        "conditional_residual_rms_ratio_candidate_over_reference",
    ]
    print(residual_comparison[columns].to_string(index=False))

    print("\n=== MATERIAL RESIDUAL SCALE ===")
    residual_noise = noise_comparison.loc[
        noise_comparison["variable"] == "sigma_ln_rate_material"
    ]
    columns = [
        "material",
        "median_reference",
        "median_candidate",
        "median_difference_candidate_minus_reference",
        "median_ratio_candidate_over_reference",
    ]
    print(residual_noise[columns].to_string(index=False))

    print(
        "\nInterpretation note: paired observation-wise LOO compares predictive performance "
        "for observations from compositions still represented in the fitted data. It is not "
        "leave-one-material-out validation and is not a mechanism probability."
    )
    print(f"\nSaved comparison to: {output_dir}")


if __name__ == "__main__":
    main()
