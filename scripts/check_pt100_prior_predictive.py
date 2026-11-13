"""Run Pt100 prior-predictive checks without sampling a posterior."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from mkm.inference.prior_predictive import sample_prior_predictive, summarize_prior_predictive
from mkm.models.ptru_registry import available_ptru_models
from mkm.postprocessing.diagnostics import build_physical_checks
from mkm.project_paths import ProjectPaths
from mkm.provenance import write_run_metadata
from mkm.workflows.pt100_fit import build_pt100_fit
from mkm.workflows.ptru_posterior import build_pt100_run_metadata


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--save-draws", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.draws < 2:
        raise ValueError("--draws must be at least 2.")

    paths = ProjectPaths.discover(__file__)
    fit = build_pt100_fit(paths, args.medium, args.model)
    prior = sample_prior_predictive(fit.built_model, draws=args.draws, random_seed=args.random_seed)
    parameter_names = tuple(variable.name for variable in fit.built_model.model.free_RVs)
    summary = summarize_prior_predictive(prior, fit.model_data, parameter_names=parameter_names)

    output_dir = paths.ptru_prior_predictive_output_dir(args.medium, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.parameter_summary.to_csv(output_dir / "prior_parameters.csv", index=False)
    summary.model_point_summary.to_parquet(output_dir / "prior_model_points.parquet", index=False)
    summary.observation_summary.to_parquet(output_dir / "prior_observations.parquet", index=False)
    build_physical_checks(prior.prior).to_csv(output_dir / "prior_physical_checks.csv", index=False)

    ln_rate = np.asarray(prior.prior["ln_rate_model"], dtype=float)
    pd.DataFrame([{
        "n_draw_values": ln_rate.size,
        "n_nonfinite_ln_rate": int(np.sum(~np.isfinite(ln_rate))),
        "n_ln_rate_above_exp_range": int(np.sum(ln_rate > np.log(np.finfo(float).max))),
    }]).to_csv(output_dir / "prior_numerical_checks.csv", index=False)

    metadata = build_pt100_run_metadata(
        paths, fit, sampler={"draws": args.draws, "random_seed": args.random_seed},
    )
    metadata["analysis"] = "prior_predictive"
    write_run_metadata(metadata, output_dir / "run_metadata.yaml")
    if args.save_draws:
        prior.to_netcdf(output_dir / "prior_predictive.nc", engine="h5netcdf")

    print(f"Prior-predictive summaries saved to: {output_dir}")
    print("Check rate ranges, coverages, and finite-value counts before fitting.")


if __name__ == "__main__":
    main()
