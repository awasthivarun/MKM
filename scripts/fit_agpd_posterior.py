from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import arviz as az
import numpy as np
import pandas as pd
import yaml

from mkm.inference.model import build_pymc_model
from mkm.inference.posterior import (
    add_log_likelihood,
    compute_posterior_deterministics,
    sample_posterior,
    summarize_sampler_health,
)
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.agpd_basic import available_agpd_models, build_agpd_mechanism


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis" / "AgPd_COOx_basic_selected.parquet"
CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
OUTPUT_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"

DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.90
RANDOM_SEED = 20260821


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    selected = pd.read_parquet(DATA_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )
    inputs = build_model_input_arrays(model_data)

    mechanism = build_agpd_mechanism(model_name=model_name, material=MATERIAL, config=config)

    built = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        sigma_prior_median=config["likelihood"]["sigma_prior_median"],
        sigma_prior_log_sd=config["likelihood"]["sigma_prior_log_sd"],
    )

    output_dir = OUTPUT_ROOT / MATERIAL / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{MATERIAL}: {model_name}")
    print(f"Sampler: nutpie / numba")
    print(f"Chains: {CHAINS}, tune: {TUNE}, draws: {DRAWS}, target_accept: {TARGET_ACCEPT}")

    start = perf_counter()

    idata = sample_posterior(
        built,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=CORES,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED,
        nuts_sampler="nutpie",
        backend="numba",
        compute_convergence_checks=False,
    )

    sampling_seconds = perf_counter() - start

    idata.to_netcdf(output_dir / "posterior_free.nc", engine="h5netcdf")

    health = summarize_sampler_health(idata)
    free_var_names = [rv.name for rv in built.model.free_RVs]

    diagnostics = az.summary(
        idata,
        var_names=free_var_names,
        kind="diagnostics",
        round_to=None,
    )

    diagnostics.to_csv(output_dir / "sampler_diagnostics.csv")

    print(f"\nSampling wall time: {sampling_seconds:.2f} s")
    print(f"Divergences: {health.divergences}")
    print(f"Mean tree depth: {health.mean_tree_depth}")
    print(f"Max tree depth: {health.max_tree_depth}")
    print(f"Maximum R-hat: {diagnostics['r_hat'].max():.4f}")
    print(f"Minimum bulk ESS: {diagnostics['ess_bulk'].min():.1f}")
    print(f"Minimum tail ESS: {diagnostics['ess_tail'].min():.1f}")

    deterministic_names = ["ln_rate_model", *built.mechanism_result.pointwise]

    posterior = compute_posterior_deterministics(
        idata,
        built,
        var_names=deterministic_names,
        backend="numba",
        progressbar=True,
    )
    idata.posterior = posterior

    for name in deterministic_names:
        if not np.all(np.isfinite(np.asarray(posterior[name]))):
            raise RuntimeError(f"Posterior deterministic '{name}' contains non-finite values.")

    idata = add_log_likelihood(
        idata,
        built,
        backend="numba",
        progressbar=True,
    )

    idata.to_netcdf(output_dir / "posterior.nc", engine="h5netcdf")

    print(f"Posterior deterministics finite: {len(deterministic_names)} variables")
    print("Log likelihood: OK")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()