from pathlib import Path
from time import perf_counter

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
from mkm.models.agpd_basic import build_agpd_mechanism


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis" / "AgPd_COOx_basic_selected.parquet"
CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
OUTPUT_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior_smoke"

MATERIAL = "Ag10Pd90"
MODELS = ["CO_BF_ER_LH"]

DRAWS = 100
TUNE = 500
CHAINS = 2
CORES = 2
TARGET_ACCEPT = 0.90
RANDOM_SEED = 20260821

NUTS_SAMPLER = "nutpie"
BACKEND = "numba"


def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    selected = pd.read_parquet(DATA_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )
    inputs = build_model_input_arrays(model_data)

    for model_name in MODELS:
        print(f"\n{MATERIAL}: {model_name}")

        mechanism = build_agpd_mechanism(model_name=model_name, material=MATERIAL, config=config)
        built = build_pymc_model(
            inputs=inputs,
            mechanism=mechanism,
            sigma_prior_median=config["likelihood"]["sigma_prior_median"],
            sigma_prior_log_sd=config["likelihood"]["sigma_prior_log_sd"],
        )

        start = perf_counter()

        idata = sample_posterior(
            built,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            nuts_sampler=NUTS_SAMPLER,
            backend=BACKEND,
            compute_convergence_checks=False,
        )

        sampling_seconds = perf_counter() - start
        health = summarize_sampler_health(idata)

        print(f"Sampling wall time: {sampling_seconds:.2f} s")
        print(f"Divergences: {health.divergences}")
        print(f"Mean tree depth: {health.mean_tree_depth}")
        print(f"Max tree depth: {health.max_tree_depth}")

        deterministic_names = ["ln_rate_model", *built.mechanism_result.pointwise]
        posterior = compute_posterior_deterministics(
            idata,
            built,
            var_names=deterministic_names,
            progressbar=False,
        )

        for name in deterministic_names:
            values = np.asarray(posterior[name])

            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Posterior deterministic '{name}' contains non-finite values.")

        print(f"Posterior deterministics finite: {len(deterministic_names)} variables")

        idata = add_log_likelihood(idata, built, progressbar=False)

        if not (hasattr(idata, "log_likelihood") or "log_likelihood" in idata):
            raise RuntimeError("Log-likelihood group was not created.")

        print("Log likelihood: OK")

        output_dir = OUTPUT_ROOT / MATERIAL / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        idata.to_netcdf(output_dir / "posterior_smoke.nc", engine="h5netcdf")


if __name__ == "__main__":
    main()