from argparse import ArgumentParser
from time import perf_counter

import arviz as az
import numpy as np

from mkm.inference.model import build_pymc_model
from mkm.inference.posterior import (
    add_log_likelihood,
    compute_posterior_deterministics,
    sample_posterior,
    summarize_sampler_health,
)
from mkm.models.agpd_basic import available_agpd_models, build_agpd_mechanism
from mkm.project_paths import ProjectPaths
from mkm.provenance import build_fit_metadata, write_run_metadata
from mkm.workflows.agpd_basic import (
    build_agpd_inputs,
    build_agpd_model_data,
    load_agpd_model_config,
    validate_agpd_material,
)


DEFAULT_MATERIAL = "Ag10Pd90"

DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.90
RANDOM_SEED = 20260821


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="setup_intercept")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    material = args.material
    likelihood_name = args.likelihood

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    validate_agpd_material(config, material)

    likelihood_config = config["likelihood"]
    setup_config = likelihood_config["setup_intercept"]
    use_setup_intercept = likelihood_name == "setup_intercept"

    model_data = build_agpd_model_data(paths, material)
    inputs = build_agpd_inputs(model_data, config, likelihood_name)
    mechanism = build_agpd_mechanism(model_name=model_name, material=material, config=config)

    built = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        sigma_prior_median=likelihood_config["sigma_prior_median"],
        sigma_prior_log_sd=likelihood_config["sigma_prior_log_sd"],
        setup_intercept=use_setup_intercept,
        setup_prior_median=setup_config["prior_median"],
        setup_prior_log_sd=setup_config["prior_log_sd"],
    )

    output_dir = paths.agpd_posterior_output_dir(material, model_name, likelihood_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler_settings = {
        "nuts_sampler": "nutpie",
        "backend": "numba",
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "cores": CORES,
        "target_accept": TARGET_ACCEPT,
        "random_seed": RANDOM_SEED,
    }
    metadata = build_fit_metadata(
        root=paths.root,
        material=material,
        model_name=model_name,
        likelihood_name=likelihood_name,
        data_path=paths.agpd_selected_path,
        model_config_path=paths.agpd_model_config_path,
        sampler=sampler_settings,
    )
    write_run_metadata(metadata, output_dir / "run_metadata.yaml")

    print(f"\n{material}: {model_name}")
    print(f"Likelihood: {likelihood_name}")

    if use_setup_intercept:
        print(f"Setups: {len(inputs.setup_labels)}")

    print("Sampler: nutpie / numba")
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
        nuts_sampler=sampler_settings["nuts_sampler"],
        backend=sampler_settings["backend"],
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

    if built.likelihood.setup_offset is not None:
        deterministic_names.append("ln_rate_setup_offset")

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
