"""Fit an AgPd composition model across multiple materials."""

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
from mkm.models.agpd_basic import (
    available_agpd_composition_models,
    available_agpd_composition_parameterizations,
    build_agpd_composition_mechanism,
)
from mkm.project_paths import ProjectPaths
from mkm.provenance import build_fit_metadata, write_run_metadata
from mkm.workflows.agpd_basic import (
    available_agpd_materials,
    build_agpd_composition_model_data,
    build_agpd_inputs,
    load_agpd_model_config,
)

DEFAULT_COMPOSITION_MODEL = "shared"
DEFAULT_PRIOR_MATERIAL = "Ag10Pd90"

DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.90
RANDOM_SEED = 20260824


PATHWAY_LOG_RATE_NAMES = {"ln_rate_BF", "ln_rate_ER", "ln_rate_LH"}


def _validate_posterior_deterministic(name, values):
    values = np.asarray(values)

    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise RuntimeError(f"Posterior deterministic '{name}' contains NaN or +inf values.")
    if name not in PATHWAY_LOG_RATE_NAMES and np.any(np.isneginf(values)):
        raise RuntimeError(f"Posterior deterministic '{name}' contains -inf values.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_composition_models())
    parser.add_argument(
        "--composition-model",
        choices=available_agpd_composition_parameterizations(),
        default=DEFAULT_COMPOSITION_MODEL,
    )
    parser.add_argument("--materials", nargs="+", default=None)
    parser.add_argument("--prior-material", default=DEFAULT_PRIOR_MATERIAL)
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept", "mvn", "rate_normal"], default="iid")
    parser.add_argument(
        "--resume-free",
        action="store_true",
        help="Reuse an existing posterior_free.nc and finish deterministic/log-likelihood reconstruction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    composition_model = args.composition_model
    likelihood_name = args.likelihood

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)

    materials = tuple(args.materials or available_agpd_materials(config))
    unknown = [material for material in materials if material not in config["surface_composition"]]
    if unknown:
        raise ValueError(f"Unknown AgPd materials: {unknown}.")
    if args.prior_material not in config["prior_profiles"]:
        raise ValueError(f"No prior profile is configured for '{args.prior_material}'.")

    model_data = build_agpd_composition_model_data(paths, materials)
    inputs = build_agpd_inputs(model_data, config, likelihood_name)
    mechanism = build_agpd_composition_mechanism(
        model_name=model_name,
        materials=tuple(inputs.materials),
        config=config,
        prior_material=args.prior_material,
        composition_model=composition_model,
    )
    likelihood_config = config["likelihood"]
    setup_config = likelihood_config["setup_intercept"]
    mvn_config = likelihood_config["mvn"]
    rate_normal_config = likelihood_config["rate_normal"]
    use_setup_intercept = likelihood_name == "setup_intercept"
    use_correlated_potential = likelihood_name == "mvn"
    use_rate_normal = likelihood_name == "rate_normal"
    built = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        sigma_prior_median=likelihood_config["sigma_prior_median"],
        sigma_prior_log_sd=likelihood_config["sigma_prior_log_sd"],
        setup_intercept=use_setup_intercept,
        setup_prior_median=setup_config["prior_median"],
        setup_prior_log_sd=setup_config["prior_log_sd"],
        correlated_potential=use_correlated_potential,
        correlation_length_prior_median_V=mvn_config["correlation_length_prior_median_V"],
        correlation_length_prior_log_sd=mvn_config["correlation_length_prior_log_sd"],
        rate_normal=use_rate_normal,
        sigma_abs_prior_median_s_inv=rate_normal_config["sigma_abs_prior_median_s_inv"],
        sigma_abs_prior_log_sd=rate_normal_config["sigma_abs_prior_log_sd"],
        sigma_rel_prior_median=rate_normal_config["sigma_rel_prior_median"],
        sigma_rel_prior_log_sd=rate_normal_config["sigma_rel_prior_log_sd"],
    )
    output_dir = paths.agpd_composition_posterior_output_dir(
        composition_model=composition_model,
        model_name=model_name,
        likelihood_name=likelihood_name,
    )
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
        material="composition",
        model_name=model_name,
        likelihood_name=likelihood_name,
        data_path=paths.agpd_selected_path,
        model_config_path=paths.agpd_model_config_path,
        sampler=sampler_settings,
    )
    metadata["composition_model"] = composition_model
    metadata["materials"] = list(inputs.materials)
    metadata["prior_profile_material"] = args.prior_material
    if use_correlated_potential:
        metadata["residual_correlation"] = {
            "axis": "E_V_SHE",
            "kernel": "exponential",
            "curve_grouping": ["condition_id", "replicate"],
            "correlation_length_prior_median_V": float(mvn_config["correlation_length_prior_median_V"]),
            "correlation_length_prior_log_sd": float(mvn_config["correlation_length_prior_log_sd"]),
        }

    if use_rate_normal:
        metadata["rate_error_model"] = {
            "distribution": "normal",
            "observation_space": "rate_s_inv",
            "sigma": "sigma_rate_abs + sigma_rate_rel * rate_model",
            "sigma_abs_prior_median_s_inv": float(rate_normal_config["sigma_abs_prior_median_s_inv"]),
            "sigma_abs_prior_log_sd": float(rate_normal_config["sigma_abs_prior_log_sd"]),
            "sigma_rel_prior_median": float(rate_normal_config["sigma_rel_prior_median"]),
            "sigma_rel_prior_log_sd": float(rate_normal_config["sigma_rel_prior_log_sd"]),
        }

    if composition_model == "linear_xAg":
        linear_config = config["composition_parameterizations"]["linear_xAg"]
        metadata["composition_x_reference"] = float(linear_config["x_reference"])
        metadata["composition_slope_parameters"] = list(linear_config["models"][model_name]["slopes"])

    write_run_metadata(metadata, output_dir / "run_metadata.yaml")
    print(f"\nAgPd composition model: {model_name}")
    print(f"Composition parameterization: {composition_model}")
    print(f"Materials: {', '.join(inputs.materials)}")
    print(f"Prior profile source: {args.prior_material}")
    print(f"Likelihood: {likelihood_name}")
    if use_rate_normal:
        print("Global rate-error terms: sigma_rate_abs, sigma_rate_rel")
    else:
        print(f"Material-specific marginal noise terms: {len(inputs.materials)}")
    if use_correlated_potential:
        print(f"Material-specific E-correlation length terms: {len(inputs.materials)}")

    if composition_model == "linear_xAg":
        print(f"x_Ag reference: {metadata['composition_x_reference']:.2f}")
        print(f"Linear x_Ag slopes: {', '.join(metadata['composition_slope_parameters'])}")
    if use_setup_intercept:
        print(f"Setups: {len(inputs.setup_labels)}")
    print("Sampler: nutpie / numba")
    print(f"Chains: {CHAINS}, tune: {TUNE}, draws: {DRAWS}, target_accept: {TARGET_ACCEPT}")

    posterior_free_path = output_dir / "posterior_free.nc"
    if args.resume_free:
        if not posterior_free_path.exists():
            raise FileNotFoundError(f"Free posterior not found: {posterior_free_path}")
        idata = az.from_netcdf(posterior_free_path)
        sampling_seconds = None
        print(f"Reusing free posterior: {posterior_free_path}")
    else:
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
        idata.to_netcdf(posterior_free_path, engine="h5netcdf")

    health = summarize_sampler_health(idata)
    free_var_names = [rv.name for rv in built.model.free_RVs]
    diagnostics = az.summary(idata, var_names=free_var_names, kind="diagnostics", round_to=None)
    diagnostics.to_csv(output_dir / "sampler_diagnostics.csv")
    if sampling_seconds is not None:
        print(f"\nSampling wall time: {sampling_seconds:.2f} s")
    else:
        print("\nSampling wall time: reused existing free posterior")
    print(f"Divergences: {health.divergences}")
    print(f"Mean tree depth: {health.mean_tree_depth}")
    print(f"Max tree depth: {health.max_tree_depth}")
    print(f"Maximum R-hat: {diagnostics['r_hat'].max():.4f}")
    print(f"Minimum bulk ESS: {diagnostics['ess_bulk'].min():.1f}")
    print(f"Minimum tail ESS: {diagnostics['ess_tail'].min():.1f}")

    deterministic_names = ["ln_rate_model", "rate_model", *built.mechanism_result.pointwise]
    if getattr(built.likelihood, "setup_offset", None) is not None:
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
        _validate_posterior_deterministic(name, posterior[name])
    idata = add_log_likelihood(idata, built, backend="numba", progressbar=True)
    idata.to_netcdf(output_dir / "posterior.nc", engine="h5netcdf")

    print(f"Posterior deterministics finite: {len(deterministic_names)} variables")
    print("Log likelihood: OK")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
