"""Fit one AgPd material or the configured all-material dataset."""

from argparse import ArgumentParser
from shutil import rmtree
from time import perf_counter

import numpy as np

from mkm.inference.posterior import (
    add_log_likelihood,
    compute_posterior_deterministics,
    load_inference_data,
    sample_posterior,
    write_inference_data,
)
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.diagnostics import build_posterior_parameter_summary
from mkm.postprocessing.sampling import build_sampling_health
from mkm.project_paths import ProjectPaths
from mkm.provenance import build_fit_metadata, read_run_metadata, write_run_metadata
from mkm.workflows.agpd_basic import build_agpd_model_data, load_agpd_model_config
from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    build_agpd_fit_model,
    fit_materials,
    fit_output_dir,
    resolve_agpd_fit_specification,
)
from mkm.workflows.agpd_posterior import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
    validate_agpd_run_metadata,
)


DEFAULT_MATERIAL = "Ag10Pd90"
DEFAULT_PRIOR_MATERIAL = "Ag10Pd90"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument(
        "--error-structure",
        choices=("shared", "material"),
        default="material",
    )
    parser.add_argument("--prior-material", default=DEFAULT_PRIOR_MATERIAL)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--nuts-sampler", default="nutpie")
    parser.add_argument("--backend", default="numba")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume finalization from the posterior.nc checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory before sampling a new posterior.",
    )
    return parser.parse_args()


def _sampler_settings(args):
    return {
        "nuts_sampler": args.nuts_sampler,
        "backend": args.backend,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_seed": args.random_seed,
    }


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    specification = resolve_agpd_fit_specification(
        config,
        model_name=args.model,
        all_materials=args.all_materials,
        material=args.material,
        parameterization=args.parameterization,
        error_structure=args.error_structure,
        prior_material=args.prior_material,
    )

    materials = fit_materials(specification, config)
    model_data = build_agpd_model_data(paths, materials)
    fit = build_agpd_fit_model(specification, model_data, config)
    built = fit.built_model
    output_dir = fit_output_dir(paths, specification)
    posterior_path = output_dir / "posterior.nc"
    metadata_path = output_dir / "run_metadata.yaml"

    if args.overwrite and output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = _sampler_settings(args)
    sampling_seconds = None

    if args.resume:
        if not posterior_path.exists():
            raise FileNotFoundError(f"Posterior checkpoint not found: {posterior_path}")
        metadata = read_run_metadata(metadata_path)
        validate_agpd_run_metadata(
            metadata,
            paths,
            specification,
            fit.inputs.materials,
            allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        )
        inference_data = load_inference_data(posterior_path)
        print(f"Reusing posterior checkpoint: {posterior_path}")
    else:
        if posterior_path.exists():
            raise FileExistsError(
                f"Posterior already exists: {posterior_path}. "
                "Use --resume or --overwrite."
            )

        print(f"Fit scope: {specification.fit_scope}")
        print(f"Materials: {', '.join(fit.inputs.materials)}")
        print(f"Model: {specification.model_name}")
        print(f"Parameterization: {specification.parameterization}")
        print(f"Error structure: {specification.error_structure}")
        print("Likelihood: Normal in linear rate space")
        print("sigma = sigma_rate_abs + sigma_rate_rel * model_rate")

        start = perf_counter()
        inference_data = sample_posterior(
            built,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
            nuts_sampler=args.nuts_sampler,
            backend=args.backend,
            compute_convergence_checks=False,
        )
        sampling_seconds = perf_counter() - start

        write_inference_data(inference_data, posterior_path)
        metadata = build_fit_metadata(
            root=paths.root,
            fit_scope=specification.fit_scope,
            materials=fit.inputs.materials,
            model_name=specification.model_name,
            error_structure=specification.error_structure,
            data_path=paths.agpd_selected_path,
            model_config_path=paths.agpd_model_config_path,
            sampler=sampler,
            parameterization=specification.parameterization,
            prior_material=specification.prior_material,
        )
        metadata["status"] = RUN_STATUS_SAMPLED
        write_run_metadata(metadata, metadata_path)

    free_parameter_names = tuple(variable.name for variable in built.model.free_RVs)
    missing = [
        name for name in free_parameter_names if name not in inference_data.posterior
    ]
    if missing:
        raise ValueError(f"Posterior checkpoint is missing fitted variables: {missing}")

    sampling_health = build_sampling_health(
        inference_data,
        parameter_names=free_parameter_names,
    )
    metadata["sampling_health"] = sampling_health
    metadata["status"] = RUN_STATUS_SAMPLED
    write_run_metadata(metadata, metadata_path)

    if "ln_rate_model" not in inference_data.posterior:
        posterior_for_check = compute_posterior_deterministics(
            inference_data,
            built,
            var_names=["ln_rate_model"],
            backend=args.backend,
            progressbar=True,
        )
    else:
        posterior_for_check = inference_data.posterior

    if not np.all(
        np.isfinite(np.asarray(posterior_for_check["ln_rate_model"], dtype=float))
    ):
        raise RuntimeError(
            'Posterior deterministic "ln_rate_model" contains non-finite values.'
        )

    inference_data.posterior = posterior_for_check

    if (
        not hasattr(inference_data, "log_likelihood")
        or "rate_observed" not in inference_data.log_likelihood
    ):
        inference_data = add_log_likelihood(
            inference_data,
            built,
            backend=args.backend,
            progressbar=True,
        )

    log_likelihood = np.asarray(
        inference_data.log_likelihood["rate_observed"],
        dtype=float,
    )
    if not np.all(np.isfinite(log_likelihood)):
        raise RuntimeError(
            'Posterior log likelihood "rate_observed" contains non-finite values.'
        )

    inference_data.posterior = inference_data.posterior[list(free_parameter_names)]
    write_inference_data(inference_data, posterior_path)

    parameter_summary = build_posterior_parameter_summary(
        inference_data,
        all_parameter_specs(specification, config),
    )
    parameter_summary.to_csv(output_dir / "posterior_parameters.csv", index=False)

    metadata["status"] = RUN_STATUS_COMPLETE
    metadata["sampling_health"] = sampling_health
    metadata["stored_posterior_variables"] = list(inference_data.posterior.data_vars)
    write_run_metadata(metadata, metadata_path)

    if sampling_seconds is not None:
        print(f"Sampling wall time: {sampling_seconds:.2f} s")
    print(f"Divergences: {sampling_health['n_divergent']}")
    print(f"Maximum R-hat: {sampling_health['max_rhat']:.4f}")
    print(f"Minimum bulk ESS: {sampling_health['min_ess_bulk']:.1f}")
    print(f"Minimum tail ESS: {sampling_health['min_ess_tail']:.1f}")
    print(f"Saved posterior and combined parameter summary to: {output_dir}")


if __name__ == "__main__":
    main()
