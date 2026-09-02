"""Fit one AgPd material or the configured all-material dataset."""

from argparse import ArgumentParser

from mkm.models.agpd_basic import available_agpd_models
from mkm.project_paths import ProjectPaths
from mkm.provenance import build_fit_metadata
from mkm.workflows.agpd_basic import build_agpd_model_data, load_agpd_model_config
from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    build_agpd_fit_model,
    fit_materials,
    fit_output_dir,
    resolved_parameterization_metadata,
    resolve_agpd_fit_specification,
)
from mkm.workflows.agpd_posterior import validate_agpd_run_metadata
from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
    run_posterior_lifecycle,
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
    output_dir = fit_output_dir(paths, specification)
    sampler = _sampler_settings(args)

    if not args.resume:
        print(f"Fit scope: {specification.fit_scope}")
        print(f"Materials: {', '.join(fit.inputs.materials)}")
        print(f"Model: {specification.model_name}")
        print(f"Parameterization: {specification.parameterization}")
        print(f"Error structure: {specification.error_structure}")
        print("Likelihood: Normal in linear rate space")
        print("sigma = sigma_rate_abs + sigma_rate_rel * model_rate")

    def metadata_factory():
        return build_fit_metadata(
            root=paths.root,
            fit_scope=specification.fit_scope,
            materials=fit.inputs.materials,
            model_name=specification.model_name,
            error_structure=specification.error_structure,
            data_path=paths.agpd_selected_path,
            model_config_path=paths.agpd_model_config_path,
            sampler=sampler,
            parameterization=specification.parameterization,
            parameterization_specification=resolved_parameterization_metadata(
                specification,
                config,
            ),
            prior_material=specification.prior_material,
        )

    def checkpoint_validator(metadata):
        validate_agpd_run_metadata(
            metadata,
            paths,
            specification,
            fit.inputs.materials,
            config,
            allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        )

    result = run_posterior_lifecycle(
        built=fit.built_model,
        output_dir=output_dir,
        parameter_specs=all_parameter_specs(specification, config),
        sampler=sampler,
        resume=args.resume,
        overwrite=args.overwrite,
        metadata_factory=metadata_factory,
        checkpoint_validator=checkpoint_validator,
        progressbar=True,
    )

    if args.resume:
        print(f"Reused posterior checkpoint: {output_dir / 'posterior.nc'}")
    if result.sampling_seconds is not None:
        print(f"Sampling wall time: {result.sampling_seconds:.2f} s")
    print(f"Divergences: {result.sampling_health['n_divergent']}")
    print(f"Maximum R-hat: {result.sampling_health['max_rhat']:.4f}")
    print(f"Minimum bulk ESS: {result.sampling_health['min_ess_bulk']:.1f}")
    print(f"Minimum tail ESS: {result.sampling_health['min_ess_tail']:.1f}")
    print(f"Saved posterior and combined parameter summary to: {output_dir}")


if __name__ == "__main__":
    main()
