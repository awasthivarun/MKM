"""Fit Pt100 / PtRu models individually or jointly, separately in acid and base."""

from argparse import ArgumentParser

from mkm.models.ptru import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE, RUN_STATUS_SAMPLED, run_posterior_lifecycle
from mkm.workflows.ptru_fit import build_ptru_fit, load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import build_ptru_run_metadata, validate_ptru_run_metadata


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--material", default="Pt100")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", choices=("shared", "independent", "linear_xRu"), default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--exclude-materials", nargs="*", default=())
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--nuts-sampler", default="nutpie")
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing run before resampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, args.medium, args.model, material=args.material,
        all_materials=args.all_materials, parameterization=args.parameterization,
        error_structure=args.error_structure, excluded_materials=args.exclude_materials,
    )
    fit = build_ptru_fit(paths, spec)
    sampler = {
        "nuts_sampler": args.nuts_sampler, "backend": args.backend,
        "draws": args.draws, "tune": args.tune, "chains": args.chains, "cores": args.cores,
        "target_accept": args.target_accept, "random_seed": args.random_seed,
    }
    print(f"PtRu fit: {args.medium} / {args.model} / {spec.fit_scope}")
    print(f"Materials: {', '.join(fit.inputs.materials)}")
    print(f"Parameterization: {spec.parameterization}; error structure: {spec.error_structure}")
    print(f"Observations: {len(fit.inputs.observation_rate):,}")
    print(f"Free parameters: {[rv.name for rv in fit.built_model.model.free_RVs]}")
    print("Likelihood: linear-rate Normal; sigma = sigma_rate_abs + sigma_rate_rel * model_rate")
    print(f"Output: {fit.output_dir}")

    def metadata_factory():
        return build_ptru_run_metadata(paths, fit, sampler)

    def validator(metadata):
        validate_ptru_run_metadata(metadata, paths, fit, allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE))

    result = run_posterior_lifecycle(
        built=fit.built_model, output_dir=fit.output_dir,
        parameter_specs=fit.parameter_specs, sampler=sampler,
        resume=args.resume, overwrite=args.overwrite,
        metadata_factory=metadata_factory, checkpoint_validator=validator, progressbar=True,
    )
    health = result.sampling_health
    print(f"Divergences: {health['n_divergent']}")
    print(f"Maximum R-hat: {health['max_rhat']:.4f}")
    print(f"Minimum bulk ESS: {health['min_ess_bulk']:.1f}")
    print(f"Minimum tail ESS: {health['min_ess_tail']:.1f}")


if __name__ == "__main__":
    main()
