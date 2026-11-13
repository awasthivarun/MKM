"""Fit acidic or basic Pt100 CO oxidation models using the shared posterior lifecycle."""

from argparse import ArgumentParser

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.models.ptru_registry import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
    run_posterior_lifecycle,
)
from mkm.workflows.pt100_fit import build_pt100_fit
from mkm.workflows.ptru_posterior import (
    build_pt100_run_metadata,
    validate_pt100_run_metadata,
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--nuts-sampler", default="nutpie")
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    fit = build_pt100_fit(paths, args.medium, args.model)
    sampler = _sampler_settings(args)

    def metadata_factory():
        return build_pt100_run_metadata(paths, fit, sampler)

    def checkpoint_validator(metadata):
        validate_pt100_run_metadata(
            metadata,
            paths,
            fit,
            allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        )

    print(f"\nPt100 posterior fit: {fit.medium} / {fit.model_name}")
    print(f"Observations: {len(fit.inputs.observation_rate):,}")
    print(f"Model points: {len(fit.inputs.model_point_E_V_SHE):,}")
    print(f"Likelihood: {RATE_NORMAL}")
    print(f"Error structure: {fit.specification.error_structure}")
    print("sigma = sigma_rate_abs + sigma_rate_rel * model_rate")
    print(f"Free parameters: {[rv.name for rv in fit.built_model.model.free_RVs]}")
    print(f"Output: {fit.output_dir}")

    result = run_posterior_lifecycle(
        built=fit.built_model,
        output_dir=fit.output_dir,
        parameter_specs=fit.parameter_specs,
        sampler=sampler,
        resume=args.resume,
        overwrite=args.overwrite,
        metadata_factory=metadata_factory,
        checkpoint_validator=checkpoint_validator,
        progressbar=True,
    )

    if result.sampling_seconds is not None:
        print(f"\nSampling wall time: {result.sampling_seconds:.2f} s")

    health = result.sampling_health
    print(f"Divergences: {health['n_divergent']}")
    print(f"Maximum R-hat: {health['max_rhat']:.4f}")
    print(f"Minimum bulk ESS: {health['min_ess_bulk']:.1f}")
    print(f"Minimum tail ESS: {health['min_ess_tail']:.1f}")
    print(f"Saved posterior and parameter summary: {fit.output_dir}")


if __name__ == "__main__":
    main()