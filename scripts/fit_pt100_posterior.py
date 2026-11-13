
"""Fit acidic or basic Pt100 CO oxidation models using the shared PyMC posterior lifecycle."""

from argparse import ArgumentParser

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES
from mkm.project_paths import ProjectPaths
from mkm.provenance import build_fit_metadata, sha256_file
from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
    run_posterior_lifecycle,
)
from mkm.workflows.pt100_fit import build_pt100_fit


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(PT100_PARAMETER_CLASSES))
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


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    fit = build_pt100_fit(paths, args.medium, args.model)

    sampler = {
        "nuts_sampler": args.nuts_sampler,
        "backend": args.backend,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_seed": args.random_seed,
    }

    # Include uncommitted Pt100 code in the run identity.
    source_files = (
        "src/mkm/mechanisms/pt100.py",
        "src/mkm/models/pt100.py",
        "src/mkm/workflows/pt100_fit.py",
        "src/mkm/mechanisms/agpd_basic/electrochem.py",
        "src/mkm/inference/model.py",
        "src/mkm/inference/likelihoods.py",
        "src/mkm/inference/priors.py",
    )
    source_hashes = {
        filename: sha256_file(paths.root / filename)
        for filename in source_files
    }

    model_specification = {
        "family": "Pt100",
        "medium": args.medium,
        "model_name": args.model,
        "assumptions": fit.config["mechanism_assumptions"],
        "source_sha256": source_hashes,
    }

    def metadata_factory():
        return build_fit_metadata(
            root=paths.root,
            fit_scope="individual",
            materials=fit.inputs.materials,
            model_name=args.model,
            error_structure="shared",
            data_path=fit.data_path,
            model_config_path=fit.model_config_path,
            model_config_paths=(fit.model_config_path, fit.prior_config_path),
            sampler=sampler,
            model_specification=model_specification,
            prior_specification=fit.parameter_specs,
        )

    def checkpoint_validator(metadata):
        expected = metadata_factory()
        keys = (
            "fit_scope", "materials", "model", "likelihood",
            "error_structure", "parameterization", "model_specification",
            "prior_specification", "resolved_fit_sha256",
        )

        mismatches = [
            key for key in keys
            if metadata.get(key) != expected.get(key)
        ]

        for key in ("data_path", "data_sha256", "model_config_sha256"):
            if metadata.get("inputs", {}).get(key) != expected["inputs"][key]:
                mismatches.append(f"inputs.{key}")

        if metadata.get("status") not in (RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE):
            mismatches.append("status")

        if mismatches:
            raise ValueError(
                "Posterior checkpoint does not match the current Pt100 fit: "
                + ", ".join(mismatches)
            )

    print(f"\nPt100 posterior fit: {args.medium} / {args.model}")
    print(f"Observations: {len(fit.inputs.observation_rate):,}")
    print(f"Model points: {len(fit.inputs.model_point_E_V_SHE):,}")
    print(f"Likelihood: {RATE_NORMAL}")
    print("Error structure: shared")
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
