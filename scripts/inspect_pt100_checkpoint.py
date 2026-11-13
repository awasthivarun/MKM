"""Audit saved Pt100 metadata without loading, modifying, or resuming a posterior."""

from argparse import ArgumentParser

from mkm.models.ptru_registry import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.provenance import read_run_metadata
from mkm.workflows.pt100_fit import build_pt100_fit
from mkm.workflows.ptru_checkpoint_audit import audit_pt100_checkpoint
from mkm.workflows.ptru_posterior import build_pt100_run_metadata


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    fit = build_pt100_fit(paths, args.medium, args.model)
    metadata_path = fit.output_dir / "run_metadata.yaml"
    posterior_path = fit.output_dir / "posterior.nc"
    if not posterior_path.is_file():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")

    saved = read_run_metadata(metadata_path)
    expected = build_pt100_run_metadata(paths, fit, saved.get("sampler") or {})
    audit = audit_pt100_checkpoint(saved, expected)

    print(f"Checkpoint: {posterior_path}")
    print(f"Audit status: {audit.status}")
    for label, differences in (
        ("Blocking differences", audit.blocking_differences),
        ("Source differences", audit.source_differences),
        ("Provenance differences", audit.provenance_differences),
    ):
        print(f"{label}: {', '.join(differences) if differences else 'none'}")

    if audit.strict_match:
        print("Metadata matches the current fit. Normal checkpoint validation still applies.")
    elif audit.status == "source_change_unverified":
        print("Data, priors and model assumptions match, but implementation identity changed.")
        print("Mathematical/numerical equivalence has NOT been established; do not bypass validation.")
    else:
        print("Checkpoint is not verified for current-code loading or resumption.")
    print("Read-only audit: posterior and metadata files were not modified.")


if __name__ == "__main__":
    main()
