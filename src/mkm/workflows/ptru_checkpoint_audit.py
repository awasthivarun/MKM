"""Read-only comparison of Pt100 checkpoint provenance with the current fit.

An audit never approves an old posterior for loading, resuming, or rewriting.
Source changes require independent mathematical/numerical equivalence checks.
"""

from dataclasses import dataclass


IDENTITY_FIELDS = (
    "fit_scope", "materials", "model", "likelihood", "error_structure",
    "parameterization", "parameterization_specification", "prior_specification",
)
INPUT_FIELDS = (
    "data_path", "data_sha256", "model_config_path", "model_config_paths", "model_config_sha256",
)
MODEL_FIELDS = ("family", "medium", "model_name", "assumptions")


@dataclass(frozen=True)
class Pt100CheckpointAudit:
    status: str
    blocking_differences: tuple[str, ...]
    source_differences: tuple[str, ...]
    provenance_differences: tuple[str, ...]

    @property
    def strict_match(self):
        return self.status == "strict_match"


def audit_pt100_checkpoint(saved, expected):
    """Classify metadata differences; never infer physical equivalence from a hash mismatch."""
    blocking = []
    source = []
    provenance = []

    for field in IDENTITY_FIELDS:
        if saved.get(field) != expected.get(field):
            blocking.append(field)

    saved_inputs = saved.get("inputs") or {}
    expected_inputs = expected.get("inputs") or {}
    for field in INPUT_FIELDS:
        if saved_inputs.get(field) != expected_inputs.get(field):
            blocking.append(f"inputs.{field}")

    saved_model = saved.get("model_specification") or {}
    expected_model = expected.get("model_specification") or {}
    for field in MODEL_FIELDS:
        if saved_model.get(field) != expected_model.get(field):
            blocking.append(f"model_specification.{field}")

    for field in sorted((set(saved_model) | set(expected_model)) - set(MODEL_FIELDS) - {"source_sha256"}):
        if saved_model.get(field) != expected_model.get(field):
            target = source if field == "registry" else blocking
            target.append(f"model_specification.{field}")

    saved_sources = saved_model.get("source_sha256") or {}
    expected_sources = expected_model.get("source_sha256") or {}
    for path in sorted(set(saved_sources) | set(expected_sources)):
        if saved_sources.get(path) != expected_sources.get(path):
            source.append(f"source_sha256:{path}")

    if saved.get("resolved_fit_sha256") != expected.get("resolved_fit_sha256"):
        provenance.append("resolved_fit_sha256")
    if saved.get("status") != "complete":
        blocking.append("status")

    if blocking:
        status = "incompatible"
    elif source:
        status = "source_change_unverified"
    elif provenance:
        status = "provenance_mismatch"
    else:
        status = "strict_match"

    return Pt100CheckpointAudit(
        status=status,
        blocking_differences=tuple(blocking),
        source_differences=tuple(source),
        provenance_differences=tuple(provenance),
    )
