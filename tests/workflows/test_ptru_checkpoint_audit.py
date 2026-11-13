"""Regression tests for read-only Pt100 checkpoint-identity assessment."""

from copy import deepcopy

from mkm.workflows.ptru_checkpoint_audit import audit_pt100_checkpoint


def checkpoint_metadata():
    return {
        "fit_scope": "individual",
        "materials": ["Pt100"],
        "model": "CO_ER_LH",
        "likelihood": "rate_normal",
        "error_structure": "shared",
        "parameterization": None,
        "parameterization_specification": None,
        "prior_specification": {"Gact1_0": {"distribution": "normal", "mu": 0.5, "sigma": 0.1}},
        "model_specification": {
            "family": "Pt100",
            "medium": "acidic",
            "model_name": "CO_ER_LH",
            "assumptions": {"OH_adsorption": "quasi_equilibrium"},
            "registry": {"pathways": ["ER", "LH"]},
            "source_sha256": {"src/mkm/mechanisms/pt100.py": "new-hash"},
        },
        "inputs": {
            "data_path": "selected.parquet", "data_sha256": "data-hash",
            "model_config_path": "pt100.yaml", "model_config_paths": ["pt100.yaml", "priors.yaml"],
            "model_config_sha256": "config-hash",
        },
        "resolved_fit_sha256": "new-fit-hash",
        "status": "complete",
    }


def test_matching_checkpoint_requires_no_migration():
    metadata = checkpoint_metadata()
    audit = audit_pt100_checkpoint(metadata, deepcopy(metadata))
    assert audit.strict_match
    assert audit.blocking_differences == ()
    assert audit.source_differences == ()


def test_pre_refactor_checkpoint_is_not_silently_accepted():
    expected = checkpoint_metadata()
    saved = deepcopy(expected)
    del saved["model_specification"]["registry"]
    saved["model_specification"]["source_sha256"] = {
        "src/mkm/mechanisms/pt100.py": "old-hash",
    }
    saved["resolved_fit_sha256"] = "old-fit-hash"

    audit = audit_pt100_checkpoint(saved, expected)
    assert audit.status == "source_change_unverified"
    assert not audit.strict_match
    assert "source_sha256:src/mkm/mechanisms/pt100.py" in audit.source_differences
    assert "model_specification.registry" in audit.source_differences
    assert audit.provenance_differences == ("resolved_fit_sha256",)


def test_changed_data_prior_or_assumptions_block_compatibility():
    expected = checkpoint_metadata()
    for field, change in (
        ("data", lambda saved: saved["inputs"].update(data_sha256="changed")),
        ("prior", lambda saved: saved.update(prior_specification={})),
        ("assumption", lambda saved: saved["model_specification"]["assumptions"].update(
            OH_adsorption="irreversible"
        )),
    ):
        saved = deepcopy(expected)
        change(saved)
        audit = audit_pt100_checkpoint(saved, expected)
        assert audit.status == "incompatible", field
        assert audit.blocking_differences, field


def test_incomplete_checkpoint_and_unexplained_hash_are_not_accepted():
    expected = checkpoint_metadata()
    saved = deepcopy(expected)
    saved["status"] = "sampled_free_variables"
    assert audit_pt100_checkpoint(saved, expected).status == "incompatible"

    saved = deepcopy(expected)
    saved["resolved_fit_sha256"] = "different-without-source-change"
    assert audit_pt100_checkpoint(saved, expected).status == "provenance_mismatch"
