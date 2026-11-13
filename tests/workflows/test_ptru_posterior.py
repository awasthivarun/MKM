from types import SimpleNamespace

import pytest

from mkm.project_paths import ProjectPaths
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE, RUN_STATUS_SAMPLED
from mkm.workflows.pt100_fit import Pt100FitSpecification
from mkm.workflows import ptru_posterior


def make_fake_fit(paths):
    data_path = paths.root / "selected.parquet"
    model_path = paths.root / "pt100.yaml"
    prior_path = paths.root / "priors.yaml"

    data_path.write_bytes(b"synthetic selected data")
    model_path.write_text("model: test\n", encoding="utf-8")
    prior_path.write_text("priors: test\n", encoding="utf-8")

    return SimpleNamespace(
        specification=Pt100FitSpecification(medium="acidic", model_name="CO_LH"),
        inputs=SimpleNamespace(materials=("Pt100",)),
        config={"mechanism_assumptions": {"CO_adsorption": "finite_rate_SSA"}},
        parameter_specs={"Gact1_0": {"distribution": "normal", "mu": 0.5, "sigma": 0.1}},
        data_path=data_path,
        model_config_path=model_path,
        prior_config_path=prior_path,
    )


def test_ptru_source_identity_tracks_shared_model_code():
    tracked = set(ptru_posterior.PT100_SOURCE_FILES)

    assert "src/mkm/mechanisms/co_ssa.py" in tracked
    assert "src/mkm/models/ptru_registry.py" in tracked
    assert "src/mkm/workflows/likelihood_configuration.py" in tracked
    assert "src/mkm/inference/model.py" in tracked


def test_ptru_checkpoint_validates_current_fit(tmp_path, monkeypatch):
    paths = ProjectPaths(root=tmp_path)
    fit = make_fake_fit(paths)

    # This test isolates the metadata contract from repository source files.
    monkeypatch.setattr(ptru_posterior, "PT100_SOURCE_FILES", ())

    metadata = ptru_posterior.build_pt100_run_metadata(paths, fit, sampler={})
    metadata["status"] = RUN_STATUS_COMPLETE

    ptru_posterior.validate_pt100_run_metadata(metadata, paths, fit)

    metadata["status"] = RUN_STATUS_SAMPLED
    ptru_posterior.validate_pt100_run_metadata(
        metadata, paths, fit,
        allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
    )


def test_ptru_checkpoint_detects_changed_data(tmp_path, monkeypatch):
    paths = ProjectPaths(root=tmp_path)
    fit = make_fake_fit(paths)
    monkeypatch.setattr(ptru_posterior, "PT100_SOURCE_FILES", ())

    metadata = ptru_posterior.build_pt100_run_metadata(paths, fit, sampler={})
    metadata["status"] = RUN_STATUS_COMPLETE

    fit.data_path.write_bytes(b"changed data")

    with pytest.raises(ValueError, match="inputs.data_sha256"):
        ptru_posterior.validate_pt100_run_metadata(metadata, paths, fit)


def test_ptru_checkpoint_detects_changed_configuration(tmp_path, monkeypatch):
    paths = ProjectPaths(root=tmp_path)
    fit = make_fake_fit(paths)
    monkeypatch.setattr(ptru_posterior, "PT100_SOURCE_FILES", ())

    metadata = ptru_posterior.build_pt100_run_metadata(paths, fit, sampler={})
    metadata["status"] = RUN_STATUS_COMPLETE

    fit.prior_config_path.write_text("priors: changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inputs.model_config_sha256"):
        ptru_posterior.validate_pt100_run_metadata(metadata, paths, fit)