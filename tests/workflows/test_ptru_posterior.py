from types import SimpleNamespace

import numpy as np
import xarray as xr

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

def test_pt100_loader_retains_all_draws_and_divergences(tmp_path, monkeypatch):
    n_chains, n_draws = 2, 1000
    posterior = xr.Dataset({
        "Gact1_0": (("chain", "draw"), np.full((n_chains, n_draws), 0.5)),
    })
    divergent = np.zeros((n_chains, n_draws), dtype=bool)
    divergent[0, 997] = True
    sample_stats = xr.Dataset({"diverging": (("chain", "draw"), divergent)})
    log_likelihood = xr.Dataset({
        "rate_observed": (("chain", "draw", "observation"), np.zeros((n_chains, n_draws, 2))),
    })
    inference_data = xr.DataTree.from_dict({
        "/posterior": posterior,
        "/sample_stats": sample_stats,
        "/log_likelihood": log_likelihood,
    })
    built = SimpleNamespace(
        model=SimpleNamespace(free_RVs=(SimpleNamespace(name="Gact1_0"),)),
        mechanism_result=SimpleNamespace(pointwise={"theta_CO": None}),
    )
    fit = SimpleNamespace(
        output_dir=tmp_path,
        built_model=built,
        specification=SimpleNamespace(medium="acidic"),
        model_data=None,
        inputs=None,
        parameter_specs={},
    )
    (tmp_path / "posterior.nc").touch()
    monkeypatch.setattr(ptru_posterior, "build_pt100_fit", lambda *_: fit)
    monkeypatch.setattr(ptru_posterior, "read_run_metadata", lambda *_: {})
    monkeypatch.setattr(ptru_posterior, "validate_pt100_run_metadata", lambda *_: None)
    monkeypatch.setattr(ptru_posterior, "load_inference_data", lambda *_: inference_data)

    def fake_deterministics(idata, *_args, **_kwargs):
        assert idata.posterior.sizes["draw"] == n_draws
        return idata.posterior.to_dataset().assign(
            ln_rate_model=(("chain", "draw"), np.zeros((n_chains, n_draws))),
            theta_CO=(("chain", "draw"), np.full((n_chains, n_draws), 0.5)),
        )

    monkeypatch.setattr(ptru_posterior, "compute_posterior_deterministics", fake_deterministics)
    run = ptru_posterior.load_pt100_posterior_run(
        SimpleNamespace(), "acidic", "CO_ER_LH", reconstruct_pointwise=True, progressbar=False,
    )

    for group_name in ("posterior", "sample_stats", "log_likelihood"):
        assert getattr(run.inference_data, group_name).sizes["draw"] == n_draws
    assert run.inference_data.sample_stats.to_dataset()["diverging"].to_numpy().sum() == 1
    assert "theta_CO" in run.inference_data.posterior


def test_pt100_loader_rejects_mismatched_sampling_draws(tmp_path, monkeypatch):
    posterior = xr.Dataset({"x": (("chain", "draw"), np.ones((2, 125)))})
    sample_stats = xr.Dataset({"diverging": (("chain", "draw"), np.ones((2, 1000), dtype=bool))})
    log_likelihood = xr.Dataset({
        "rate_observed": (("chain", "draw", "observation"), np.zeros((2, 125, 1))),
    })
    inference_data = xr.DataTree.from_dict({
        "/posterior": posterior,
        "/sample_stats": sample_stats,
        "/log_likelihood": log_likelihood,
    })
    built = SimpleNamespace(model=SimpleNamespace(free_RVs=(SimpleNamespace(name="x"),)))
    fit = SimpleNamespace(
        output_dir=tmp_path, built_model=built, specification=None,
        model_data=None, inputs=None, parameter_specs={},
    )
    (tmp_path / "posterior.nc").touch()
    monkeypatch.setattr(ptru_posterior, "build_pt100_fit", lambda *_: fit)
    monkeypatch.setattr(ptru_posterior, "read_run_metadata", lambda *_: {})
    monkeypatch.setattr(ptru_posterior, "validate_pt100_run_metadata", lambda *_: None)
    monkeypatch.setattr(ptru_posterior, "load_inference_data", lambda *_: inference_data)

    with pytest.raises(ValueError, match="different draw dimensions"):
        ptru_posterior.load_pt100_posterior_run(SimpleNamespace(), "acidic", "CO_ER_LH")
