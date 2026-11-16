from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    run_posterior_lifecycle,
)


def test_posterior_lifecycle_owns_finalization_and_parameter_summary(tmp_path, monkeypatch):
    posterior = xr.Dataset(
        {
            "theta": (("chain", "draw"), np.array([[1.0, 2.0]])),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    inference_data = SimpleNamespace(posterior=posterior)
    built = SimpleNamespace(
        model=SimpleNamespace(free_RVs=[SimpleNamespace(name="theta")]),
    )

    monkeypatch.setattr(
        "mkm.workflows.posterior_lifecycle.sample_posterior",
        lambda *args, **kwargs: inference_data,
    )
    monkeypatch.setattr(
        "mkm.workflows.posterior_lifecycle.write_inference_data",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "mkm.workflows.posterior_lifecycle.build_sampling_health",
        lambda *args, **kwargs: {
            "n_divergent": 0,
            "max_rhat": 1.0,
            "min_ess_bulk": 100.0,
            "min_ess_tail": 100.0,
        },
    )
    monkeypatch.setattr(
        "mkm.workflows.posterior_lifecycle.compute_posterior_deterministics",
        lambda *args, **kwargs: xr.Dataset(
            {
                "theta": (("chain", "draw"), np.array([[1.0, 2.0]])),
                "ln_rate_model": (
                    ("chain", "draw", "model_point"),
                    np.array([[[0.0], [0.1]]]),
                ),
            },
            coords={"chain": [0], "draw": [0, 1], "model_point": [0]},
        ),
    )

    def add_log_likelihood(data, *args, **kwargs):
        data.log_likelihood = xr.Dataset(
            {
                "rate_observed": (
                    ("chain", "draw", "observation"),
                    np.array([[[-1.0], [-1.1]]]),
                )
            },
            coords={"chain": [0], "draw": [0, 1], "observation": [0]},
        )
        return data

    monkeypatch.setattr(
        "mkm.workflows.posterior_lifecycle.add_log_likelihood",
        add_log_likelihood,
    )
    monkeypatch.setattr(
        "mkm.workflows.posterior_lifecycle.build_posterior_parameter_summary",
        lambda *args, **kwargs: pd.DataFrame(
            [{"parameter": "theta", "mean": 1.5}]
        ),
    )

    result = run_posterior_lifecycle(
        built=built,
        output_dir=tmp_path,
        parameter_specs={"theta": {"distribution": "normal"}},
        sampler={
            "draws": 2,
            "tune": 0,
            "chains": 1,
            "cores": 1,
            "target_accept": 0.9,
            "random_seed": 1,
            "nuts_sampler": "nutpie",
            "backend": "numba",
        },
        metadata_factory=lambda: {"fit_scope": "individual"},
        checkpoint_validator=lambda metadata: None,
        progressbar=False,
    )

    assert result.metadata["status"] == RUN_STATUS_COMPLETE
    assert tuple(result.inference_data.posterior.data_vars) == ("theta",)
    assert "rate_observed" in result.inference_data.log_likelihood

    summary = pd.read_csv(tmp_path / "posterior_parameters.csv")
    assert summary.to_dict("records") == [{"parameter": "theta", "mean": 1.5}]


def test_lifecycle_records_sampling_identity_before_sampler_and_rejects_change(tmp_path, monkeypatch):
    from mkm.workflows import posterior_lifecycle as lifecycle
    from mkm.provenance import sha256_file
    source = tmp_path / "source.py"
    source.write_text("before")
    seen = []

    def write_metadata(data, output_path):
        seen.append((data["status"], data["source_sha256"]))

    def sample(*_args, **_kwargs):
        assert seen and seen[0][0] == lifecycle.RUN_STATUS_SAMPLING
        source.write_text("after")
        return SimpleNamespace(posterior=xr.Dataset({"theta": (("chain", "draw"), [[1.0]])}))

    monkeypatch.setattr(lifecycle, "write_run_metadata", write_metadata)
    monkeypatch.setattr(lifecycle, "sample_posterior", sample)
    monkeypatch.setattr(lifecycle, "write_inference_data", lambda *_: None)
    built = SimpleNamespace(model=SimpleNamespace(free_RVs=[SimpleNamespace(name="theta")]))
    sampler = {"draws": 1, "tune": 0, "chains": 1, "cores": 1, "target_accept": 0.9,
               "random_seed": 1, "nuts_sampler": None, "backend": "numba"}

    def validate(data):
        if data["source_sha256"] != sha256_file(source):
            raise ValueError("source mismatch")

    import pytest
    with pytest.raises(ValueError, match="source mismatch"):
        lifecycle.run_posterior_lifecycle(
            built=built, output_dir=tmp_path / "fit", parameter_specs={}, sampler=sampler,
            metadata_factory=lambda: {"source_sha256": sha256_file(source)},
            checkpoint_validator=validate,
        )
    assert len(seen) == 1
    assert seen[0][0] == lifecycle.RUN_STATUS_SAMPLING


def test_resume_recovers_atomic_checkpoint_with_interrupted_status_write(tmp_path, monkeypatch):
    from mkm.workflows import posterior_lifecycle as lifecycle
    from mkm.provenance import read_run_metadata, write_run_metadata
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "posterior.nc").write_bytes(b"atomic checkpoint")
    write_run_metadata({"status": lifecycle.RUN_STATUS_SAMPLING, "identity": "unchanged"},
                       directory / "run_metadata.yaml")
    inference_data = SimpleNamespace(posterior=xr.Dataset({"theta": (("chain", "draw"), [[1.0]])}))
    monkeypatch.setattr(lifecycle, "load_inference_data", lambda *_: inference_data)
    sampler = {"draws": 1, "tune": 0, "chains": 1, "cores": 1, "target_accept": 0.9,
               "random_seed": 1, "nuts_sampler": None, "backend": "numba"}

    def validate(metadata):
        assert metadata["status"] == lifecycle.RUN_STATUS_SAMPLED
        assert metadata["identity"] == "unchanged"

    _, metadata, _, _, _ = lifecycle._sample_or_load(
        built=None, output_dir=directory, sampler=sampler, resume=True, overwrite=False,
        metadata_factory=lambda: None, checkpoint_validator=validate,
    )
    assert metadata["status"] == lifecycle.RUN_STATUS_SAMPLED
    assert read_run_metadata(directory / "run_metadata.yaml")["status"] == lifecycle.RUN_STATUS_SAMPLED
