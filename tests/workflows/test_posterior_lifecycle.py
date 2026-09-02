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
