from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.loo import compute_loo_diagnostics, compute_loo_reff


def test_compute_loo_diagnostics_maps_pointwise_values(monkeypatch):
    observations = pd.DataFrame({"observation_id": [0, 1], "ln_rate": [-1.0, -2.0]})
    fake_loo = SimpleNamespace(
        elpd=-3.0,
        se=0.5,
        p=1.2,
        n_samples=100,
        n_data_points=2,
        good_k=0.7,
        warning=False,
        elpd_i=np.array([-1.0, -2.0]),
        pareto_k=np.array([0.2, 0.8]),
    )

    captured = {}

    def fake_loo_call(*args, **kwargs):
        captured.update(kwargs)
        return fake_loo

    monkeypatch.setattr("mkm.postprocessing.loo.azs.loo", fake_loo_call)

    result = compute_loo_diagnostics(
        inference_data=object(),
        observations=observations,
        model_name="TEST",
    )

    assert result.summary.loc[0, "elpd"] == pytest.approx(-3.0)
    assert result.summary.loc[0, "max_pareto_k"] == pytest.approx(0.8)
    assert result.summary.loc[0, "n_pareto_k_above_good_k"] == 1
    assert result.pointwise["elpd_loo"].tolist() == pytest.approx([-1.0, -2.0])
    assert result.pointwise["pareto_k"].tolist() == pytest.approx([0.2, 0.8])
    assert "reff" not in captured


def test_compute_loo_reff_ignores_nonfinite_model_point_deterministics():
    posterior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.15, 0.25, 0.35, 0.45],
                    ]
                ),
            ),
            "sigma_rate_abs": (
                ("chain", "draw", "material"),
                np.full((2, 4, 1), 0.4),
            ),
            "ln_rate_BF": (
                ("chain", "draw", "model_point"),
                np.array(
                    [
                        [[-1.0, -np.inf], [-1.1, -np.inf], [-1.2, -np.inf], [-1.3, -np.inf]],
                        [[-0.9, -np.inf], [-1.0, -np.inf], [-1.1, -np.inf], [-1.2, -np.inf]],
                    ]
                ),
            ),
        },
        coords={
            "chain": [0, 1],
            "draw": [0, 1, 2, 3],
            "material": ["Ag10Pd90"],
            "model_point": [0, 1],
        },
    )
    idata = SimpleNamespace(posterior=posterior)

    reff = compute_loo_reff(idata)

    assert np.isfinite(reff)
    assert reff > 0


def test_compute_loo_diagnostics_passes_explicit_finite_reff(monkeypatch):
    posterior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.15, 0.25, 0.35, 0.45],
                    ]
                ),
            ),
            "ln_rate_BF": (
                ("chain", "draw", "model_point"),
                np.full((2, 4, 1), -np.inf),
            ),
        },
        coords={"chain": [0, 1], "draw": [0, 1, 2, 3], "model_point": [0]},
    )
    idata = SimpleNamespace(posterior=posterior)
    observations = pd.DataFrame({"observation_id": [0], "ln_rate": [-1.0]})
    fake_loo = SimpleNamespace(
        elpd=-1.0,
        se=0.1,
        p=0.5,
        n_samples=8,
        n_data_points=1,
        good_k=0.7,
        warning=False,
        elpd_i=np.array([-1.0]),
        pareto_k=np.array([0.1]),
    )

    captured = {}

    def fake_loo_call(*args, **kwargs):
        captured.update(kwargs)
        return fake_loo

    monkeypatch.setattr("mkm.postprocessing.loo.azs.loo", fake_loo_call)

    result = compute_loo_diagnostics(idata, observations, model_name="TEST")

    assert np.isfinite(captured["reff"])
    assert captured["reff"] > 0
    assert result.summary.loc[0, "reff"] == pytest.approx(captured["reff"])

def test_compute_loo_reff_supports_datatree_posterior():
    posterior = xr.Dataset(
        {
            "theta": (
                ("chain", "draw"),
                np.array(
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.15, 0.25, 0.35, 0.45],
                    ]
                ),
            ),
            "sigma_rate_abs": (
                ("chain", "draw", "material"),
                np.full((2, 4, 1), 0.4),
            ),
            "ln_rate_BF": (
                ("chain", "draw", "model_point"),
                np.full((2, 4, 1), -np.inf),
            ),
        },
        coords={
            "chain": [0, 1],
            "draw": [0, 1, 2, 3],
            "material": ["Ag10Pd90"],
            "model_point": [0],
        },
    )

    tree = xr.DataTree.from_dict({"/posterior": posterior})
    idata = SimpleNamespace(posterior=tree["posterior"])

    reff = compute_loo_reff(idata)

    assert np.isfinite(reff)
    assert reff > 0