from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from mkm.inference.posterior import (
    add_log_likelihood,
    compute_posterior_deterministics,
    load_inference_data,
    sample_posterior,
    summarize_sampler_health,
    write_inference_data,
)


def _make_model():
    with pm.Model() as model:
        x = pm.Normal("x")
        pm.Deterministic("twice_x", 2 * x)
        pm.Normal("observed", mu=x, sigma=1, observed=np.array([0.1, -0.1]))

    return SimpleNamespace(model=model)


def test_posterior_sampling_stores_free_variables_only():
    built = _make_model()
    idata = sample_posterior(
        built,
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        random_seed=123,
        nuts_sampler="pymc",
        compute_convergence_checks=False,
    )

    assert "x" in idata.posterior
    assert "twice_x" not in idata.posterior


def test_posterior_deterministics_can_be_reconstructed():
    built = _make_model()
    idata = sample_posterior(
        built,
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        random_seed=123,
        nuts_sampler="pymc",
        compute_convergence_checks=False,
    )

    posterior = compute_posterior_deterministics(
        idata,
        built,
        var_names=["twice_x"],
        progressbar=False,
    )

    assert "twice_x" in posterior
    np.testing.assert_allclose(posterior["twice_x"], 2 * posterior["x"])


def test_log_likelihood_can_be_added_after_sampling():
    built = _make_model()
    idata = sample_posterior(
        built,
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        random_seed=123,
        nuts_sampler="pymc",
        compute_convergence_checks=False,
    )

    idata = add_log_likelihood(idata, built, progressbar=False)

    assert hasattr(idata, "log_likelihood") or "log_likelihood" in idata


def test_posterior_sampling_validates_configuration():
    built = _make_model()

    with pytest.raises(ValueError, match="draws"):
        sample_posterior(built, draws=0)

    with pytest.raises(ValueError, match="target_accept"):
        sample_posterior(built, target_accept=1.0)


def test_sampler_health_reports_divergences():
    built = _make_model()
    idata = sample_posterior(
        built,
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        random_seed=123,
        nuts_sampler="pymc",
        compute_convergence_checks=False,
    )

    health = summarize_sampler_health(idata)

    assert health.divergences >= 0


def test_inference_data_checkpoint_can_be_loaded_and_atomically_replaced(tmp_path):
    inference_data = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {"x": (("chain", "draw"), np.array([[1.0, 2.0]]))}
            )
        }
    )
    posterior_path = tmp_path / "posterior.nc"

    write_inference_data(inference_data, posterior_path)
    loaded = load_inference_data(posterior_path)
    write_inference_data(loaded, posterior_path)
    reloaded = load_inference_data(posterior_path)

    np.testing.assert_allclose(reloaded.posterior["x"], [[1.0, 2.0]])
    assert not (tmp_path / ".posterior.nc.tmp").exists()
