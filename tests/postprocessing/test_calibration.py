from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.calibration import build_loo_pit_datatree, compute_normal_loo_pit
from mkm.postprocessing.predictions import build_observation_distribution_draws


def _inputs():
    return SimpleNamespace(
        observation_model_point_index=np.array([0]),
        observation_setup_index=None,
        observation_material_index=np.array([0]),
        condition_material_index=np.array([0]),
        model_point_condition_index=np.array([0]),
    )


def test_observation_distribution_draws_iid():
    posterior = xr.Dataset(
        {
            "ln_rate_model": (("chain", "draw", "model_point"), np.array([[[-1.0], [1.0]]])),
            "sigma_ln_rate_material": (("chain", "draw", "material"), np.ones((1, 2, 1))),
        }
    )
    data = SimpleNamespace(posterior=posterior)

    result = build_observation_distribution_draws(data, _inputs(), "iid")

    assert result.conditional_mu.shape == (1, 2, 1)
    assert result.conditional_mu[:, :, 0].ravel().tolist() == pytest.approx([-1.0, 1.0])
    assert result.sigma[:, :, 0].ravel().tolist() == pytest.approx([1.0, 1.0])


def test_normal_loo_pit_is_half_for_symmetric_predictive_distribution():
    posterior = xr.Dataset(
        {
            "ln_rate_model": (("chain", "draw", "model_point"), np.array([[[-1.0], [1.0]]])),
            "sigma_ln_rate_material": (("chain", "draw", "material"), np.ones((1, 2, 1))),
        }
    )
    data = SimpleNamespace(posterior=posterior)

    log_weights = xr.DataArray(
        np.zeros((1, 2, 1)),
        dims=("chain", "draw", "observation"),
    )
    loo = SimpleNamespace(log_weights=log_weights)
    observations = pd.DataFrame({"ln_rate": [0.0]})

    result = compute_normal_loo_pit(
        inference_data=data,
        loo_result=loo,
        observations=observations,
        inputs=_inputs(),
        likelihood_name="iid",
    )

    assert result.pointwise.loc[0, "loo_pit"] == pytest.approx(0.5)


def test_loo_pit_datatree_uses_observation_dimension():
    data = build_loo_pit_datatree([0.1, 0.5, 0.9])

    assert "loo_pit" in data.children
    assert data["loo_pit"]["ln_rate_observed"].dims == ("observation",)
    assert data["loo_pit"]["ln_rate_observed"].values.tolist() == pytest.approx([0.1, 0.5, 0.9])