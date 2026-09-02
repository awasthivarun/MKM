from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from mkm.inference.prior_predictive import (
    _flatten_draw_dimensions,
    _summarize_draws,
    summarize_prior_model_points,
    summarize_prior_observations,
    summarize_prior_parameters,
)


def test_flatten_and_summary_preserve_non_sample_axes():
    values = np.arange(24, dtype=float).reshape(2, 3, 4)
    flattened = _flatten_draw_dimensions(values)
    summary = _summarize_draws(values)

    assert flattened.shape == (6, 4)
    assert summary["mean"].shape == (4,)


def test_parameter_summary_labels_material_error_components():
    prior = xr.Dataset(
        {
            "x": (("chain", "draw"), np.ones((2, 3))),
            "sigma_rate_abs": (
                ("chain", "draw", "material"),
                np.ones((2, 3, 2)),
            ),
        },
        coords={"material": ["Ag10Pd90", "Pd100"]},
    )
    result = summarize_prior_parameters(
        SimpleNamespace(prior=prior),
        parameter_names=("x", "sigma_rate_abs"),
    )

    assert set(result["parameter"]) == {
        "x",
        "sigma_rate_abs[material=Ag10Pd90]",
        "sigma_rate_abs[material=Pd100]",
    }



def test_parameter_summary_preserves_singleton_chain_dimension():
    prior = xr.Dataset(
        {
            "x": (("chain", "draw"), np.array([[1.0, 2.0, 3.0]])),
        }
    )
    result = summarize_prior_parameters(
        SimpleNamespace(prior=prior),
        parameter_names=("x",),
    )

    assert result.loc[0, "parameter"] == "x"
    assert result.loc[0, "q50"] == 2.0


def test_model_point_summary_reports_log_and_linear_rate():
    ln_rate = np.log(np.array([[[1.0, 2.0], [2.0, 4.0]]]))
    prior = xr.Dataset(
        {"ln_rate_model": (("chain", "draw", "model_point"), ln_rate)}
    )
    points = pd.DataFrame({"model_point_id": [0, 1]})

    result = summarize_prior_model_points(SimpleNamespace(prior=prior), points)
    assert {"ln_rate_model_q50", "rate_model_q50"}.issubset(result.columns)
    np.testing.assert_allclose(result["rate_model_q50"], [1.5, 3.0])


def test_observation_summary_uses_rate_observed():
    predictive = xr.Dataset(
        {
            "rate_observed": (
                ("chain", "draw", "observation"),
                np.array([[[1.0, 2.0], [2.0, 4.0]]]),
            )
        }
    )
    observations = pd.DataFrame({"observation_id": [0, 1]})
    result = summarize_prior_observations(
        SimpleNamespace(prior_predictive=predictive),
        observations,
    )

    assert "rate_observed_q50" in result
    np.testing.assert_allclose(result["rate_observed_q50"], [1.5, 3.0])
