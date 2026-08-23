import numpy as np
import pandas as pd
import xarray as xr

from mkm.postprocessing.observables import (
    summarize_posterior_linear_observable,
    summarize_posterior_model_variable,
    summarize_pointwise_posterior_variable,
)
from mkm.observable_maps import LinearObservableMap
from types import SimpleNamespace

def _make_idata():
    ln_rate = np.array(
        [
            [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]],
            [[3.0, 4.0, 6.0], [4.0, 5.0, 7.0]],
        ]
    )

    posterior = xr.Dataset(
        {
            "ln_rate_model": (("chain", "draw", "model_point"), ln_rate),
            "rate_fraction_BF": (
                ("chain", "draw", "model_point"),
                np.full_like(ln_rate, 0.25),
            ),
        }
    )

    return {"posterior": posterior}


def test_posterior_model_variable_summary_aligns_with_model_points():
    idata = _make_idata()
    model_points = pd.DataFrame({"model_point_id": [0, 1, 2]})

    summary = summarize_posterior_model_variable(
        idata,
        model_points,
        "rate_fraction_BF",
    )

    assert len(summary) == 3
    np.testing.assert_allclose(summary["rate_fraction_BF_mean"], 0.25)


def test_posterior_linear_observable_preserves_chain_information():
    idata = _make_idata()

    observable_map = LinearObservableMap(
        outputs=pd.DataFrame({"observable_id": [0], "label": ["difference"]}),
        terms=pd.DataFrame(
            {
                "observable_id": [0, 0],
                "model_point_id": [0, 2],
                "coefficient": [-1.0, 1.0],
            }
        ),
    )

    summary = summarize_posterior_linear_observable(idata, observable_map)

    assert len(summary.pooled) == 1
    assert len(summary.by_chain) == 2
    np.testing.assert_allclose(summary.pooled.loc[0, "q50"], 3.0)

def test_inference_posterior_diagnostics_remains_compatible():
    from mkm.inference.posterior_diagnostics import (
        summarize_posterior_linear_observable as old_import,
    )
    from mkm.postprocessing.observables import (
        summarize_posterior_linear_observable as new_import,
    )

    assert old_import is new_import

def test_pointwise_summary_uses_generic_statistic_columns():
    posterior = xr.Dataset(
        {
            "theta_CO": (
                ("chain", "draw", "model_point"),
                np.array(
                    [
                        [[0.1, 0.2], [0.3, 0.4]],
                        [[0.2, 0.3], [0.4, 0.5]],
                    ]
                ),
            ),
        }
    )

    inference_data = SimpleNamespace(
        posterior=posterior
    )

    model_points = pd.DataFrame(
        {
            "model_point_id": [0, 1],
        }
    )

    result = summarize_pointwise_posterior_variable(
        inference_data=inference_data,
        model_points=model_points,
        variable_name="theta_CO",
    )

    assert {
        "mean",
        "sd",
        "q025",
        "q50",
        "q975",
    }.issubset(result.columns)

    assert "theta_CO_q50" not in result.columns

def test_model_variable_summary_retains_prefixed_columns():
    posterior = xr.Dataset(
        {
            "theta_CO": (
                ("chain", "draw", "model_point"),
                np.array(
                    [
                        [[0.1], [0.2]],
                        [[0.3], [0.4]],
                    ]
                ),
            ),
        }
    )

    inference_data = SimpleNamespace(
        posterior=posterior
    )

    model_points = pd.DataFrame(
        {
            "model_point_id": [0],
        }
    )

    result = summarize_posterior_model_variable(
        inference_data=inference_data,
        model_points=model_points,
        variable_name="theta_CO",
    )

    assert "theta_CO_q50" in result.columns
    assert "q50" not in result.columns