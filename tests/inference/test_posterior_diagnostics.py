import numpy as np
import pandas as pd
import xarray as xr

from mkm.inference.posterior_diagnostics import (
    summarize_posterior_linear_observable,
    summarize_posterior_model_variable,
)
from mkm.observable_maps import LinearObservableMap


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