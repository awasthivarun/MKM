"""Guard model-point alignment and log-to-linear posterior summaries."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.model_points import summarize_model_points


def _model_points():
    return pd.DataFrame({"model_point_id": [0, 1], "material": ["Pt100", "PtRu"]})


def _posterior():
    return xr.Dataset({
        "ln_rate_model": (("chain", "draw", "model_point"), np.log(np.array([
            [[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [4.0, 8.0]],
        ]))),
        "theta_CO": (("chain", "draw", "model_point"), np.array([
            [[0.2, 0.5], [0.4, 0.6]], [[0.3, 0.7], [0.5, 0.8]],
        ])),
    })


def test_model_point_summary_keeps_point_order_and_summarizes_rates_in_linear_space():
    points = _model_points()
    original = points.copy(deep=True)
    summary = summarize_model_points(SimpleNamespace(posterior=_posterior()), points, ("theta_CO",))

    assert summary["material"].tolist() == ["Pt100", "PtRu"]
    np.testing.assert_allclose(summary["rate_model_median"], [2.5, 4.0])
    np.testing.assert_allclose(summary["ln_rate_model_median"], np.median(np.log([
        [1.0, 2.0], [3.0, 4.0], [2.0, 4.0], [4.0, 8.0],
    ]), axis=0))
    np.testing.assert_allclose(summary["theta_CO_median"], [0.35, 0.65])
    pd.testing.assert_frame_equal(points, original)


def test_model_point_summary_skips_missing_optional_variables_and_deduplicates_requested_names():
    summary = summarize_model_points(
        SimpleNamespace(posterior=_posterior()), _model_points(),
        ("theta_CO", "theta_CO", "absent_variable"),
    )
    assert len(summary.columns) == len(set(summary.columns))
    assert "theta_CO_mean" in summary
    assert not any("absent_variable" in column for column in summary)


@pytest.mark.parametrize("drop_name,message", [
    ("ln_rate_model", "missing 'ln_rate_model'"),
    ("theta_CO", None),
])
def test_model_point_summary_handles_missing_posterior_variables(drop_name, message):
    posterior = _posterior().drop_vars(drop_name)
    if message is not None:
        with pytest.raises(ValueError, match=message):
            summarize_model_points(SimpleNamespace(posterior=posterior), _model_points())
    else:
        summary = summarize_model_points(SimpleNamespace(posterior=posterior), _model_points(), ("theta_CO",))
        assert not any(column.startswith("theta_CO_") for column in summary)


@pytest.mark.parametrize("variable", ["ln_rate_model", "theta_CO"])
def test_model_point_summary_rejects_misaligned_posterior_variables(variable):
    posterior = _posterior()
    posterior = posterior.assign({variable: (("chain", "draw", "other_point"), np.ones((2, 2, 3)))})
    with pytest.raises(ValueError, match="align with model points"):
        summarize_model_points(SimpleNamespace(posterior=posterior), _model_points(), ("theta_CO",))
