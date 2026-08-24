from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.predictions import build_observation_diagnostics


def test_build_observation_diagnostics_iid():
    posterior = xr.Dataset(
        {
            "ln_rate_model": (
                ("chain", "draw", "model_point"),
                np.array(
                    [
                        [[1.0, 2.0], [1.0, 2.0]],
                        [[1.0, 2.0], [1.0, 2.0]],
                    ]
                ),
            ),
            "sigma_ln_rate_material": (
                ("chain", "draw", "material"),
                np.full((2, 2, 1), 0.2),
            ),
        }
    )
    inference_data = SimpleNamespace(posterior=posterior)

    observations = pd.DataFrame({"ln_rate": [1.1, 1.9]})
    model_data = SimpleNamespace(observations=observations)

    inputs = SimpleNamespace(
        observation_model_point_index=np.array([0, 1], dtype=np.int64),
        observation_material_index=np.array([0, 0], dtype=np.int64),
        condition_material_index=np.array([0], dtype=np.int64),
        model_point_condition_index=np.array([0, 0], dtype=np.int64),
        observation_setup_index=None,
    )

    result = build_observation_diagnostics(
        inference_data=inference_data,
        model_data=model_data,
        inputs=inputs,
        likelihood_name="iid",
        random_seed=123,
    )

    assert result["ln_rate_mechanism_median"].tolist() == pytest.approx([1.0, 2.0])
    assert result["ln_rate_conditional_median"].tolist() == pytest.approx([1.0, 2.0])
    assert result["ln_rate_mechanism_hdi95_lower"].tolist() == pytest.approx([1.0, 2.0])
    assert result["ln_rate_mechanism_hdi95_upper"].tolist() == pytest.approx([1.0, 2.0])

    assert result["rate_mechanism_median"].tolist() == pytest.approx(
        [np.exp(1.0), np.exp(2.0)]
    )
    assert result["rate"].tolist() == pytest.approx(
        [np.exp(1.1), np.exp(1.9)]
    )

    assert result["residual_conditional"].tolist() == pytest.approx([0.1, -0.1])
    assert "observed_inside_predictive_95_hdi" in result.columns


def test_build_observation_diagnostics_rejects_unknown_likelihood():
    posterior = xr.Dataset(
        {
            "ln_rate_model": (
                ("chain", "draw", "model_point"),
                np.zeros((1, 2, 1)),
            ),
        }
    )

    inference_data = SimpleNamespace(posterior=posterior)
    model_data = SimpleNamespace(observations=pd.DataFrame({"ln_rate": [0.0]}))
    inputs = SimpleNamespace(
        observation_model_point_index=np.array([0]),
        observation_setup_index=None,
    )

    with pytest.raises(ValueError, match="Unsupported likelihood"):
        build_observation_diagnostics(
            inference_data=inference_data,
            model_data=model_data,
            inputs=inputs,
            likelihood_name="banana",
        )
