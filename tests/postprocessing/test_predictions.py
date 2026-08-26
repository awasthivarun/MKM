from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.predictions import (
    build_observation_diagnostics,
    draw_mvn_posterior_predictive,
)


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

def test_draw_mvn_posterior_predictive_correlates_neighboring_potentials():
    n_samples = 6000
    conditional_mu = np.zeros((n_samples, 2))
    sigma = np.ones((n_samples, 2))
    correlation_length = np.full((n_samples, 2), 0.10)
    inputs = SimpleNamespace(
        observation_model_point_index=np.array([0, 1], dtype=np.int64),
        model_point_condition_index=np.array([0, 0], dtype=np.int64),
        model_point_E_V_SHE=np.array([0.00, 0.01]),
        observation_replicate=np.array(["A", "A"]),
    )

    predictive = draw_mvn_posterior_predictive(
        conditional_mu=conditional_mu,
        sigma=sigma,
        correlation_length=correlation_length,
        inputs=inputs,
        rng=np.random.default_rng(123),
    )

    correlation = np.corrcoef(predictive[:, 0], predictive[:, 1])[0, 1]
    assert correlation == pytest.approx(np.exp(-0.01 / 0.10), abs=0.03)


def test_build_observation_diagnostics_mvn():
    posterior = xr.Dataset(
        {
            "ln_rate_model": (
                ("chain", "draw", "model_point"),
                np.array([[[1.0, 2.0], [1.0, 2.0]]]),
            ),
            "sigma_ln_rate_material": (
                ("chain", "draw", "material"),
                np.full((1, 2, 1), 0.2),
            ),
            "ell_E_V_material": (
                ("chain", "draw", "material"),
                np.full((1, 2, 1), 0.03),
            ),
        }
    )
    inference_data = SimpleNamespace(posterior=posterior)

    observations = pd.DataFrame({"ln_rate": [1.1, 1.9]})
    model_data = SimpleNamespace(observations=observations)
    inputs = SimpleNamespace(
        observation_model_point_index=np.array([0, 1], dtype=np.int64),
        condition_material_index=np.array([0], dtype=np.int64),
        model_point_condition_index=np.array([0, 0], dtype=np.int64),
        model_point_E_V_SHE=np.array([0.00, 0.01]),
        observation_replicate=np.array(["A", "A"]),
        observation_setup_index=None,
    )

    result = build_observation_diagnostics(
        inference_data=inference_data,
        model_data=model_data,
        inputs=inputs,
        likelihood_name="mvn",
        random_seed=123,
    )

    assert result["ln_rate_mechanism_median"].tolist() == pytest.approx([1.0, 2.0])
    assert result["sigma_ln_rate_median"].tolist() == pytest.approx([0.2, 0.2])
    assert "observed_inside_predictive_95_hdi" in result.columns



def test_build_observation_diagnostics_rate_normal():
    posterior = xr.Dataset(
        {
            "ln_rate_model": (
                ("chain", "draw", "model_point"),
                np.array(
                    [
                        [[np.log(2.0), np.log(4.0)], [np.log(2.0), np.log(4.0)]],
                        [[np.log(2.0), np.log(4.0)], [np.log(2.0), np.log(4.0)]],
                    ]
                ),
            ),
            "sigma_rate_abs": (
                ("chain", "draw"),
                np.full((2, 2), 0.1),
            ),
            "sigma_rate_rel": (
                ("chain", "draw"),
                np.full((2, 2), 0.2),
            ),
        }
    )
    inference_data = SimpleNamespace(posterior=posterior)

    observations = pd.DataFrame(
        {
            "rate_s_inv": [2.5, 3.5],
            "ln_rate": np.log([2.5, 3.5]),
        }
    )
    model_data = SimpleNamespace(observations=observations)
    inputs = SimpleNamespace(
        observation_model_point_index=np.array([0, 1], dtype=np.int64),
        observation_setup_index=None,
    )

    result = build_observation_diagnostics(
        inference_data=inference_data,
        model_data=model_data,
        inputs=inputs,
        likelihood_name="rate_normal",
        random_seed=123,
    )

    assert result["rate_mechanism_median"].tolist() == pytest.approx([2.0, 4.0])
    assert result["rate_conditional_median"].tolist() == pytest.approx([2.0, 4.0])
    assert result["sigma_rate_median"].tolist() == pytest.approx([0.5, 0.9])
    assert result["residual_mechanism"].tolist() == pytest.approx([0.5, -0.5])
    assert result["residual_conditional"].tolist() == pytest.approx([0.5, -0.5])
    assert result["standardized_residual_conditional"].tolist() == pytest.approx([1.0, -0.5 / 0.9])
    assert "observed_inside_predictive_95_hdi" in result.columns
