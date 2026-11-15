"""Numerical contracts for posterior predictive rates and PSIS-weighted LOO-PIT."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import ndtr

from mkm.model_inputs import ModelInputArrays
from mkm.postprocessing.calibration import build_loo_pit_datatree, compute_normal_loo_pit
from mkm.postprocessing.predictions import build_observation_diagnostics, build_observation_distribution_draws


def _inputs():
    return ModelInputArrays(
        materials=("Pt", "PtRu"),
        condition_material_index=np.array([1, 0]),
        condition_ln_electrolyte_concentration=np.log([0.5, 0.5]),
        condition_ln_CO_mole_fraction=np.log([0.1, 0.1]),
        model_point_condition_index=np.array([0, 1]),
        model_point_E_V_SHE=np.array([0.2, 0.3]),
        observation_model_point_index=np.array([1, 0, 1]),
        observation_rate=np.array([1.2, 2.0, 0.9]),
    )


def _inference_data(*, material_error=True):
    # Coordinates are intentionally the reverse of inputs.materials.
    ln_rate = np.broadcast_to(np.log([2.0, 1.0]), (2, 3, 2)).copy()
    variables = {"ln_rate_model": (("chain", "draw", "model_point"), ln_rate)}
    coords = {"chain": [0, 1], "draw": [0, 1, 2], "model_point": [0, 1]}
    if material_error:
        coords["material"] = ["PtRu", "Pt"]
        variables["sigma_rate_abs"] = (
            ("material", "chain", "draw"), np.broadcast_to(np.array([0.2, 0.05])[:, None, None], (2, 2, 3)),
        )
        variables["sigma_rate_rel"] = (
            ("chain", "draw", "material"), np.broadcast_to([0.3, 0.1], (2, 3, 2)),
        )
    else:
        variables["sigma_rate_abs"] = (("chain", "draw"), np.full((2, 3), 0.1))
        variables["sigma_rate_rel"] = (("chain", "draw"), np.full((2, 3), 0.2))
    return SimpleNamespace(posterior=xr.Dataset(variables, coords=coords))


def test_material_error_draws_follow_material_coordinates_and_observation_mappings():
    draws = build_observation_distribution_draws(_inference_data(), _inputs(), sample_predictive=False)
    assert draws.model_rate.shape == (2, 3, 3)
    np.testing.assert_allclose(draws.model_rate[0, 0], [1.0, 2.0, 1.0])
    np.testing.assert_allclose(draws.sigma_rate[0, 0], [0.15, 0.8, 0.15])
    assert draws.predictive_rate is None


def test_shared_error_draws_broadcast_to_each_observation():
    draws = build_observation_distribution_draws(
        _inference_data(material_error=False), _inputs(), sample_predictive=False,
    )
    np.testing.assert_allclose(draws.sigma_rate[0, 0], [0.3, 0.5, 0.3])


def test_posterior_predictive_noise_is_seeded_and_uses_exact_rate_sigma():
    idata, inputs = _inference_data(), _inputs()
    first = build_observation_distribution_draws(idata, inputs, random_seed=31)
    again = build_observation_distribution_draws(idata, inputs, random_seed=31)
    different = build_observation_distribution_draws(idata, inputs, random_seed=32)
    np.testing.assert_array_equal(first.predictive_rate, again.predictive_rate)
    assert np.any(first.predictive_rate != different.predictive_rate)
    expected_noise = np.random.default_rng(31).standard_normal(first.model_rate.shape)
    np.testing.assert_allclose(first.predictive_rate, first.model_rate + first.sigma_rate * expected_noise)


def test_diagnostics_without_predictive_sampling_omit_predictive_columns():
    observations = pd.DataFrame({"rate_s_inv": [1.2, 2.0, 0.9], "observation_id": [0, 1, 2]})
    result = build_observation_diagnostics(
        _inference_data(), SimpleNamespace(observations=observations), _inputs(), sample_predictive=False,
    )
    np.testing.assert_allclose(result["rate_model_median"], [1.0, 2.0, 1.0])
    np.testing.assert_allclose(result["sigma_rate_median"], [0.15, 0.8, 0.15])
    np.testing.assert_allclose(result["residual"], [0.2, 0.0, -0.1])
    np.testing.assert_allclose(result["standardized_residual"], [0.2 / 0.15, 0.0, -0.1 / 0.15])
    assert not any("predictive" in column for column in result)
    pd.testing.assert_frame_equal(observations, pd.DataFrame({
        "rate_s_inv": [1.2, 2.0, 0.9], "observation_id": [0, 1, 2],
    }))


@pytest.mark.parametrize("variable", ["ln_rate_model", "sigma_rate_abs", "sigma_rate_rel"])
def test_missing_posterior_variables_are_rejected(variable):
    idata = _inference_data()
    idata.posterior = idata.posterior.drop_vars(variable)
    with pytest.raises(ValueError, match=f"missing.*{variable}"):
        build_observation_distribution_draws(idata, _inputs(), sample_predictive=False)


def test_missing_material_coordinate_is_rejected():
    idata = _inference_data()
    idata.posterior = idata.posterior.sel(material=["PtRu"])
    with pytest.raises(ValueError, match="missing materials"):
        build_observation_distribution_draws(idata, _inputs(), sample_predictive=False)


def test_unsupported_error_parameter_dimensions_are_rejected():
    idata = _inference_data()
    idata.posterior = idata.posterior.assign({
        "sigma_rate_abs": (("chain", "draw", "material", "extra"), np.ones((2, 3, 2, 1))),
    })
    with pytest.raises(ValueError, match="unsupported dimensions"):
        build_observation_distribution_draws(idata, _inputs(), sample_predictive=False)


@pytest.mark.parametrize("variable,value,message", [
    ("ln_rate_model", np.inf, "model rates must be finite and positive"),
    ("sigma_rate_abs", -10.0, "standard deviations must be finite and positive"),
])
def test_nonphysical_posterior_draws_are_rejected(variable, value, message):
    idata = _inference_data()
    idata.posterior[variable] = xr.full_like(idata.posterior[variable], value)
    with pytest.raises(ValueError, match=message):
        build_observation_distribution_draws(idata, _inputs(), sample_predictive=False)


def test_loo_pit_uses_normalized_psis_weights_and_linear_rate_cdf():
    idata = _inference_data(material_error=False)
    idata.posterior["ln_rate_model"] = idata.posterior["ln_rate_model"] + xr.DataArray(
        np.array([[0.0, 0.2, -0.1], [0.4, -0.3, 0.1]]), dims=("chain", "draw"),
    )
    observations = pd.DataFrame({"rate_s_inv": [1.1, 1.7, 1.0], "observation_id": [0, 1, 2]})
    weights = np.array([
        [[0.05, 0.1, 0.2], [0.15, 0.2, 0.1], [0.1, 0.1, 0.1]],
        [[0.1, 0.2, 0.1], [0.2, 0.2, 0.3], [0.4, 0.2, 0.2]],
    ])
    log_weights = xr.DataArray(np.log(weights).transpose(2, 1, 0), dims=("observation", "draw", "chain"))
    result = compute_normal_loo_pit(idata, SimpleNamespace(log_weights=log_weights), observations, _inputs())
    draws = build_observation_distribution_draws(idata, _inputs(), sample_predictive=False)
    cdf = ndtr((observations["rate_s_inv"].to_numpy()[None, None, :] - draws.model_rate) / draws.sigma_rate)
    expected = np.sum(cdf * weights / weights.sum(axis=(0, 1), keepdims=True), axis=(0, 1))
    np.testing.assert_allclose(result.pointwise["loo_pit"], expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.pointwise["loo_pit_centered"], expected - 0.5)
    assert result.summary.loc[0, "n_points"] == 3
    pd.testing.assert_frame_equal(observations, pd.DataFrame({
        "rate_s_inv": [1.1, 1.7, 1.0], "observation_id": [0, 1, 2],
    }))


def test_loo_pit_normalizes_large_negative_log_weights_without_underflow():
    observations = pd.DataFrame({"rate_s_inv": [1.0, 2.0, 1.0]})
    log_weights = np.full((2, 3, 3), -10000.0)
    log_weights[0, 1, :] = -9999.0
    result = compute_normal_loo_pit(
        _inference_data(material_error=False), SimpleNamespace(log_weights=log_weights), observations, _inputs(),
    )
    assert result.pointwise["loo_pit"].between(0.0, 1.0).all()
    assert np.isfinite(result.pointwise["loo_pit"]).all()


@pytest.mark.parametrize("log_weights,message", [
    (None, "does not contain PSIS log weights"),
    (np.zeros((4, 4)), "LOO log weights have shape"),
    (np.full((2, 3, 3), -np.inf), "no finite draw"),
    (np.full((2, 3, 3), np.nan), "contain NaN"),
    (np.full((2, 3, 3), np.inf), "contain NaN or \\+inf"),
])
def test_loo_pit_rejects_missing_mismatched_or_invalid_weights(log_weights, message):
    with pytest.raises(ValueError, match=message):
        compute_normal_loo_pit(
            _inference_data(), SimpleNamespace(log_weights=log_weights),
            pd.DataFrame({"rate_s_inv": [1.0, 2.0, 1.0]}), _inputs(),
        )


def test_loo_pit_rejects_observation_count_mismatch_and_nonfinite_observations():
    loo = SimpleNamespace(log_weights=np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match="observation table contains 2"):
        compute_normal_loo_pit(_inference_data(), loo, pd.DataFrame({"rate_s_inv": [1.0, 2.0]}), _inputs())
    with pytest.raises(ValueError, match="Observed rates contain non-finite values"):
        compute_normal_loo_pit(
            _inference_data(), loo, pd.DataFrame({"rate_s_inv": [1.0, np.nan, 2.0]}), _inputs(),
        )


def test_loo_pit_datatree_stores_observation_dimension():
    result = build_loo_pit_datatree([0.1, 0.4, 0.9], variable_name="rate_observed")
    np.testing.assert_array_equal(result["loo_pit"]["rate_observed"].values, [0.1, 0.4, 0.9])
    assert result["loo_pit"]["rate_observed"].dims == ("observation",)
