import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from mkm.inference.model import build_pymc_model
from mkm.mechanisms.base import MechanismResult
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays


def _build_test_inputs():
    records = []

    for replicate, offset in zip(["A", "B", "C"], [-0.1, 0.0, 0.1]):
        for grid_index in [0, 1, 2]:
            ln_rate = 1.0 + 2.0 * 0.01 * grid_index + offset

            records.append(
                {
                    "material": "M1",
                    "C_KOH_M": 0.25,
                    "CO_mole_fraction": 0.01,
                    "replicate": replicate,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": 0.01 * grid_index,
                    "rate_s_inv": np.exp(ln_rate),
                    "ln_rate": ln_rate,
                }
            )

    data = pd.DataFrame(records)

    model_data = build_model_data(selected_replicates=data, electrolyte_concentration_column="C_KOH_M")

    return build_model_input_arrays(model_data)


def _test_mechanism(inputs):
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    potential_slope = pm.Normal("potential_slope", mu=0.0, sigma=5.0)

    E = pt.as_tensor_variable(inputs.E_V_SHE)

    ln_rate = intercept + potential_slope * E
    theta_test = pt.sigmoid(ln_rate)

    return MechanismResult(ln_rate=ln_rate, pointwise={"theta_test": theta_test})


def _test_mechanism(inputs):
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    potential_slope = pm.Normal("potential_slope", mu=0.0, sigma=5.0)

    E = pt.as_tensor_variable(inputs.E_V_SHE)

    ln_rate = intercept + potential_slope * E
    theta_test = pt.sigmoid(ln_rate)

    return MechanismResult(ln_rate=ln_rate, pointwise={"theta_test": theta_test})


def test_build_pymc_model_registers_outputs():
    inputs = _build_test_inputs()

    built = build_pymc_model(inputs=inputs, mechanism=_test_mechanism)
    model = built.model

    assert "ln_rate_model" in model.named_vars
    assert "theta_test" in model.named_vars
    assert "sigma_ln_rate_material" in model.named_vars
    assert "ln_rate_observed" in model.named_vars


def test_built_model_initial_logp_is_finite():
    inputs = _build_test_inputs()

    built = build_pymc_model(inputs=inputs, mechanism=_test_mechanism)
    model = built.model

    initial_point = model.initial_point()
    logp = model.compile_logp()(initial_point)

    assert np.isfinite(logp)


def test_model_outputs_have_model_point_dimension():
    inputs = _build_test_inputs()

    built = build_pymc_model(inputs=inputs, mechanism=_test_mechanism)

    with built.model:
        prior = pm.sample_prior_predictive(draws=5, var_names=["ln_rate_model", "theta_test"], random_seed=123)

    assert prior.prior["ln_rate_model"].sizes["model_point"] == 3
    assert prior.prior["theta_test"].sizes["model_point"] == 3


def test_model_outputs_have_model_point_dimension():
    inputs = _build_test_inputs()

    built = build_pymc_model(inputs=inputs, mechanism=_test_mechanism)

    with built.model:
        prior = pm.sample_prior_predictive(draws=5, var_names=["ln_rate_model", "theta_test"], random_seed=123)

    assert prior.prior["ln_rate_model"].sizes["model_point"] == 3
    assert prior.prior["theta_test"].sizes["model_point"] == 3


def test_model_rejects_scalar_ln_rate():
    inputs = _build_test_inputs()

    def bad_mechanism(point_inputs):
        return MechanismResult(ln_rate=pm.Normal("single_rate", mu=0.0, sigma=1.0))

    with pytest.raises(ValueError, match="one-dimensional"):
        build_pymc_model(inputs=inputs, mechanism=bad_mechanism)


def test_model_rejects_wrong_number_of_rates():
    inputs = _build_test_inputs()

    def bad_mechanism(point_inputs):
        return MechanismResult(ln_rate=pt.zeros(2))

    with pytest.raises(ValueError, match="model points"):
        build_pymc_model(inputs=inputs, mechanism=bad_mechanism)


def test_model_requires_mechanism_result():
    inputs = _build_test_inputs()

    def bad_mechanism(point_inputs):
        return pt.zeros(len(point_inputs.E_V_SHE))

    with pytest.raises(TypeError, match="MechanismResult"):
        build_pymc_model(inputs=inputs, mechanism=bad_mechanism)

def test_build_pymc_model_supports_potential_correlated_likelihood():
    inputs = _build_test_inputs()

    built = build_pymc_model(
        inputs=inputs,
        mechanism=_test_mechanism,
        correlated_potential=True,
        correlation_length_prior_median_V=0.02,
        correlation_length_prior_log_sd=1.0,
    )

    assert "ell_E_V_material" in built.model.named_vars
    assert "ln_rate_observation_rho" in built.model.named_vars
    assert built.likelihood.correlation_length_material is not None

    initial_point = built.model.initial_point()
    logp = built.model.compile_logp()(initial_point)
    assert np.isfinite(logp)



def test_build_pymc_model_supports_rate_normal_likelihood():
    inputs = _build_test_inputs()

    built = build_pymc_model(
        inputs=inputs,
        mechanism=_test_mechanism,
        rate_normal=True,
        sigma_abs_prior_median_s_inv=2.0e-4,
        sigma_abs_prior_log_sd=1.0,
        sigma_rel_prior_median=0.18,
        sigma_rel_prior_log_sd=0.75,
    )

    assert "rate_model" in built.model.named_vars
    assert "sigma_rate_abs" in built.model.named_vars
    assert "sigma_rate_rel" in built.model.named_vars
    assert "rate_observation_sigma" in built.model.named_vars
    assert "rate_observed" in built.model.named_vars

    initial_point = built.model.initial_point()
    logp = built.model.compile_logp()(initial_point)
    assert np.isfinite(logp)


def test_rate_normal_cannot_be_combined_with_other_error_models():
    inputs = _build_test_inputs()

    with pytest.raises(ValueError, match="cannot be enabled together"):
        build_pymc_model(
            inputs=inputs,
            mechanism=_test_mechanism,
            rate_normal=True,
            correlated_potential=True,
        )
