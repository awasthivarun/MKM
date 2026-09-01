import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from mkm.inference.model import build_pymc_model
from mkm.mechanisms.base import MechanismResult
from mkm.model_inputs import ModelInputArrays


def _inputs():
    return ModelInputArrays(
        materials=("M1",),
        condition_material_index=np.array([0], dtype=np.int64),
        condition_ln_electrolyte_concentration=np.log([0.25]),
        condition_ln_CO_mole_fraction=np.log([0.01]),
        model_point_condition_index=np.array([0, 0], dtype=np.int64),
        model_point_E_V_SHE=np.array([0.20, 0.30]),
        observation_model_point_index=np.array([0, 0, 1], dtype=np.int64),
        observation_rate=np.array([1.0, 1.1, 2.0]),
    )


def _likelihood_kwargs():
    return {
        "sigma_abs_prior_median_s_inv": 2.0e-4,
        "sigma_abs_prior_log_sd": 1.0,
        "sigma_rel_prior_median": 0.18,
        "sigma_rel_prior_log_sd": 0.75,
    }


def _mechanism(point_inputs):
    intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
    ln_rate = intercept + pt.as_tensor_variable(point_inputs.E_V_SHE)
    return MechanismResult(
        ln_rate=ln_rate,
        pointwise={"theta_CO": pt.sigmoid(ln_rate)},
    )


def test_build_pymc_model_registers_mechanism_and_rate_likelihood_outputs():
    built = build_pymc_model(
        _inputs(),
        _mechanism,
        error_structure="material",
        likelihood_kwargs=_likelihood_kwargs(),
    )

    assert {
        "intercept",
        "ln_rate_model",
        "theta_CO",
        "sigma_rate_abs",
        "sigma_rate_rel",
        "rate_observed",
    }.issubset(built.model.named_vars)
    assert built.model.named_vars_to_dims["ln_rate_model"] == ("model_point",)
    assert built.model.named_vars_to_dims["theta_CO"] == ("model_point",)
    assert built.model.named_vars_to_dims["rate_observed"] == ("observation",)
    assert built.likelihood_name == "rate_normal"
    assert built.error_structure == "material"


def test_built_model_initial_logp_is_finite():
    built = build_pymc_model(
        _inputs(),
        _mechanism,
        error_structure="material",
        likelihood_kwargs=_likelihood_kwargs(),
    )
    assert np.isfinite(built.model.compile_logp()(built.model.initial_point()))


def test_model_rejects_scalar_or_wrong_length_log_rate():
    def scalar_mechanism(_):
        return MechanismResult(ln_rate=pt.as_tensor_variable(0.0))

    def wrong_length_mechanism(_):
        return MechanismResult(ln_rate=pt.zeros(3))

    with pytest.raises(ValueError, match="one-dimensional"):
        build_pymc_model(
            _inputs(),
            scalar_mechanism,
            likelihood_kwargs=_likelihood_kwargs(),
        )
    with pytest.raises(ValueError, match="length"):
        build_pymc_model(
            _inputs(),
            wrong_length_mechanism,
            likelihood_kwargs=_likelihood_kwargs(),
        )


def test_model_requires_mechanism_result():
    with pytest.raises(TypeError, match="MechanismResult"):
        build_pymc_model(
            _inputs(),
            lambda point_inputs: pt.zeros(len(point_inputs.E_V_SHE)),
            likelihood_kwargs=_likelihood_kwargs(),
        )


def test_pointwise_name_cannot_conflict_with_existing_model_variable():
    def conflicting(point_inputs):
        parameter = pm.Normal("theta_CO", 0.0, 1.0)
        return MechanismResult(
            ln_rate=pt.zeros(len(point_inputs.E_V_SHE)),
            pointwise={"theta_CO": parameter + pt.zeros(len(point_inputs.E_V_SHE))},
        )

    with pytest.raises(ValueError, match="conflicts"):
        build_pymc_model(
            _inputs(),
            conflicting,
            likelihood_kwargs=_likelihood_kwargs(),
        )
