"""An inactive pathway has zero reported share without changing its zero rate."""

from dataclasses import fields

import numpy as np
import pytest
import pytensor

from mkm.mechanisms.agpd_basic import build_agpd_point_state
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_registry import get_agpd_model_definition


PARAMETERS = {
    "deltaG1_0": -0.20, "deltaG4_0": 0.02, "deltaG5_0": 0.02,
    "q": 0.5, "Gact1_0": 0.46, "Gact2_BF_0": 0.70,
    "Gact2_ER_0": 0.72, "Gact2_LH_0": 0.70,
    "beta_2_BF": 0.5, "beta_2_ER": 0.5,
}


def _evaluate(model_name):
    config = {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {
            "Pd100": {"Ag_fraction": 0.0, "Pd_fraction": 1.0},
            "Ag50Pd50": {"Ag_fraction": 0.5, "Pd_fraction": 0.5},
        },
    }
    inputs = ModelPointInputs(
        materials=("Pd100", "Ag50Pd50"), material_index=np.array([0, 1]),
        E_V_SHE=np.array([0.0, 0.0]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.1, 0.1])),
    )
    definition = get_agpd_model_definition(model_name)
    parameters = definition.parameter_class(**{field.name: PARAMETERS[field.name]
                                               for field in fields(definition.parameter_class)})
    evaluated = definition.evaluator(build_agpd_point_state(inputs, config), parameters, 293.15)
    result = getattr(evaluated, "mechanism_result", evaluated)
    keys = tuple(key for key in result.pointwise if key.startswith("rate_fraction_"))
    values = pytensor.function([], [result.ln_rate, *(result.pointwise[key] for key in keys)])()
    return np.asarray(values[0]), {key: np.asarray(value) for key, value in zip(keys, values[1:])}


@pytest.mark.parametrize("model_name", ("BF", "CO_BF"))
def test_bf_only_on_pure_pd_reports_zero_fraction_and_preserves_zero_rate(model_name):
    ln_rate, shares = _evaluate(model_name)
    assert np.isneginf(ln_rate[0])
    assert np.isfinite(ln_rate[1])
    np.testing.assert_array_equal(shares["rate_fraction_BF"], [0.0, 1.0])


@pytest.mark.parametrize("model_name", ("BF_ER", "CO_BF_ER"))
def test_inactive_bf_share_does_not_perturb_active_er_path(model_name):
    ln_rate, shares = _evaluate(model_name)
    assert np.all(np.isfinite(ln_rate))
    np.testing.assert_allclose(shares["rate_fraction_BF"][0], 0.0, atol=0.0)
    np.testing.assert_allclose(shares["rate_fraction_ER"][0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(sum(shares.values()), [1.0, 1.0], rtol=1e-12)
