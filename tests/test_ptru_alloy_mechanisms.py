"""Independent surface balances and parameter contracts for PtRu alloys."""

from dataclasses import fields

import numpy as np
import pytest
import pytensor

from mkm.constants import K_B_EV_K
from mkm.mechanisms.ptru import PT100_PARAMETER_CLASSES, evaluate_pt100_co
from mkm.mechanisms.ptru import PTRU_ALLOY_PARAMETER_CLASSES, evaluate_ptru_alloy_co
from mkm.model_inputs import ModelPointInputs


TEMPERATURE_K = 293.15
PARAMETERS = {
    "deltaG1_0": -0.20,
    "deltaG4_0": 0.80,
    "deltaG5_0": 0.70,
    "Gact1_0": 0.58,
    "Gact2_BF_0": 0.80,
    "Gact2_ER_0": 0.85,
    "Gact2_LH_0": 0.80,
    "beta_2_ER": 0.45,
}


def make_state(medium, material="Pt66Ru33"):
    return ModelPointInputs(
        materials=(material,), material_index=np.zeros(3, dtype=int),
        E_V_SHE=np.array([0.62, 0.72, 0.82]) if medium == "acidic" else np.array([-0.04, 0.04, 0.12]),
        ln_electrolyte_concentration=np.log([0.25, 0.50, 1.00]),
        ln_CO_mole_fraction=np.log([0.01, 0.10, 1.00]),
    )


def evaluate(model_name, medium, ru_fraction=1 / 3):
    parameter_class = PTRU_ALLOY_PARAMETER_CLASSES[model_name]
    parameters = parameter_class(**{field.name: PARAMETERS[field.name] for field in fields(parameter_class)})
    result = evaluate_ptru_alloy_co(make_state(medium), parameters, TEMPERATURE_K, medium, model_name, ru_fraction)
    function = pytensor.function([], [result.ln_rate, *result.pointwise.values()])
    values = function()
    return values[0], dict(zip(result.pointwise, values[1:], strict=True))


@pytest.mark.parametrize("model_name", list(PTRU_ALLOY_PARAMETER_CLASSES))
def test_pathway_parameter_contract_has_no_partial_ru_discharge(model_name):
    fields_set = {field.name for field in fields(PTRU_ALLOY_PARAMETER_CLASSES[model_name])}
    assert "q" not in fields_set
    assert "beta_2_BF" not in fields_set
    assert ("deltaG5_0" in fields_set) == ("BF" in model_name.split("_"))
    assert ("Gact2_BF_0" in fields_set) == ("BF" in model_name.split("_"))


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model_name", list(PTRU_ALLOY_PARAMETER_CLASSES))
def test_coverages_and_pathway_fractions_are_normalized(medium, model_name):
    log_rate, point = evaluate(model_name, medium)
    assert np.isfinite(log_rate).all()
    np.testing.assert_allclose(point["theta_empty"] + point["theta_CO"] + point["theta_OH"], 1, atol=1e-10)
    np.testing.assert_allclose(point["theta_empty_Pt"], point["theta_empty"])
    np.testing.assert_allclose(point["theta_OH_Pt"], point["theta_OH"])
    if "BF" in model_name.split("_"):
        np.testing.assert_allclose(point["theta_empty_Ru"] + point["theta_OH_Ru"], 1, atol=1e-10)
        assert (point["theta_OH_Ru"] >= 0).all() and (point["theta_OH_Ru"] <= 1).all()
    else:
        assert "theta_OH_Ru" not in point
    fractions = [value for key, value in point.items() if key.startswith("rate_fraction_")]
    np.testing.assert_allclose(np.sum(fractions, axis=0), 1, atol=1e-10)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_ru_oh_coverage_has_full_one_electron_potential_slope(medium):
    _, point = evaluate("CO_BF", medium)
    logit = np.log(point["theta_OH_Ru"] / point["theta_empty_Ru"])
    state = make_state(medium)
    source = -state.ln_electrolyte_concentration if medium == "acidic" else state.ln_electrolyte_concentration
    expected = -(PARAMETERS["deltaG5_0"] - state.E_V_SHE) / (K_B_EV_K * TEMPERATURE_K) + source
    np.testing.assert_allclose(logit, expected, atol=1e-10)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model_name", list(PT100_PARAMETER_CLASSES))
def test_no_bf_alloy_kinetics_reduce_to_pt100(medium, model_name):
    alloy_rate, alloy_point = evaluate(model_name, medium)
    parameter_class = PT100_PARAMETER_CLASSES[model_name]
    parameters = parameter_class(**{field.name: PARAMETERS[field.name] for field in fields(parameter_class)})
    pt_result = evaluate_pt100_co(
        make_state(medium, material="Pt100"), parameters, TEMPERATURE_K, medium, model_name,
    )
    pt_rate = pytensor.function([], pt_result.ln_rate)()
    np.testing.assert_allclose(alloy_rate, pt_rate, atol=1e-10)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_bf_rate_uses_configured_ru_neighbor_fraction(medium):
    _, points_low = evaluate("CO_BF", medium, ru_fraction=0.25)
    _, points_high = evaluate("CO_BF", medium, ru_fraction=0.5)
    # The BF per-CO consumption coefficient scales linearly with Ru availability;
    # CO coverage itself shifts under finite-rate SSA, so compare rate / theta_CO.
    low = np.exp(points_low["ln_rate_BF"]) / points_low["theta_CO"]
    high = np.exp(points_high["ln_rate_BF"]) / points_high["theta_CO"]
    np.testing.assert_allclose(high / low, 2.0, atol=1e-10)
