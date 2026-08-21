import numpy as np
import pytensor.tensor as pt
import pytest

from mkm.mechanisms.agpd_basic import AgPdBFParameters, build_agpd_point_state, evaluate_agpd_bf
from mkm.model_inputs import ModelPointInputs


def _make_alloy_inputs():
    return ModelPointInputs(
        materials=("Ag10Pd90",),
        material_index=np.array([0, 0, 0]),
        E_V_SHE=np.array([0.00, 0.05, 0.10]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.01, 0.01])),
    )


def _make_alloy_config():
    return {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {"Ag10Pd90": {"Ag_fraction": 0.1, "Pd_fraction": 0.9}},
    }


def _make_parameters():
    return AgPdBFParameters(deltaG1_0=-0.4, deltaG4_0=0.0, deltaG5_0=0.0, beta_2=0.5, q=0.4, Gact2_0=0.7)


def test_bf_rate_is_finite():
    state = build_agpd_point_state(inputs=_make_alloy_inputs(), config=_make_alloy_config())
    result = evaluate_agpd_bf(state=state, parameters=_make_parameters(), temperature_K=293.15)

    ln_rate = result.mechanism_result.ln_rate.eval()

    assert np.all(np.isfinite(ln_rate))


def test_bf_coverages_are_physical():
    state = build_agpd_point_state(inputs=_make_alloy_inputs(), config=_make_alloy_config())
    result = evaluate_agpd_bf(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise

    for name in ["theta_CO", "theta_OH_Pd", "theta_empty_Pd", "theta_OH_Ag", "theta_empty_Ag"]:
        values = pointwise[name].eval()

        assert np.all(values >= 0)
        assert np.all(values <= 1)


def test_bf_site_balances():
    state = build_agpd_point_state(inputs=_make_alloy_inputs(), config=_make_alloy_config())
    result = evaluate_agpd_bf(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise

    theta_CO = pointwise["theta_CO"].eval()
    theta_OH_Pd = pointwise["theta_OH_Pd"].eval()
    theta_empty_Pd = pointwise["theta_empty_Pd"].eval()

    theta_OH_Ag = pointwise["theta_OH_Ag"].eval()
    theta_empty_Ag = pointwise["theta_empty_Ag"].eval()

    np.testing.assert_allclose(theta_CO + theta_OH_Pd + theta_empty_Pd, 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(theta_OH_Ag + theta_empty_Ag, 1.0, rtol=1e-12, atol=1e-12)


def test_bf_rate_scales_with_ag_fraction_at_fixed_state():
    inputs = _make_alloy_inputs()

    config_10 = _make_alloy_config()

    config_50 = _make_alloy_config()
    config_50["surface_composition"]["Ag10Pd90"]["Ag_fraction"] = 0.5
    config_50["surface_composition"]["Ag10Pd90"]["Pd_fraction"] = 0.5

    state_10 = build_agpd_point_state(inputs=inputs, config=config_10)
    state_50 = build_agpd_point_state(inputs=inputs, config=config_50)

    result_10 = evaluate_agpd_bf(state=state_10, parameters=_make_parameters(), temperature_K=293.15)
    result_50 = evaluate_agpd_bf(state=state_50, parameters=_make_parameters(), temperature_K=293.15)

    ln_rate_10 = result_10.log_rate_BF.eval()
    ln_rate_50 = result_50.log_rate_BF.eval()

    np.testing.assert_allclose(ln_rate_50 - ln_rate_10, np.log(0.5 / 0.1), rtol=1e-12, atol=1e-12)


def test_bf_only_rejects_zero_ag_fraction():
    inputs = ModelPointInputs(
        materials=("Pd100",),
        material_index=np.array([0]),
        E_V_SHE=np.array([0.0]),
        ln_electrolyte_concentration=np.log(np.array([0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01])),
    )

    config = {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {"Pd100": {"Ag_fraction": 0.0, "Pd_fraction": 1.0}},
    }

    state = build_agpd_point_state(inputs=inputs, config=config)

    with pytest.raises(ValueError, match="positive Ag"):
        evaluate_agpd_bf(state=state, parameters=_make_parameters(), temperature_K=293.15)