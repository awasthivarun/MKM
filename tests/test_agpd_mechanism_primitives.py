import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from mkm.constants import H_J_S, K_B_EV_K, K_B_J_K
from mkm.mechanisms.agpd_basic import (
    build_agpd_point_state,
    calculate_ag_qea_coverages,
    calculate_pd_qea_coverages,
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_eyring_prefactor_s_inv,
    log_tst_rate_constant,
    thermal_energy_eV,
)
from mkm.model_inputs import ModelPointInputs


def _make_point_inputs():
    return ModelPointInputs(
        materials=("Pd100", "Ag10Pd90", "Ag50Pd50"),
        material_index=np.array([0, 1, 2]),
        E_V_SHE=np.array([0.00, 0.05, 0.10]),
        ln_electrolyte_concentration=np.log(np.array([0.25, 0.50, 1.00])),
        ln_CO_mole_fraction=np.log(np.array([0.001, 0.01, 0.10])),
    )


def _make_config():
    return {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {
            "Pd100": {"Ag_fraction": 0.0, "Pd_fraction": 1.0},
            "Ag10Pd90": {"Ag_fraction": 0.1, "Pd_fraction": 0.9},
            "Ag50Pd50": {"Ag_fraction": 0.5, "Pd_fraction": 0.5},
        },
    }


def test_thermal_energy():
    temperature_K = 293.15

    np.testing.assert_allclose(thermal_energy_eV(temperature_K), K_B_EV_K * temperature_K)


def test_eyring_prefactor():
    temperature_K = 293.15
    expected = np.log(K_B_J_K * temperature_K / H_J_S)

    np.testing.assert_allclose(log_eyring_prefactor_s_inv(temperature_K), expected)


def test_zero_free_energy_has_unit_equilibrium_constant():
    log_K = log_equilibrium_constant(delta_G_eV=0.0, temperature_K=293.15)

    np.testing.assert_allclose(log_K.eval(), 0.0)


def test_negative_free_energy_favors_equilibrium():
    log_K = log_equilibrium_constant(delta_G_eV=-0.10, temperature_K=293.15)

    assert log_K.eval() > 0


def test_electrochemical_free_energy_decreases_with_potential():
    E = np.array([0.0, 0.1])
    delta_G = electrochemical_free_energy(delta_G_0_eV=0.2, electron_transfer=1.0, E_V_SHE=E)

    np.testing.assert_allclose(delta_G.eval(), np.array([0.2, 0.1]))


def test_partial_charge_transfer_free_energy():
    E = np.array([0.0, 0.1])
    delta_G = electrochemical_free_energy(delta_G_0_eV=0.2, electron_transfer=0.4, E_V_SHE=E)

    np.testing.assert_allclose(delta_G.eval(), np.array([0.2, 0.16]))


def test_activation_energy_uses_beta():
    E = np.array([0.0, 0.1])
    activation_G = electrochemical_activation_energy(
        activation_G_0_eV=0.7,
        beta=0.5,
        electron_transfer=0.8,
        E_V_SHE=E,
    )

    np.testing.assert_allclose(activation_G.eval(), np.array([0.7, 0.66]))


def test_agpd_point_state_uses_dimensionless_activities():
    state = build_agpd_point_state(inputs=_make_point_inputs(), config=_make_config())

    np.testing.assert_allclose(state.ln_a_OH, np.log(np.array([0.25, 0.50, 1.00])))
    np.testing.assert_allclose(state.ln_a_CO, np.log(np.array([0.001, 0.01, 0.10])))


def test_total_pressure_changes_co_activity():
    config = _make_config()
    config["gas"]["total_pressure_bar"] = 0.85

    state = build_agpd_point_state(inputs=_make_point_inputs(), config=config)
    expected = _make_point_inputs().ln_CO_mole_fraction + np.log(0.85)

    np.testing.assert_allclose(state.ln_a_CO, expected)


def test_agpd_point_state_maps_surface_composition():
    state = build_agpd_point_state(inputs=_make_point_inputs(), config=_make_config())

    np.testing.assert_allclose(state.Ag_fraction, np.array([0.0, 0.1, 0.5]))
    np.testing.assert_allclose(state.Pd_fraction, np.array([1.0, 0.9, 0.5]))


def test_pd_qea_coverages_satisfy_site_balance():
    state = build_agpd_point_state(inputs=_make_point_inputs(), config=_make_config())

    coverages = calculate_pd_qea_coverages(
        log_K_CO=pt.as_tensor_variable(np.array([1.0, 1.0, 1.0])),
        log_K_OH_Pd=pt.as_tensor_variable(np.array([0.5, 0.5, 0.5])),
        state=state,
    )

    theta_empty = pt.exp(coverages.log_theta_empty_Pd)
    theta_CO = pt.exp(coverages.log_theta_CO)
    theta_OH = pt.exp(coverages.log_theta_OH_Pd)

    total = theta_empty + theta_CO + theta_OH

    np.testing.assert_allclose(total.eval(), 1.0, rtol=1e-12, atol=1e-12)


def test_ag_qea_coverages_satisfy_site_balance():
    state = build_agpd_point_state(inputs=_make_point_inputs(), config=_make_config())

    coverages = calculate_ag_qea_coverages(
        log_K_OH_Ag=pt.as_tensor_variable(np.array([0.5, 0.5, 0.5])),
        state=state,
    )

    theta_empty = pt.exp(coverages.log_theta_empty_Ag)
    theta_OH = pt.exp(coverages.log_theta_OH_Ag)

    total = theta_empty + theta_OH

    np.testing.assert_allclose(total.eval(), 1.0, rtol=1e-12, atol=1e-12)


def test_surface_fractions_must_sum_to_one():
    config = _make_config()
    config["surface_composition"]["Ag10Pd90"]["Pd_fraction"] = 0.8

    with pytest.raises(ValueError, match="sum to 1"):
        build_agpd_point_state(inputs=_make_point_inputs(), config=config)