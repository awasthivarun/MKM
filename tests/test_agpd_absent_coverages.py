"""Absent Ag sites must not report an OH occupancy in mixed-material predictions."""

import numpy as np

from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLHParameters, build_agpd_point_state, evaluate_agpd_co_bf_er_lh,
)
from mkm.mechanisms.agpd_basic.surface import calculate_ag_qea_coverages
from mkm.model_inputs import ModelPointInputs


def test_ag_qea_coverage_is_empty_on_pd100_and_normal_on_alloy():
    inputs = ModelPointInputs(
        materials=("Pd100", "Ag10Pd90"), material_index=np.array([0, 1]),
        E_V_SHE=np.array([0.0, 0.0]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.01])),
    )
    config = {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"activity_model": "ideal_molarity", "standard_state_concentration_M": 1.0},
        "surface_composition": {
            "Pd100": {"Ag_fraction": 0.0, "Pd_fraction": 1.0},
            "Ag10Pd90": {"Ag_fraction": 0.1, "Pd_fraction": 0.9},
        },
    }
    state = build_agpd_point_state(inputs, config)
    coverages = calculate_ag_qea_coverages(log_K_OH_Ag=0.0, state=state)
    theta_oh = np.exp(coverages.log_theta_OH_Ag.eval())
    theta_empty = np.exp(coverages.log_theta_empty_Ag.eval())
    np.testing.assert_allclose(theta_oh, [0.0, 1.0 / 3.0], atol=1e-12)
    np.testing.assert_allclose(theta_empty, [1.0, 2.0 / 3.0], atol=1e-12)

    parameters = AgPdCOBFERRLHParameters(
        deltaG1_0=-0.2, deltaG4_0=0.0, deltaG5_0=0.0, q=0.5,
        beta_2_BF=0.5, beta_2_ER=0.5, Gact1_0=0.7, Gact2_BF_0=0.7,
        Gact2_ER_0=0.7, Gact2_LH_0=0.7,
    )
    result = evaluate_agpd_co_bf_er_lh(state, parameters, temperature_K=293.15)
    pointwise = result.mechanism_result.pointwise
    np.testing.assert_allclose(pointwise["theta_OH_Ag"].eval()[0], 0.0)
    np.testing.assert_allclose(pointwise["theta_empty_Ag"].eval()[0], 1.0)
    assert np.isfinite(result.mechanism_result.ln_rate.eval()).all()
