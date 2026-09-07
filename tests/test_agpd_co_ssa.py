import numpy as np
import pytensor.tensor as pt

from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLHParameters,
    AgPdCOERParameters,
    build_agpd_point_state,
    evaluate_agpd_co_bf_er_lh,
    evaluate_agpd_co_bf_er_lh_capped,
    evaluate_agpd_co_er,
    log_equilibrium_constant,
    log_tst_rate_constant,
    solve_pd_co_ssa_qea_oh,
)
from mkm.model_inputs import ModelPointInputs


def _make_inputs():
    return ModelPointInputs(
        materials=("Ag10Pd90",),
        material_index=np.array([0, 0, 0]),
        E_V_SHE=np.array([0.00, 0.05, 0.10]),
        ln_electrolyte_concentration=np.log(np.array([0.50, 0.50, 0.50])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.01, 0.01])),
    )


def _make_config(theta_CO_max=0.66):
    return {
        "gas": {
            "standard_state_pressure_bar": 1.0,
            "total_pressure_bar": 1.0,
        },
        "electrolyte": {
            "standard_state_concentration_M": 1.0,
            "activity_model": "ideal_molarity",
        },
        "surface_composition": {
            "Ag10Pd90": {
                "Ag_fraction": 0.10,
                "Pd_fraction": 0.90,
            }
        },
        "co_coverage_cap": {
            "Ag10Pd90": float(theta_CO_max),
        },
    }


def _make_parameters():
    return AgPdCOBFERRLHParameters(
        deltaG1_0=-0.40,
        deltaG4_0=0.00,
        deltaG5_0=0.00,
        beta_2_BF=0.50,
        beta_2_ER=0.40,
        q=0.40,
        Gact1_0=0.70,
        Gact2_BF_0=0.70,
        Gact2_ER_0=0.72,
        Gact2_LH_0=0.75,
    )


def test_co_ssa_coverages_are_physical():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_co_bf_er_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    for name in ["theta_CO", "theta_OH_Pd", "theta_empty_Pd", "theta_OH_Ag", "theta_empty_Ag"]:
        values = pointwise[name].eval()
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0)
        assert np.all(values <= 1)


def test_co_ssa_site_balances():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_co_bf_er_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    Pd_total = (
        pointwise["theta_CO"].eval()
        + pointwise["theta_OH_Pd"].eval()
        + pointwise["theta_empty_Pd"].eval()
    )
    Ag_total = pointwise["theta_OH_Ag"].eval() + pointwise["theta_empty_Ag"].eval()

    np.testing.assert_allclose(Pd_total, 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(Ag_total, 1.0, rtol=1e-12, atol=1e-12)


def test_co_ssa_satisfies_steady_state_balance():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    parameters = _make_parameters()
    result = evaluate_agpd_co_bf_er_lh(state=state, parameters=parameters, temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    theta_CO = pointwise["theta_CO"].eval()
    theta_empty = pointwise["theta_empty_Pd"].eval()
    theta_OH_Ag = pointwise["theta_OH_Ag"].eval()
    log_k1 = result.log_k1.eval() if hasattr(result.log_k1, "eval") else np.asarray(result.log_k1)
    log_k_minus_1 = result.log_k_minus_1.eval()
    log_k2_BF = result.log_k2_BF.eval()
    log_k2_ER = result.log_k2_ER.eval()
    log_k2_LH = result.log_k2_LH.eval() if hasattr(result.log_k2_LH, "eval") else np.asarray(result.log_k2_LH)

    forward_CO = np.exp(log_k1) * np.exp(state.ln_a_CO) * theta_empty
    reverse_CO = np.exp(log_k_minus_1) * theta_CO
    BF = np.exp(log_k2_BF) * state.Ag_fraction * theta_OH_Ag * theta_CO
    ER = np.exp(log_k2_ER) * np.exp(state.ln_a_OH) * theta_CO
    theta_OH_Pd = pointwise["theta_OH_Pd"].eval()
    LH = np.exp(log_k2_LH) * state.Pd_fraction * theta_OH_Pd * theta_CO

    residual = forward_CO - reverse_CO - BF - ER - LH
    scale = forward_CO + reverse_CO + BF + ER + LH
    np.testing.assert_allclose(residual / scale, 0.0, atol=1e-10)


def test_co_ssa_total_rate_equals_pathway_sum():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_co_bf_er_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    expected = np.logaddexp(
        np.logaddexp(result.log_rate_BF.eval(), result.log_rate_ER.eval()), result.log_rate_LH.eval()
    )
    np.testing.assert_allclose(result.log_rate_total.eval(), expected, rtol=1e-12, atol=1e-12)


def test_co_ssa_pathway_fractions_sum_to_one():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_co_bf_er_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    total = (
        pointwise["rate_fraction_BF"].eval()
        + pointwise["rate_fraction_ER"].eval()
        + pointwise["rate_fraction_LH"].eval()
    )
    np.testing.assert_allclose(total, 1.0, rtol=1e-12, atol=1e-12)


def test_co_ssa_approaches_co_qea_when_consumption_is_slow():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    parameters = AgPdCOBFERRLHParameters(
        deltaG1_0=-0.20,
        deltaG4_0=0.00,
        deltaG5_0=0.00,
        beta_2_BF=0.50,
        beta_2_ER=0.50,
        q=0.40,
        Gact1_0=0.50,
        Gact2_BF_0=2.00,
        Gact2_ER_0=2.00,
        Gact2_LH_0=2.00,
    )

    result = evaluate_agpd_co_bf_er_lh(state=state, parameters=parameters, temperature_K=293.15)
    theta_CO_ssa = result.mechanism_result.pointwise["theta_CO"].eval()
    log_K1 = log_equilibrium_constant(delta_G_eV=-0.20, temperature_K=293.15).eval()
    deltaG4 = 0.0 - state.E_V_SHE
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=293.15).eval()

    term_CO = log_K1 + state.ln_a_CO
    term_OH = log_K4 + state.ln_a_OH
    denominator = 1.0 + np.exp(term_CO) + np.exp(term_OH)
    theta_CO_qea = np.exp(term_CO) / denominator
    np.testing.assert_allclose(theta_CO_ssa, theta_CO_qea, rtol=1e-6, atol=1e-8)


def test_no_lh_co_ssa_uses_exact_linear_solution():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    parameters = AgPdCOERParameters(
        deltaG1_0=-0.20,
        deltaG4_0=0.00,
        beta_2_ER=0.40,
        Gact1_0=0.60,
        Gact2_ER_0=0.72,
    )
    result = evaluate_agpd_co_er(state=state, parameters=parameters, temperature_K=293.15)
    pointwise = result.mechanism_result.pointwise

    theta_CO = pointwise["theta_CO"].eval()
    theta_empty = pointwise["theta_empty_Pd"].eval()
    theta_OH = pointwise["theta_OH_Pd"].eval()
    np.testing.assert_allclose(theta_CO + theta_empty + theta_OH, 1.0, rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(theta_CO))
    assert np.all(np.isfinite(result.mechanism_result.ln_rate.eval()))


def test_capped_co_ssa_limits_molecular_co_coverage():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(theta_CO_max=0.66))
    result = evaluate_agpd_co_bf_er_lh_capped(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    theta_CO = pointwise["theta_CO"].eval()
    theta_CO_site_occupation = pointwise["theta_CO_site_occupation"].eval()

    assert np.all(np.isfinite(theta_CO))
    assert np.all(theta_CO >= 0.0)
    assert np.all(theta_CO <= 0.66 + 1e-12)
    np.testing.assert_allclose(theta_CO_site_occupation, theta_CO / 0.66, rtol=1e-12, atol=1e-12)


def test_capped_co_ssa_uses_standard_pd_site_balance():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(theta_CO_max=0.66))
    result = evaluate_agpd_co_bf_er_lh_capped(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    Pd_total = (
        pointwise["theta_CO"].eval()
        + pointwise["theta_OH_Pd"].eval()
        + pointwise["theta_empty_Pd"].eval()
    )
    np.testing.assert_allclose(Pd_total, 1.0, rtol=1e-10, atol=1e-10)


def test_uncapped_model_ignores_configured_co_cap():
    state_low_cap = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(theta_CO_max=0.05))
    state_high_cap = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(theta_CO_max=0.66))
    parameters = _make_parameters()

    uncapped_low_cap = evaluate_agpd_co_bf_er_lh(state=state_low_cap, parameters=parameters, temperature_K=293.15)
    uncapped_high_cap = evaluate_agpd_co_bf_er_lh(state=state_high_cap, parameters=parameters, temperature_K=293.15)
    capped_low_cap = evaluate_agpd_co_bf_er_lh_capped(state=state_low_cap, parameters=parameters, temperature_K=293.15)

    theta_CO_uncapped_low = uncapped_low_cap.mechanism_result.pointwise["theta_CO"].eval()
    theta_CO_uncapped_high = uncapped_high_cap.mechanism_result.pointwise["theta_CO"].eval()
    theta_CO_capped_low = capped_low_cap.mechanism_result.pointwise["theta_CO"].eval()

    np.testing.assert_allclose(theta_CO_uncapped_low, theta_CO_uncapped_high, rtol=1e-12, atol=1e-12)
    assert np.all(theta_CO_capped_low <= 0.05 + 1e-12)


def test_capped_co_ssa_satisfies_packing_blocked_steady_state_balance():
    theta_CO_max = 0.66
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(theta_CO_max=theta_CO_max))
    parameters = _make_parameters()
    result = evaluate_agpd_co_bf_er_lh_capped(state=state, parameters=parameters, temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise
    theta_CO = pointwise["theta_CO"].eval()
    theta_empty = pointwise["theta_empty_Pd"].eval()
    theta_OH_Ag = pointwise["theta_OH_Ag"].eval()
    theta_OH_Pd = pointwise["theta_OH_Pd"].eval()
    log_k1 = result.log_k1.eval() if hasattr(result.log_k1, "eval") else np.asarray(result.log_k1)
    log_k_minus_1 = result.log_k_minus_1.eval()
    log_k2_BF = result.log_k2_BF.eval()
    log_k2_ER = result.log_k2_ER.eval()
    log_k2_LH = result.log_k2_LH.eval() if hasattr(result.log_k2_LH, "eval") else np.asarray(result.log_k2_LH)

    packing_availability = 1.0 - theta_CO / theta_CO_max
    forward_CO = np.exp(log_k1) * np.exp(state.ln_a_CO) * theta_empty * packing_availability
    reverse_CO = np.exp(log_k_minus_1) * theta_CO
    BF = np.exp(log_k2_BF) * state.Ag_fraction * theta_OH_Ag * theta_CO
    ER = np.exp(log_k2_ER) * np.exp(state.ln_a_OH) * theta_CO
    LH = np.exp(log_k2_LH) * state.Pd_fraction * theta_OH_Pd * theta_CO

    residual = forward_CO - reverse_CO - BF - ER - LH
    scale = forward_CO + reverse_CO + BF + ER + LH
    np.testing.assert_allclose(residual / scale, 0.0, atol=1e-10)


def test_capped_co_ssa_approaches_packing_qea_when_consumption_is_slow():
    theta_CO_max = 0.66
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(theta_CO_max=theta_CO_max))
    parameters = AgPdCOBFERRLHParameters(
        deltaG1_0=-0.20,
        deltaG4_0=0.00,
        deltaG5_0=0.00,
        beta_2_BF=0.50,
        beta_2_ER=0.50,
        q=0.40,
        Gact1_0=0.50,
        Gact2_BF_0=2.00,
        Gact2_ER_0=2.00,
        Gact2_LH_0=2.00,
    )

    result = evaluate_agpd_co_bf_er_lh_capped(state=state, parameters=parameters, temperature_K=293.15)
    theta_CO_ssa = result.mechanism_result.pointwise["theta_CO"].eval()
    log_K1 = log_equilibrium_constant(delta_G_eV=-0.20, temperature_K=293.15).eval()
    deltaG4 = 0.0 - state.E_V_SHE
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=293.15).eval()

    rho = np.exp(log_K1 + state.ln_a_CO)
    A_Pd = 1.0 + np.exp(log_K4 + state.ln_a_OH)
    inv_cap = 1.0 / theta_CO_max
    quadratic_a = rho * inv_cap
    quadratic_b = rho * (1.0 + inv_cap) + A_Pd
    discriminant = A_Pd**2 + 2.0 * rho * A_Pd * (1.0 + inv_cap) + rho**2 * (1.0 - inv_cap) ** 2
    theta_CO_qea = 2.0 * rho / (quadratic_b + np.sqrt(discriminant))

    np.testing.assert_allclose(theta_CO_ssa, theta_CO_qea, rtol=1e-6, atol=1e-8)
