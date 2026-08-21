import numpy as np

from mkm.mechanisms.agpd_basic import (
    AgPdBFLHParameters,
    build_agpd_point_state,
    evaluate_agpd_bf_lh,
    logsumexp_pathways,
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


def _make_config(Ag_fraction=0.10):
    return {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {"Ag10Pd90": {"Ag_fraction": Ag_fraction, "Pd_fraction": 1.0 - Ag_fraction}},
    }


def _make_parameters():
    return AgPdBFLHParameters(
        deltaG1_0=-0.4,
        deltaG4_0=0.0,
        deltaG5_0=0.0,
        beta_2=0.5,
        q=0.4,
        Gact2_BF_0=0.70,
        Gact2_LH_0=0.75,
    )


def test_logsumexp_pathways_is_numerically_stable():
    log_rate_1 = np.array([1000.0])
    log_rate_2 = np.array([999.0])

    result = logsumexp_pathways(log_rate_1, log_rate_2).eval()
    expected = 1000.0 + np.log1p(np.exp(-1.0))

    np.testing.assert_allclose(result, expected)


def test_bf_lh_total_rate_is_finite():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_bf_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    ln_rate = result.log_rate_total.eval()

    assert np.all(np.isfinite(ln_rate))


def test_bf_lh_total_equals_sum_of_pathways():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_bf_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    log_total = result.log_rate_total.eval()
    log_BF = result.log_rate_BF.eval()
    log_LH = result.log_rate_LH.eval()

    np.testing.assert_allclose(log_total, np.logaddexp(log_BF, log_LH), rtol=1e-12, atol=1e-12)


def test_bf_lh_pathway_fractions_sum_to_one():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_bf_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise

    BF_fraction = pointwise["rate_fraction_BF"].eval()
    LH_fraction = pointwise["rate_fraction_LH"].eval()

    np.testing.assert_allclose(BF_fraction + LH_fraction, 1.0, rtol=1e-12, atol=1e-12)

    assert np.all(BF_fraction >= 0)
    assert np.all(BF_fraction <= 1)
    assert np.all(LH_fraction >= 0)
    assert np.all(LH_fraction <= 1)


def test_bf_lh_pure_pd_limit_reduces_to_lh():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config(Ag_fraction=0.0))
    result = evaluate_agpd_bf_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    log_total = result.log_rate_total.eval()
    log_LH = result.log_rate_LH.eval()
    log_BF = result.log_rate_BF.eval()

    np.testing.assert_allclose(log_total, log_LH, rtol=1e-12, atol=1e-12)
    assert np.all(np.isneginf(log_BF))


def test_bf_lh_coverages_remain_physical():
    state = build_agpd_point_state(inputs=_make_inputs(), config=_make_config())
    result = evaluate_agpd_bf_lh(state=state, parameters=_make_parameters(), temperature_K=293.15)

    pointwise = result.mechanism_result.pointwise

    theta_CO = pointwise["theta_CO"].eval()
    theta_OH_Pd = pointwise["theta_OH_Pd"].eval()
    theta_empty_Pd = pointwise["theta_empty_Pd"].eval()

    theta_OH_Ag = pointwise["theta_OH_Ag"].eval()
    theta_empty_Ag = pointwise["theta_empty_Ag"].eval()

    np.testing.assert_allclose(theta_CO + theta_OH_Pd + theta_empty_Pd, 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(theta_OH_Ag + theta_empty_Ag, 1.0, rtol=1e-12, atol=1e-12)