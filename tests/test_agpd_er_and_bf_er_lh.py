import numpy as np

from mkm.mechanisms.agpd_basic import (
    AgPdBFERRLHParameters,
    AgPdERParameters,
    build_agpd_point_state,
    evaluate_agpd_bf_er_lh,
    evaluate_agpd_er,
)
from mkm.model_inputs import (
    ModelPointInputs,
)


def _make_inputs():
    return ModelPointInputs(
        materials=(
            "Ag10Pd90",
        ),
        material_index=np.array([
            0,
            0,
            0,
        ]),
        E_V_SHE=np.array([
            0.00,
            0.05,
            0.10,
        ]),
        ln_electrolyte_concentration=np.log(
            np.array([
                0.50,
                0.50,
                0.50,
            ])
        ),
        ln_CO_mole_fraction=np.log(
            np.array([
                0.01,
                0.01,
                0.01,
            ])
        ),
    )


def _make_config(
    Ag_fraction=0.10,
):
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
                "Ag_fraction": Ag_fraction,
                "Pd_fraction": (
                    1.0 - Ag_fraction
                ),
            },
        },
    }


def _make_er_parameters():
    return AgPdERParameters(
        deltaG1_0=-0.4,
        deltaG4_0=0.0,
        beta_2_ER=0.5,
        Gact2_ER_0=0.70,
    )


def _make_three_path_parameters():
    return AgPdBFERRLHParameters(
        deltaG1_0=-0.4,
        deltaG4_0=0.0,
        deltaG5_0=0.0,
        beta_2_BF=0.5,
        beta_2_ER=0.4,
        q=0.4,
        Gact2_BF_0=0.70,
        Gact2_ER_0=0.72,
        Gact2_LH_0=0.75,
    )


def test_er_rate_is_finite():
    state = build_agpd_point_state(
        inputs=_make_inputs(),
        config=_make_config(),
    )

    result = evaluate_agpd_er(
        state=state,
        parameters=_make_er_parameters(),
        temperature_K=293.15,
    )

    assert np.all(
        np.isfinite(
            result.log_rate_ER.eval()
        )
    )


def test_er_rate_has_no_composition_prefactor():
    inputs = _make_inputs()

    state_10 = build_agpd_point_state(
        inputs=inputs,
        config=_make_config(
            Ag_fraction=0.10
        ),
    )

    state_50 = build_agpd_point_state(
        inputs=inputs,
        config=_make_config(
            Ag_fraction=0.50
        ),
    )

    result_10 = evaluate_agpd_er(
        state=state_10,
        parameters=_make_er_parameters(),
        temperature_K=293.15,
    )

    result_50 = evaluate_agpd_er(
        state=state_50,
        parameters=_make_er_parameters(),
        temperature_K=293.15,
    )

    np.testing.assert_allclose(
        result_10.log_rate_ER.eval(),
        result_50.log_rate_ER.eval(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_three_path_total_equals_logsumexp():
    state = build_agpd_point_state(
        inputs=_make_inputs(),
        config=_make_config(),
    )

    result = evaluate_agpd_bf_er_lh(
        state=state,
        parameters=(
            _make_three_path_parameters()
        ),
        temperature_K=293.15,
    )

    expected = np.logaddexp(
        np.logaddexp(
            result.log_rate_BF.eval(),
            result.log_rate_ER.eval(),
        ),
        result.log_rate_LH.eval(),
    )

    np.testing.assert_allclose(
        result.log_rate_total.eval(),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_three_path_fractions_sum_to_one():
    state = build_agpd_point_state(
        inputs=_make_inputs(),
        config=_make_config(),
    )

    result = evaluate_agpd_bf_er_lh(
        state=state,
        parameters=(
            _make_three_path_parameters()
        ),
        temperature_K=293.15,
    )

    pointwise = (
        result
        .mechanism_result
        .pointwise
    )

    BF = pointwise[
        "rate_fraction_BF"
    ].eval()

    ER = pointwise[
        "rate_fraction_ER"
    ].eval()

    LH = pointwise[
        "rate_fraction_LH"
    ].eval()

    np.testing.assert_allclose(
        BF + ER + LH,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )

    for fraction in [
        BF,
        ER,
        LH,
    ]:
        assert np.all(
            fraction >= 0
        )

        assert np.all(
            fraction <= 1
        )


def test_three_path_pure_pd_removes_bf_only():
    state = build_agpd_point_state(
        inputs=_make_inputs(),
        config=_make_config(
            Ag_fraction=0.0
        ),
    )

    result = evaluate_agpd_bf_er_lh(
        state=state,
        parameters=(
            _make_three_path_parameters()
        ),
        temperature_K=293.15,
    )

    log_BF = (
        result.log_rate_BF.eval()
    )

    log_ER = (
        result.log_rate_ER.eval()
    )

    log_LH = (
        result.log_rate_LH.eval()
    )

    log_total = (
        result.log_rate_total.eval()
    )

    assert np.all(
        np.isneginf(
            log_BF
        )
    )

    expected = np.logaddexp(
        log_ER,
        log_LH,
    )

    np.testing.assert_allclose(
        log_total,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_three_path_site_balances_remain_physical():
    state = build_agpd_point_state(
        inputs=_make_inputs(),
        config=_make_config(),
    )

    result = evaluate_agpd_bf_er_lh(
        state=state,
        parameters=(
            _make_three_path_parameters()
        ),
        temperature_K=293.15,
    )

    pointwise = (
        result
        .mechanism_result
        .pointwise
    )

    theta_Pd_total = (
        pointwise[
            "theta_CO"
        ].eval()
        + pointwise[
            "theta_OH_Pd"
        ].eval()
        + pointwise[
            "theta_empty_Pd"
        ].eval()
    )

    theta_Ag_total = (
        pointwise[
            "theta_OH_Ag"
        ].eval()
        + pointwise[
            "theta_empty_Ag"
        ].eval()
    )

    np.testing.assert_allclose(
        theta_Pd_total,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        theta_Ag_total,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )