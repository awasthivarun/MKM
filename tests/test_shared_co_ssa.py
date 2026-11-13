"""Independent rate and site-balance checks for the common finite-CO solver."""

import numpy as np
import pytest

from mkm.mechanisms.agpd_basic.finite_co import solve_pd_co_ssa_qea_oh
from mkm.mechanisms.agpd_basic.state import AgPdPointState
from mkm.mechanisms.co_ssa import solve_co_ssa_qea_oh


def evaluate_coverages(*, lh=False, cap=None):
    # Three independent states span weak and strong CO/OH adsorption.
    term_oh = np.log(np.array([0.2, 1.0, 4.0]))
    k_ads = np.array([0.2, 2.0, 20.0])
    k_des = 1.3
    k_er = 0.7
    k_lh_app = np.array([0.1, 0.5, 2.0]) if lh else None
    solved = solve_co_ssa_qea_oh(
        term_oh, np.log(k_ads), np.log(k_des),
        log_k_other=(np.log(k_er),),
        log_k_lh_app=None if k_lh_app is None else np.log(k_lh_app),
        theta_co_max=cap,
    )
    empty = np.exp(solved.log_theta_empty.eval())
    co = np.exp(solved.log_theta_CO.eval())
    oh = np.exp(solved.log_theta_OH.eval())
    return term_oh, k_ads, k_des + k_er, k_lh_app, empty, co, oh


@pytest.mark.parametrize("lh", [False, True])
@pytest.mark.parametrize("cap", [None, 0.66])
def test_finite_co_solver_satisfies_site_and_steady_state_balances(lh, cap):
    term_oh, k_ads, k_cons, k_lh_app, empty, co, oh = evaluate_coverages(lh=lh, cap=cap)
    np.testing.assert_allclose(empty + co + oh, 1.0, rtol=0, atol=1e-12)
    for theta in (empty, co, oh):
        assert np.all(np.isfinite(theta))
        assert np.all((theta >= 0.0) & (theta <= 1.0))
    np.testing.assert_allclose(oh / empty, np.exp(term_oh), rtol=1e-12)
    if cap is not None:
        assert np.all(co <= cap + 1e-12)

    adsorption_factor = 1.0 if cap is None else 1.0 - co / cap
    forward = k_ads * empty * adsorption_factor
    reverse = k_cons * co
    if k_lh_app is not None:
        reverse = reverse + k_lh_app * empty * co
    np.testing.assert_allclose(forward, reverse, rtol=1e-11, atol=1e-12)


def test_agpd_adapter_and_common_solver_have_identical_coverages():
    term_oh = np.log(np.array([0.4, 1.0, 2.0]))
    state = AgPdPointState(
        E_V_SHE=np.array([0.0, 0.1, 0.2]),
        ln_a_OH=np.zeros(3),
        ln_a_CO=np.zeros(3),
        Ag_fraction=np.full(3, 0.5),
        Pd_fraction=np.full(3, 0.5),
    )
    shared = solve_co_ssa_qea_oh(
        term_oh, np.log(2.0), np.log(1.3),
        log_k_other=(np.log(0.7),), log_k_lh_app=np.log(0.5), theta_co_max=0.66,
    )
    agpd = solve_pd_co_ssa_qea_oh(
        log_K_OH_Pd=term_oh,
        log_k1_a_CO=np.log(2.0),
        log_k_minus_1=np.log(1.3),
        log_k_ER_app=np.log(0.7),
        log_k_LH_app=np.log(0.5),
        state=state,
        theta_CO_max=0.66,
    )
    for common, adapter in (
        (shared.log_theta_empty, agpd.log_theta_empty_Pd),
        (shared.log_theta_CO, agpd.log_theta_CO),
        (shared.log_theta_OH, agpd.log_theta_OH_Pd),
    ):
        np.testing.assert_allclose(common.eval(), adapter.eval(), rtol=1e-12, atol=1e-12)
