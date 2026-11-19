"""Independent root and local-stability tests for the fully finite Frumkin equations."""

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_agpd_lateral_stability.py"


def _diagnostic():
    spec = importlib.util.spec_from_file_location("diagnose_agpd_lateral_stability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rounded_parameters():
    return dict(deltaG1_0=-0.35545, deltaG4_0=-0.24523, deltaG5_0=0.13215,
                beta_2_BF=0.81423, q=0.79722, Gact2_BF_0=0.763,
                beta_2_ER=0.51383, Gact2_ER_0=0.80581, Gact2_LH_0=0.7067,
                Gact1_0=0.47384, deltaG3_0=-0.77012, Gact3_0=0.3772,
                beta_3=0.68646, Gact4_0=0.50529, beta_4=0.36802,
                Gact5_0=0.41221, beta_5=0.86829,
                z_CO_CO_eV=-0.073116, z_CO_OH_eV=0.012086, z_OH_OH_eV=0.0579)


def test_point115_has_three_residual_satisfying_roots_and_two_stable_states():
    diagnostic = _diagnostic()
    condition = (0.08, np.log(0.1), np.log(0.25), 0.5, 0.5)
    roots = diagnostic.find_distinct_roots(_rounded_parameters(), condition, 293.15, 0.65, n_starts=31)
    assert len(roots) == 3
    np.testing.assert_allclose([item["theta"][0] for item in roots], [0.15647, 0.35911, 0.81969], atol=2e-4)
    assert [item["stability"] for item in roots] == ["stable", "unstable", "stable"]
    assert all(item["error"] < 1e-9 for item in roots)
    for item in roots:
        c, p, h, s = item["theta"]
        assert 0.0 < 1.0 - c - p - h < 1.0
        assert 0.0 < s < 1.0
        rates = diagnostic.kinetic_balances(item["theta"], _rounded_parameters(), condition, 293.15, 0.65)[0]
        assert np.max(np.abs(rates)) < 1e-6


def test_complex_step_ode_jacobian_agrees_with_central_difference():
    diagnostic = _diagnostic()
    params = _rounded_parameters()
    condition = (0.08, np.log(0.1), np.log(0.25), 0.5, 0.5)
    theta = np.array([0.35, 0.64, 1e-7, 0.01])
    analytic = diagnostic.physical_jacobian(theta, params, condition, 293.15, 0.65)
    finite_difference = np.empty((4, 4))
    step = 1e-7
    for i in range(4):
        increment = np.zeros(4)
        increment[i] = step
        upper = diagnostic.kinetic_balances(theta + increment, params, condition, 293.15, 0.65)[0]
        lower = diagnostic.kinetic_balances(theta - increment, params, condition, 293.15, 0.65)[0]
        finite_difference[:, i] = (upper - lower) / (2 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=5e-5, atol=1e-3)


def test_saved_prior_parameter_values_are_used_without_rounding():
    diagnostic = _diagnostic()
    values = _rounded_parameters()
    prior = xr.Dataset({name: (("chain", "draw"), np.array([[value, value + 1e-8]]))
                        for name, value in values.items()})
    loaded = diagnostic.load_saved_parameter_draw(prior, "cxo_CO_BF_ER_LH_3_pOH_sOH", 0, 1)
    assert loaded == {name: value + 1e-8 for name, value in values.items()}


def test_point64_and_488_have_converged_stable_roots():
    diagnostic = _diagnostic()
    for E, co, oh in ((-0.02, 0.01, 0.25), (0.12, 1.0, 1.0)):
        condition = (E, np.log(co), np.log(oh), 0.5, 0.5)
        roots = diagnostic.find_distinct_roots(_rounded_parameters(), condition, 293.15, 0.65, n_starts=15)
        assert roots
        assert all(item["stability"] == "stable" and item["error"] < 1e-9 for item in roots)
