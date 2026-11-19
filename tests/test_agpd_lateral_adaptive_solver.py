"""Adaptive Frumkin solver regressions: prior-draw failures, gradients, and branch safeguards."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytensor
import pytensor.tensor as pt

from mkm.mechanisms.agpd_basic.lateral_interactions import evaluate_agpd_lateral_interactions
from mkm.models.agpd_physics import parse_agpd_physics
from mkm.models.agpd_registry import get_agpd_model_definition


DRAW_2 = dict(
    deltaG1_0=-0.35545, deltaG4_0=-0.24523, deltaG5_0=0.13215, beta_2_BF=0.81423, q=0.79722,
    Gact2_BF_0=0.763, beta_2_ER=0.51383, Gact2_ER_0=0.80581, Gact2_LH_0=0.7067, Gact1_0=0.47384,
    deltaG3_0=-0.77012, Gact3_0=0.3772, beta_3=0.68646, Gact4_0=0.50529, beta_4=0.36802,
    Gact5_0=0.41221, beta_5=0.86829, z_CO_CO_eV=0.0045052, z_CO_OH_eV=0.012086, z_OH_OH_eV=0.0579,
)
DRAW_30 = dict(DRAW_2, deltaG1_0=-0.40672, deltaG4_0=-0.18145, deltaG5_0=0.14354,
               beta_2_BF=0.57099, q=0.66613, Gact2_BF_0=0.77199, beta_2_ER=0.15346,
               Gact2_ER_0=0.74522, Gact2_LH_0=0.60069, Gact1_0=0.47551,
               deltaG3_0=-0.629, Gact3_0=0.21744, beta_3=0.39265, Gact4_0=0.40237,
               beta_4=0.32746, Gact5_0=0.42875, beta_5=0.35241,
               z_CO_CO_eV=0.047036, z_CO_OH_eV=-0.066729, z_OH_OH_eV=-0.024417)


def _evaluate(name, values, potential, co_activity, oh_activity, **settings):
    definition = get_agpd_model_definition(name)
    parameters = definition.parameter_class(**{field.name: values[field.name] for field in fields(definition.parameter_class)})
    state = SimpleNamespace(
        E_V_SHE=np.asarray(potential, dtype=float), ln_a_CO=np.log(co_activity), ln_a_OH=np.log(oh_activity),
        Pd_fraction=np.full(len(potential), 0.5), Ag_fraction=np.full(len(potential), 0.5),
    )
    physics = parse_agpd_physics(name)
    config = dict(fixed_point_iterations=384, adaptive_convergence=True, fixed_point_damping=0.5,
                  relative_residual_tolerance=1e-7, branch_tolerance=1e-5, bep_coefficient=0.65)
    config.update(settings)
    return evaluate_agpd_lateral_interactions(
        state, parameters, 293.15, pathways=physics.pathways, finite_steps=physics.finite_steps,
        interactions=physics.interactions, lateral_config=config,
    )


def _values(result):
    keys = ('coverage_solver_relative_residual', 'coverage_branch_disagreement', 'coverage_solver_iterations')
    evaluated = pytensor.function([], [result.ln_rate, *(result.pointwise[name] for name in keys)])()
    return tuple(np.asarray(value, dtype=float) for value in evaluated)


def test_adaptive_recovers_difficult_full_ssa_prior_draws():
    name = 'cxo_CO_BF_ER_LH_3_pOH_sOH'
    cases = [
        (DRAW_2, [0.08, 0.12], [0.1, 1.0], [0.25, 1.0]),
        (DRAW_30, [-0.16], [0.01], [0.25]),
    ]
    for params, E, co, oh in cases:
        old = _values(_evaluate(name, params, E, co, oh, fixed_point_iterations=96, adaptive_convergence=False))
        updated = _values(_evaluate(name, params, E, co, oh))
        reference = _values(_evaluate(name, params, E, co, oh, adaptive_convergence=False))
        assert np.any(~np.isfinite(old[0]))
        assert np.all(np.isfinite(updated[0]))
        assert np.all(updated[1] < 1e-7)
        assert np.all(updated[2] < 1e-5)
        assert np.all((updated[3] > 0) & (updated[3] < 384))
        np.testing.assert_allclose(updated[0], reference[0], rtol=1e-5, atol=1e-5)


def test_adaptive_stops_early_for_simple_interaction_model():
    params = dict(DRAW_2, z_CO_CO_eV=0.015)
    result = _evaluate('c_ER', params, [0.08], [0.1], [0.25])
    rate, residual, diff, iterations = _values(result)
    assert np.all(np.isfinite(rate))
    assert np.all(residual < 1e-7)
    assert np.all(diff < 1e-5)
    assert np.all(iterations < 96)


def test_adaptive_preserves_multistable_rejection():
    values = dict(DRAW_2, z_CO_CO_eV=-0.073116)
    result = _evaluate('cxo_CO_BF_ER_LH_3_pOH_sOH', values, [0.08], [0.1], [0.25])
    rate, residual, diff, iterations = _values(result)
    assert np.isnan(rate[0])
    assert residual[0] < 1e-7
    assert diff[0] > 0.1
    assert iterations[0] <= 384


def test_adaptive_c_er_symbolic_gradient():
    name = 'c_ER'
    z = pt.dscalar('z_CO_CO_eV')
    definition = get_agpd_model_definition(name)
    values = {field.name: DRAW_2[field.name] for field in fields(definition.parameter_class)}
    values['z_CO_CO_eV'] = z
    parameters = definition.parameter_class(**values)
    state = SimpleNamespace(
        E_V_SHE=np.array([0.08]), ln_a_CO=np.log(np.array([0.1])), ln_a_OH=np.log(np.array([0.25])),
        Pd_fraction=np.array([0.5]), Ag_fraction=np.array([0.5]),
    )
    result = evaluate_agpd_lateral_interactions(
        state, parameters, 293.15, pathways=('ER',), finite_steps=(), interactions=('c',),
        lateral_config=dict(fixed_point_iterations=384, adaptive_convergence=True, fixed_point_damping=0.5,
                            relative_residual_tolerance=1e-7, branch_tolerance=1e-5, bep_coefficient=0.65),
    )
    lograte = pt.sum(result.ln_rate)
    function = pytensor.function([z], [lograte, pt.grad(lograte, z)])
    value, gradient = function(0.015)
    h = 1e-5
    finite_difference = (function(0.015+h)[0]-function(0.015-h)[0])/(2*h)
    assert np.isfinite(value) and np.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-3, atol=1e-3)


def test_adaptive_scan_keeps_prior_random_variables_outside_loop():
    from pytensor.graph import ancestors
    from pytensor.scan.op import Scan
    from pytensor.tensor.random.op import RandomVariable

    name = 'c_ER'
    definition = get_agpd_model_definition(name)
    values = {field.name: DRAW_2[field.name] for field in fields(definition.parameter_class)}
    values['deltaG1_0'] = pt.random.normal(-0.35, 0.02)
    values['z_CO_CO_eV'] = pt.random.normal(0.03, 0.01)
    parameters = definition.parameter_class(**values)
    state = SimpleNamespace(
        E_V_SHE=np.array([0.08]), ln_a_CO=np.log(np.array([0.1])), ln_a_OH=np.log(np.array([0.25])),
        Pd_fraction=np.array([0.5]), Ag_fraction=np.array([0.5]),
    )
    result = evaluate_agpd_lateral_interactions(
        state, parameters, 293.15, pathways=('ER',), finite_steps=(), interactions=('c',),
        lateral_config=dict(fixed_point_iterations=3, adaptive_convergence=True, fixed_point_damping=0.5,
                            relative_residual_tolerance=1e-7, branch_tolerance=1e-5, bep_coefficient=0.65),
    )
    scans = [node.owner.op for node in ancestors([result.ln_rate])
             if node.owner and isinstance(node.owner.op, Scan)]
    assert len(scans) == 1
    assert not any(node.owner and isinstance(node.owner.op, RandomVariable)
                   for node in ancestors(scans[0].inner_outputs))
