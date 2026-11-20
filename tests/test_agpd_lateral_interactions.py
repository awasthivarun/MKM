"""Frumkin interactions: reciprocity, physical balances and zero-interaction limits."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from mkm.mechanisms.agpd_basic.lateral_interactions import (
    _excess_chemical_potentials, evaluate_agpd_lateral_interactions,
)
from mkm.models.agpd_registry import get_agpd_model_definition
from mkm.models.agpd_physics import parse_agpd_physics


DEFAULTS = dict(
    deltaG1_0=-0.20, deltaG4_0=0.02, deltaG5_0=0.02, deltaG3_0=-0.08, q=0.5,
    Gact1_0=0.46, Gact4_0=0.65, Gact5_0=0.65, Gact2_BF_0=0.70,
    Gact2_ER_0=0.72, Gact2_LH_0=0.70, Gact3_0=0.68,
    beta_4=0.5, beta_5=0.5, beta_2_BF=0.5, beta_2_ER=0.5, beta_3=0.5,
    z_CO_CO_eV=0.0, z_CO_OH_eV=0.0, z_OH_OH_eV=0.0,
)
SOLVER = dict(fixed_point_iterations=64, fixed_point_damping=0.5,
              relative_residual_tolerance=1e-7, branch_tolerance=1e-5, bep_coefficient=0.65)


def _state(material="Ag50Pd50"):
    fraction = {"Ag50Pd50": 0.5, "Pd100": 0.0, "Ag90Pd10": 0.9}[material]
    return SimpleNamespace(
        E_V_SHE=np.array([-0.05, 0.0, 0.05]),
        ln_a_CO=np.log(np.array([0.01, 0.1, 0.5])),
        ln_a_OH=np.log(np.array([0.5, 0.5, 0.5])),
        Pd_fraction=np.repeat(1.0-fraction, 3), Ag_fraction=np.repeat(fraction, 3),
    )


def _parameters(model_name, **overrides):
    definition = get_agpd_model_definition(model_name)
    values = DEFAULTS | overrides
    selected = {field.name: values[field.name] for field in fields(definition.parameter_class)}
    return definition.parameter_class(**selected)


def _evaluate(model_name, material="Ag50Pd50", **overrides):
    physics = parse_agpd_physics(model_name)
    result = evaluate_agpd_lateral_interactions(
        _state(material), _parameters(model_name, **overrides), 293.15,
        pathways=physics.pathways, finite_steps=physics.finite_steps,
        interactions=physics.interactions, lateral_config=SOLVER,
    )
    return result


def test_reciprocal_excess_chemical_potentials():
    fp, fs, q = 0.4, 0.6, 0.3
    c, p, s = pt.dscalars("c", "p", "s")
    mu_c, mu_p, mu_s = _excess_chemical_potentials(c, p, s, fp, fs, q, 0.12, -0.08, 0.10)
    # Mixed derivatives of one scalar excess energy must be reciprocal after site weighting.
    np.testing.assert_allclose(float(pt.grad(mu_c, p).eval({c: 0.4, p: 0.3, s: 0.2})) / fp,
                               float(pt.grad(mu_p, c).eval({c: 0.4, p: 0.3, s: 0.2})) / fp)
    np.testing.assert_allclose(float(pt.grad(mu_c, s).eval({c: 0.4, p: 0.3, s: 0.2})) / fs,
                               float(pt.grad(mu_s, c).eval({c: 0.4, p: 0.3, s: 0.2})) / fp)


@pytest.mark.parametrize("name", (
    "c_ER", "x_ER_LH", "o_BF_ER", "cxo_BF_ER_LH", "cxo_CO_ER_LH_pOH",
    "cxo_CO_BF_ER_LH_3_pOH_sOH", "x_BF_3_sOH", "o_ER_3_pOH",
))
def test_interacting_models_have_physical_coverages_and_balances(name):
    result = _evaluate(name, z_CO_CO_eV=0.01, z_CO_OH_eV=0.01, z_OH_OH_eV=0.01)
    fields = tuple(result.pointwise)
    evaluated = pytensor.function([], [result.ln_rate, *(result.pointwise[key] for key in fields)])()
    log_rate = np.asarray(evaluated[0], dtype=float)
    values = {key: np.asarray(value, dtype=float) for key, value in zip(fields, evaluated[1:])}
    for key in ("theta_CO", "theta_COOH", "theta_OH_Pd", "theta_OH_Ag",
                "theta_empty_Pd", "theta_empty_Ag"):
        assert np.all((values[key] >= 0.0) & (values[key] <= 1.0)), (name, key)
    np.testing.assert_allclose(
        values["theta_CO"]+values["theta_COOH"]+values["theta_OH_Pd"]+values["theta_empty_Pd"],
        1.0, atol=1e-10,
    )
    np.testing.assert_allclose(values["theta_OH_Ag"]+values["theta_empty_Ag"], 1.0, atol=1e-10)
    assert np.all(values["coverage_solver_relative_residual"] < SOLVER["relative_residual_tolerance"])
    assert np.all(values["coverage_branch_disagreement"] < SOLVER["branch_tolerance"])
    assert np.all(np.isfinite(log_rate))
    np.testing.assert_allclose(values["electron_flux"], 2.0*np.exp(log_rate), rtol=1e-7)


@pytest.mark.parametrize("base", ("ER", "BF_ER_LH", "CO_ER", "CO_BF_ER_LH_3_pOH_sOH"))
def test_zero_interaction_limit_matches_noninteracting_evaluator(base):
    name = "cxo_"+base
    lateral = _evaluate(name).ln_rate.eval()
    definition = get_agpd_model_definition(base)
    state = _state()
    params = _parameters(base)
    kwargs = {}
    if "3" in base:
        kwargs["reversible_config"] = dict(affinity_iterations=48, affinity_damping=1.0,
                                            relative_residual_tolerance=1e-8)
    reference = definition.evaluator(state, params, 293.15, **kwargs)
    reference = getattr(reference, "mechanism_result", reference)
    np.testing.assert_allclose(lateral, reference.ln_rate.eval(), rtol=2e-5, atol=2e-5)


def test_bf_only_on_pure_pd_is_exactly_inactive():
    evaluated = _evaluate("cxo_BF_3_sOH", material="Pd100")
    assert np.all(np.isneginf(evaluated.ln_rate.eval()))
    assert np.all(evaluated.pointwise["electron_flux"].eval() == 0.0)


def test_canonical_interaction_names_and_parameter_membership():
    for name, expected in (("c_ER", {"z_CO_CO_eV"}), ("xo_ER", {"z_CO_OH_eV", "z_OH_OH_eV"}),
                           ("cxo_CO_BF_ER_LH_3_pOH_sOH", {"z_CO_CO_eV", "z_CO_OH_eV", "z_OH_OH_eV"})):
        definition = get_agpd_model_definition(name)
        present = {field.name for field in fields(definition.parameter_class)}
        assert present & {"z_CO_CO_eV", "z_CO_OH_eV", "z_OH_OH_eV"} == expected
        assert parse_agpd_physics(name).name == name


@pytest.mark.slow
def test_symbolic_gradient_with_respect_to_cross_interaction():
    name = "x_CO_BF_ER_LH_3_pOH_sOH"
    params = vars(_parameters(name))
    z = pt.dscalar("z_CO_OH_eV")
    params["z_CO_OH_eV"] = z
    physics = parse_agpd_physics(name)
    result = evaluate_agpd_lateral_interactions(
        _state(), get_agpd_model_definition(name).parameter_class(**params), 293.15,
        pathways=physics.pathways, finite_steps=physics.finite_steps,
        interactions=physics.interactions, lateral_config=SOLVER,
    )
    summed_rate = pt.sum(result.ln_rate)
    gradient = pt.grad(summed_rate, z)
    function = pytensor.function([z], [summed_rate, gradient])
    value, derivative = function(0.01)
    scale = 1e-5
    difference = (function(0.01+scale)[0] - function(0.01-scale)[0]) / (2.0*scale)
    assert np.isfinite(value) and np.isfinite(derivative)
    np.testing.assert_allclose(derivative, difference, rtol=1e-3, atol=1e-3)


def test_unconverged_coverage_solution_is_not_silently_accepted():
    name = "c_ER"
    physics = parse_agpd_physics(name)
    result = evaluate_agpd_lateral_interactions(
        _state(), _parameters(name, z_CO_CO_eV=0.06), 293.15,
        pathways=physics.pathways, finite_steps=physics.finite_steps,
        interactions=physics.interactions, lateral_config=dict(SOLVER, fixed_point_iterations=1),
    )
    assert np.all(np.isnan(result.ln_rate.eval()))


def test_configured_bep_coefficient_changes_rates_only_when_interactions_are_nonzero():
    name = "c_ER"
    physics = parse_agpd_physics(name)

    def rate(z, gamma):
        result = evaluate_agpd_lateral_interactions(
            _state(), _parameters(name, z_CO_CO_eV=z), 293.15,
            pathways=physics.pathways, finite_steps=physics.finite_steps,
            interactions=physics.interactions, lateral_config=dict(SOLVER, bep_coefficient=gamma),
        )
        return np.asarray(result.ln_rate.eval(), dtype=float)

    np.testing.assert_allclose(rate(0.0, 0.0), rate(0.0, 0.65), atol=1e-10)
    assert np.any(np.abs(rate(0.06, 0.0) - rate(0.06, 0.65)) > 1e-3)


def test_scan_keeps_prior_random_variables_outside_solver_loop():
    from pytensor.graph import ancestors
    from pytensor.scan.op import Scan
    from pytensor.tensor.random.op import RandomVariable

    name = "c_ER"
    values = vars(_parameters(name))
    values["deltaG1_0"] = pt.random.normal(-0.20, 0.01)
    values["z_CO_CO_eV"] = pt.random.normal(0.01, 0.005)
    result = evaluate_agpd_lateral_interactions(
        _state(), get_agpd_model_definition(name).parameter_class(**values), 293.15,
        pathways=("ER",), finite_steps=(), interactions=("c",),
        lateral_config={**SOLVER, "fixed_point_iterations": 3},
    )
    scans = [node.owner.op for node in ancestors([result.ln_rate])
             if node.owner and isinstance(node.owner.op, Scan)]
    assert len(scans) == 1
    assert not any(node.owner and isinstance(node.owner.op, RandomVariable)
                   for node in ancestors(scans[0].inner_outputs))


def test_pymc_prior_predictive_can_compile_frumkin_scan():
    import pymc as pm

    name = "c_ER"
    with pm.Model():
        values = vars(_parameters(name))
        values["deltaG1_0"] = pm.Normal("deltaG1_0", mu=-0.20, sigma=0.01)
        values["z_CO_CO_eV"] = pm.Normal("z_CO_CO_eV", mu=0.01, sigma=0.005)
        result = evaluate_agpd_lateral_interactions(
            _state(), get_agpd_model_definition(name).parameter_class(**values), 293.15,
            pathways=("ER",), finite_steps=(), interactions=("c",),
            lateral_config={**SOLVER, "fixed_point_iterations": 3},
        )
        pm.Deterministic("ln_rate_model", result.ln_rate)
        draws = pm.sample_prior_predictive(
            draws=2, random_seed=42, var_names=["ln_rate_model"], return_inferencedata=False,
        )
    assert draws["ln_rate_model"].shape == (2, 3)
