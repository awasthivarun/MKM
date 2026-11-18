"""Finite pOH/sOH adsorption with instant COOH removal: conservation and QEA limits."""

from dataclasses import fields

import numpy as np
import pytensor.tensor as pt
import pytest

from mkm.mechanisms.agpd_basic import build_agpd_point_state
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import get_agpd_model_definition, get_agpd_prior_profile
from mkm.postprocessing.drc import transition_state_controls
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config


def _state():
    model_inputs = ModelPointInputs(
        materials=("Ag50Pd50",), material_index=np.zeros(3, dtype=np.int64),
        E_V_SHE=np.array([-0.05, 0.0, 0.05]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.1, 0.5])),
    )
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    return build_agpd_point_state(model_inputs, config), config


def _parameters(definition, fast_oh=False):
    defaults = dict(
        deltaG1_0=-0.2, deltaG4_0=0.02, deltaG5_0=0.02, q=0.50,
        Gact1_0=0.46, Gact4_0=0.35 if fast_oh else 0.72,
        Gact5_0=0.35 if fast_oh else 0.72, Gact2_BF_0=0.70,
        Gact2_ER_0=0.73, Gact2_LH_0=0.72, beta_4=0.50, beta_5=0.50,
        beta_2_BF=0.5, beta_2_ER=0.5,
    )
    return definition.parameter_class(**{f.name: defaults[f.name] for f in fields(definition.parameter_class)})


def _evaluate(name, fast_oh=False):
    state, config = _state()
    definition = get_agpd_model_definition(name)
    evaluator = getattr(definition.evaluator, "func", definition.evaluator)
    kwargs = {"mixed_config": config["mixed_adsorption"]} if evaluator.__name__ == (
        "evaluate_agpd_mixed_adsorption"
    ) else {}
    result = definition.evaluator(state, _parameters(definition, fast_oh), 293.15, **kwargs)
    return result.mechanism_result


@pytest.mark.parametrize("name", (
    "CO_BF_ER_LH_pOH", "CO_BF_ER_LH_sOH", "CO_BF_ER_LH_pOH_sOH", "BF_ER_LH_pOH_sOH",
    "CO_ER_pOH", "ER_pOH", "BF_sOH", "BF_ER_pOH_sOH", "ER_sOH",
))
def test_mixed_oh_models_close_site_balances_and_solver(name):
    result = _evaluate(name)
    values = {key: np.asarray(tensor.eval(), dtype=float) for key, tensor in result.pointwise.items()}
    for key in ("theta_CO", "theta_OH_Pd", "theta_OH_Ag", "theta_empty_Pd", "theta_empty_Ag"):
        assert np.all(np.isfinite(values[key])), key
        assert np.all((0 <= values[key]) & (values[key] <= 1)), key
    np.testing.assert_allclose(values["theta_CO"] + values["theta_OH_Pd"] + values["theta_empty_Pd"], 1)
    np.testing.assert_allclose(values["theta_OH_Ag"] + values["theta_empty_Ag"], 1)
    assert np.all(values["coverage_solver_relative_residual"] < 1e-7)
    assert np.all(np.isfinite(result.ln_rate.eval()))
    fractions = sum(values[f"rate_fraction_{pathway}"] for pathway in get_agpd_model_definition(name).pathways)
    np.testing.assert_allclose(fractions, 1.0, atol=1e-9)


@pytest.mark.parametrize("co_finite", (True, False))
def test_fast_oh_kinetics_reproduce_qea_limit(co_finite):
    reference_name = "CO_BF_ER_LH" if co_finite else "BF_ER_LH"
    extended_name = reference_name + "_pOH_sOH"
    reference = _evaluate(reference_name)
    extended = _evaluate(extended_name, fast_oh=True)
    np.testing.assert_allclose(extended.ln_rate.eval(), reference.ln_rate.eval(), rtol=2e-3, atol=2e-3)
    for key in ("theta_CO", "theta_OH_Pd", "theta_OH_Ag"):
        np.testing.assert_allclose(
            extended.pointwise[key].eval(), reference.pointwise[key].eval(), rtol=2e-3, atol=2e-3
        )


def test_finite_primary_oh_balance_includes_lh_consumption_at_fixed_co_coverage():
    from mkm.mechanisms.agpd_basic.mixed_adsorption import _adsorption_state

    common = dict(a4=1.0, b4=1.0, a5=1.0, b5=1.0, bf=0.0, er=0.0,
                  k1a=1.0, km1=1.0, K1a=1.0, co_finite=True)
    finite = _adsorption_state(pt.as_tensor_variable(0.2), lh=2.0, **common)
    qea = _adsorption_state(pt.as_tensor_variable(0.2), lh=0.0, **common)
    assert float(finite["theta_pOH"].eval()) < float(qea["theta_pOH"].eval())


def test_redundant_finite_primary_oh_does_not_change_bf_er_prediction():
    qea = _evaluate("BF_ER")
    finite = _evaluate("BF_ER_pOH")
    np.testing.assert_allclose(qea.ln_rate.eval(), finite.ln_rate.eval(), rtol=1e-8, atol=1e-8)
    assert "Gact4_0" in {field.name for field in fields(get_agpd_model_definition("BF_ER_pOH").parameter_class)}


def test_finite_oh_parameter_priors_and_drc_controls_use_same_model_name():
    _, config = _state()
    model_name = "CO_BF_ER_LH_pOH_sOH"
    profile = get_agpd_prior_profile(config, "Ag50Pd50", model_name)
    assert "Gact4_0" in profile["parameters"] and "Gact5_0" in profile["parameters"]
    controls = {control.name for control in transition_state_controls(model_name)}
    assert controls == {"CO_adsorption", "BF", "ER", "LH", "pOH", "sOH"}


@pytest.mark.parametrize(("model_name", "parameter_name"), (
    ("CO_ER_LH_pOH", "Gact4_0"), ("CO_BF_ER_sOH", "Gact5_0"),
))
def test_mixed_adsorption_rate_has_finite_symbolic_gradient(model_name, parameter_name):
    state, config = _state()
    definition = get_agpd_model_definition(model_name)
    params = vars(_parameters(definition))
    variable = pt.dscalar(parameter_name)
    params[parameter_name] = variable
    evaluation = definition.evaluator(
        state, definition.parameter_class(**params), 293.15, mixed_config=config["mixed_adsorption"]
    )
    gradient = pt.grad(pt.sum(evaluation.mechanism_result.ln_rate), variable)
    value = float(gradient.eval({variable: 0.50}))
    assert np.isfinite(value)
