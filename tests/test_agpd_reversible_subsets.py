"""Finite-step-3 pathway subsets: physical balances, registry and differentiability."""

from dataclasses import fields

import numpy as np
import pytensor.tensor as pt
import pytest

from mkm.mechanisms.agpd_basic import build_agpd_point_state
from mkm.mechanisms.agpd_basic.reversible_subsets import _coverages_from_affinity
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import get_agpd_model_definition, get_agpd_prior_profile
from mkm.postprocessing.drc import transition_state_controls
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config


def _state(material="Ag50Pd50"):
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    inputs = ModelPointInputs(
        materials=(material,), material_index=np.zeros(3, dtype=np.int64),
        E_V_SHE=np.array([-0.05, 0.0, 0.05]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.1, 0.5])),
    )
    return build_agpd_point_state(inputs, config), config


def _parameters(model_name, *, fast_steps=()):
    defaults = dict(
        deltaG1_0=-0.20, deltaG4_0=0.02, deltaG5_0=0.02, deltaG3_0=-0.08, q=0.5,
        Gact1_0=0.10 if "CO" in fast_steps else 0.46,
        Gact4_0=0.10 if "pOH" in fast_steps else 0.65,
        Gact5_0=0.10 if "sOH" in fast_steps else 0.65,
        Gact2_BF_0=0.70, Gact2_ER_0=0.72, Gact2_LH_0=0.70,
        Gact3_0=0.08 if "3" in fast_steps else 0.68,
        beta_4=0.5, beta_5=0.5, beta_2_BF=0.5, beta_2_ER=0.5, beta_3=0.5,
    )
    definition = get_agpd_model_definition(model_name)
    return definition.parameter_class(**{field.name: defaults[field.name] for field in fields(definition.parameter_class)})


def _evaluate(model_name, *, material="Ag50Pd50", fast_steps=()):
    state, config = _state(material)
    definition = get_agpd_model_definition(model_name)
    result = definition.evaluator(
        state, _parameters(model_name, fast_steps=fast_steps), 293.15,
        reversible_config=config["reversible_ssa"],
    )
    return result


@pytest.mark.parametrize("model_name", (
    "ER_3", "BF_3", "BF_ER_3", "CO_ER_3", "ER_LH_3_pOH",
    "BF_3_sOH", "CO_BF_ER_LH_3", "CO_BF_ER_LH_3_pOH",
    "CO_BF_ER_LH_3_sOH", "CO_BF_ER_LH_3_pOH_sOH", "BF_ER_LH_3_pOH_sOH",
    "CO_BF_3_pOH_sOH", "ER_3_pOH_sOH",
))
def test_finite_step3_variants_have_physical_coverages_and_flux_balance(model_name):
    evaluated = _evaluate(model_name)
    pointwise = evaluated.pointwise
    values = {key: np.asarray(value.eval(), dtype=float) for key, value in pointwise.items()}
    for name in ("theta_CO", "theta_COOH", "theta_OH_Pd", "theta_empty_Pd", "theta_OH_Ag", "theta_empty_Ag"):
        assert np.all(np.isfinite(values[name])), name
        assert np.all((values[name] >= 0.0) & (values[name] <= 1.0)), name
    np.testing.assert_allclose(
        sum(values[name] for name in ("theta_CO", "theta_COOH", "theta_OH_Pd", "theta_empty_Pd")),
        1.0, atol=1e-9,
    )
    np.testing.assert_allclose(values["theta_OH_Ag"] + values["theta_empty_Ag"], 1.0, atol=1e-10)
    assert np.all(values["ssa_relative_residual_COOH"] < 1e-8)
    log_rate = np.asarray(evaluated.ln_rate.eval(), dtype=float)
    assert np.all(np.isfinite(log_rate))
    np.testing.assert_allclose(values["electron_flux"], 2.0 * np.exp(log_rate),
                               rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        sum(values[f"net_pathway_fraction_{path}"] for path in get_agpd_model_definition(model_name).pathways),
        1.0, rtol=1e-8, atol=1e-8,
    )


def test_reversible_reduction_closes_adsorption_balances_and_detailed_balance():
    # Synthetic site-population and rate constants with K_ER*aOH = K4*aOH*K_LH = K5*aOH*K_BF.
    fp, fs, kbf, kbf_reverse, klh, klh_reverse = 0.4, 0.6, 1.3, 5.2, 2.0, 2.4
    values = dict(
        K1a=3.0, K4a=0.6, K5a=2.0, k3a=0.8, k1_reverse=1.7,
        co_finite=True, pOH_finite=True, sOH_finite=True,
        k4_reverse=5.0, k5_forward_aOH=12.0, k5_reverse=6.0,
        kBF=kbf, kBF_reverse=kbf_reverse, kER_reverse=2.0,
        kLH=klh, kLH_reverse=klh_reverse, K_ER_aOH=0.5, fp=fp, fs=fs,
    )
    affinity = 0.3
    solved = _coverages_from_affinity(pt.as_tensor_variable(affinity), **values)
    coverage = {key: float(value.eval()) for key, value in solved.items()}
    co, cooh = coverage["theta_CO"], coverage["theta_COOH"]
    h, e = coverage["theta_OH_Pd"], coverage["theta_empty_Pd"]
    s, g = coverage["theta_OH_Ag"], coverage["theta_empty_Ag"]
    departure = np.expm1(affinity)
    rbf = fs * (kbf * co * s - kbf_reverse * cooh * g)
    rer = 1.0 * co - 2.0 * cooh
    rlh = fp * (klh * co * h - klh_reverse * cooh * e)
    for pathway, flux in (("BF", rbf), ("ER", rer), ("LH", rlh)):
        np.testing.assert_allclose(flux, coverage[f"coefficient_{pathway}"] * cooh * departure, rtol=1e-10)
    np.testing.assert_allclose(3.0 * 1.7 * e - 1.7 * co, 0.8 * cooh, rtol=1e-10)
    np.testing.assert_allclose(3.0 * e - 5.0 * h, rlh, rtol=1e-10)
    np.testing.assert_allclose(12.0 * g - 6.0 * s, fp / fs * rbf, rtol=1e-10)
    np.testing.assert_allclose(co + cooh + h + e, 1.0, rtol=1e-10)
    np.testing.assert_allclose(s + g, 1.0, rtol=1e-10)


@pytest.mark.parametrize("finite_step", ("CO", "pOH", "sOH"))
def test_very_fast_adsorption_approaches_qea_limit(finite_step):
    reference = _evaluate("BF_ER_LH_3")
    trial = _evaluate(
        f"{'CO_' if finite_step == 'CO' else ''}BF_ER_LH_3"
        f"{'_pOH' if finite_step == 'pOH' else ''}"
        f"{'_sOH' if finite_step == 'sOH' else ''}",
        fast_steps=(finite_step,),
    )
    np.testing.assert_allclose(trial.ln_rate.eval(), reference.ln_rate.eval(), rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("model_name", ("ER", "CO_ER"))
def test_fast_cooh_oxidation_approaches_instant_step3(model_name):
    state, config = _state()
    reference_definition = get_agpd_model_definition(model_name)
    reference = reference_definition.evaluator(
        state, _parameters(model_name), 293.15
    ).mechanism_result.ln_rate.eval()
    finite = _evaluate(model_name + "_3", fast_steps=("3",)).ln_rate.eval()
    np.testing.assert_allclose(finite, reference, rtol=1e-4, atol=1e-4)


def test_redundant_primary_oh_step_is_admitted_but_has_no_effect_without_lh():
    primary_qea = _evaluate("BF_ER_3")
    primary_finite = _evaluate("BF_ER_3_pOH")
    np.testing.assert_allclose(primary_qea.ln_rate.eval(), primary_finite.ln_rate.eval(), atol=1e-9)


def test_bf_only_on_pure_pd_has_zero_rate_not_nan():
    pointwise = _evaluate("BF_3_sOH", material="Pd100")
    rate = np.asarray(pointwise.ln_rate.eval())
    assert np.all(np.isneginf(rate))
    assert np.all(np.asarray(pointwise.pointwise["electron_flux"].eval()) == 0.0)
    assert np.all(np.asarray(pointwise.pointwise["ssa_relative_residual_COOH"].eval()) == 0.0)


def test_finite_step3_priors_drc_and_canonical_name():
    _, config = _state()
    model_name = "CO_BF_ER_LH_3_pOH_sOH"
    definition = get_agpd_model_definition(model_name)
    profile = get_agpd_prior_profile(config, "Ag50Pd50", model_name)
    assert {"deltaG3_0", "Gact3_0", "beta_3", "Gact4_0", "Gact5_0"} <= set(profile["parameters"])
    assert definition.supports_individual and definition.supports_all_materials
    controls = {control.name for control in transition_state_controls(model_name)}
    assert controls == {"CO_adsorption", "BF", "ER", "LH", "CO2", "pOH", "sOH"}


@pytest.mark.parametrize(("model_name", "parameter_name"), (
    ("CO_BF_ER_LH_3_pOH_sOH", "Gact3_0"),
    ("CO_ER_LH_3_pOH", "Gact4_0"),
    ("BF_3_sOH", "Gact5_0"),
))
def test_reversible_rate_has_finite_symbolic_gradient(model_name, parameter_name):
    state, config = _state()
    definition = get_agpd_model_definition(model_name)
    params = vars(_parameters(model_name))
    variable = pt.dscalar(parameter_name)
    params[parameter_name] = variable
    result = definition.evaluator(
        state, definition.parameter_class(**params), 293.15,
        reversible_config=config["reversible_ssa"],
    )
    gradient = pt.grad(pt.sum(result.ln_rate), variable)
    assert np.isfinite(float(gradient.eval({variable: 0.65})))
