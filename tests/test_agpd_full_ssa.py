import numpy as np

from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLH3OHParameters,
    build_agpd_point_state,
    evaluate_agpd_co_bf_er_lh_3_oh,
)
from mkm.model_inputs import ModelPointInputs


def _inputs():
    return ModelPointInputs(
        materials=("Ag50Pd50",),
        material_index=np.array([0, 0, 0], dtype=np.int64),
        E_V_SHE=np.array([-0.10, 0.00, 0.10]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.10, 1.0])),
    )


def _config():
    return {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {"Ag50Pd50": {"Ag_fraction": 0.5, "Pd_fraction": 0.5}},
        "full_ssa": {
            "newton_iterations": 20,
            "newton_damping": 0.8,
            "newton_fallback_fraction": 0.1,
            "initial_cooh_fraction_of_max": 0.05,
            "coverage_epsilon": 1e-12,
            "overall_reaction_deltaG_0_eV": -1.86,
        },
    }


def _parameters():
    return AgPdCOBFERRLH3OHParameters(
        deltaG1_0=-0.12,
        deltaG4_0=0.10,
        deltaG5_0=0.00,
        deltaG3_0=0.55,
        q=0.45,
        Gact1_0=0.52,
        Gact4_0=0.50,
        Gact5_0=0.48,
        Gact2_BF_0=0.68,
        Gact2_ER_0=0.74,
        Gact2_LH_0=0.76,
        Gact3_0=0.64,
        beta_4=0.50,
        beta_5=0.50,
        beta_2_BF=0.50,
        beta_2_ER=0.50,
        beta_3=0.50,
    )


def _result():
    config = _config()
    state = build_agpd_point_state(_inputs(), config)
    return state, evaluate_agpd_co_bf_er_lh_3_oh(
        state=state,
        parameters=_parameters(),
        temperature_K=293.15,
        full_ssa_config=config["full_ssa"],
    )


def test_full_ssa_coverages_are_physical_and_close_site_balances():
    _, result = _result()
    pointwise = result.mechanism_result.pointwise
    names = ("theta_CO", "theta_COOH", "theta_OH_Pd", "theta_empty_Pd", "theta_OH_Ag", "theta_empty_Ag")
    values = {name: np.asarray(pointwise[name].eval(), dtype=float) for name in names}
    for value in values.values():
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0.0)
        assert np.all(value <= 1.0)
    np.testing.assert_allclose(
        values["theta_CO"] + values["theta_COOH"] + values["theta_OH_Pd"] + values["theta_empty_Pd"],
        1.0,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(values["theta_OH_Ag"] + values["theta_empty_Ag"], 1.0, rtol=1e-10, atol=1e-10)


def test_full_ssa_thermodynamic_cycles_close_exactly():
    state, result = _result()
    deltaG_ER = np.asarray(result.deltaG_ER.eval(), dtype=float)
    deltaG_LH = np.asarray(result.deltaG_LH.eval(), dtype=float)
    deltaG_BF = np.asarray(result.deltaG_BF.eval(), dtype=float)
    deltaG_OH_Pd = np.asarray(result.deltaG_OH_Pd.eval(), dtype=float)
    deltaG_OH_Ag = np.asarray(result.deltaG_OH_Ag.eval(), dtype=float)
    deltaG_COOH = np.asarray(result.deltaG_COOH.eval(), dtype=float)
    deltaG_3 = np.asarray(result.deltaG_3.eval(), dtype=float)
    np.testing.assert_allclose(deltaG_ER, deltaG_OH_Pd + deltaG_LH, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(deltaG_ER, deltaG_OH_Ag + deltaG_BF, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(deltaG_COOH + deltaG_3, -1.86 - 2.0 * state.E_V_SHE, rtol=1e-12, atol=1e-12)


def test_full_ssa_balances_and_two_electron_stoichiometry():
    _, result = _result()
    rate = np.exp(np.asarray(result.mechanism_result.ln_rate.eval(), dtype=float))
    residuals = (
        result.ssa_residual_CO,
        result.ssa_residual_OH_Pd,
        result.ssa_residual_OH_Ag,
        result.ssa_residual_COOH,
    )
    for residual in residuals:
        relative = np.asarray(residual.eval(), dtype=float) / np.maximum(rate, 1e-30)
        np.testing.assert_allclose(relative, 0.0, atol=1e-6)
    solver_residual = np.asarray(
        result.mechanism_result.pointwise["ssa_relative_residual_COOH"].eval(),
        dtype=float,
    )
    np.testing.assert_allclose(solver_residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result.electron_flux.eval(), dtype=float),
        2.0 * rate,
        rtol=1e-6,
        atol=1e-10,
    )


def test_full_ssa_net_pathway_fractions_sum_to_one_but_are_not_forced_positive():
    _, result = _result()
    pointwise = result.mechanism_result.pointwise
    total = sum(
        np.asarray(pointwise[name].eval(), dtype=float)
        for name in ("net_pathway_fraction_BF", "net_pathway_fraction_ER", "net_pathway_fraction_LH")
    )
    np.testing.assert_allclose(total, 1.0, rtol=1e-6, atol=1e-6)
