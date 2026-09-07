import numpy as np
import pytest

from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLHParameters,
    build_agpd_point_state,
    evaluate_agpd_co_bf_er_lh,
    evaluate_agpd_co_bf_er_lh_ag10_no_bf,
    evaluate_agpd_co_bf_er_lh_capped,
    evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf,
)
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import (
    available_agpd_all_material_models,
    available_agpd_models,
    build_agpd_mechanism,
    get_agpd_model_definition,
    get_agpd_parameterization,
    get_agpd_prior_profile,
)


MODEL_NAMES = (
    "CO_BF_ER_LH_Ag10_no_BF",
    "CO_BF_ER_LH_capped_Ag10_no_BF",
)


def _inputs():
    return ModelPointInputs(
        materials=("Ag10Pd90", "Ag25Pd75"),
        material_index=np.array([0, 0, 1, 1], dtype=np.int64),
        E_V_SHE=np.array([0.00, 0.10, 0.00, 0.10]),
        ln_electrolyte_concentration=np.log(np.full(4, 0.50)),
        ln_CO_mole_fraction=np.log(np.full(4, 0.01)),
    )


def _config(theta_CO_max=0.66):
    return {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {
            "Ag10Pd90": {"Ag_fraction": 0.10, "Pd_fraction": 0.90},
            "Ag25Pd75": {"Ag_fraction": 0.25, "Pd_fraction": 0.75},
        },
        "co_coverage_cap": {"Ag10Pd90": theta_CO_max, "Ag25Pd75": theta_CO_max},
    }


def _parameters(deltaG5_0=0.0):
    return AgPdCOBFERRLHParameters(
        deltaG1_0=-0.40,
        deltaG4_0=0.00,
        deltaG5_0=deltaG5_0,
        beta_2_BF=0.50,
        beta_2_ER=0.40,
        q=0.40,
        Gact1_0=0.70,
        Gact2_BF_0=0.70,
        Gact2_ER_0=0.72,
        Gact2_LH_0=0.75,
    )


@pytest.mark.parametrize(
    ("ordinary_evaluator", "no_bf_evaluator"),
    [
        (evaluate_agpd_co_bf_er_lh, evaluate_agpd_co_bf_er_lh_ag10_no_bf),
        (evaluate_agpd_co_bf_er_lh_capped, evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf),
    ],
)
def test_ag10_no_bf_disables_bf_only_for_ag10(ordinary_evaluator, no_bf_evaluator):
    state = build_agpd_point_state(inputs=_inputs(), config=_config())
    parameters = _parameters()
    ordinary = ordinary_evaluator(state=state, parameters=parameters, temperature_K=293.15)
    no_bf = no_bf_evaluator(state=state, parameters=parameters, temperature_K=293.15)

    ag10 = state.material_index == 0
    ag25 = state.material_index == 1
    ordinary_fraction = ordinary.mechanism_result.pointwise["rate_fraction_BF"].eval()
    no_bf_fraction = no_bf.mechanism_result.pointwise["rate_fraction_BF"].eval()

    assert np.all(ordinary_fraction[ag10] > 0.0)
    np.testing.assert_array_equal(no_bf_fraction[ag10], 0.0)
    assert np.all(np.isneginf(no_bf.log_rate_BF.eval()[ag10]))
    np.testing.assert_allclose(no_bf.log_rate_total.eval()[ag25], ordinary.log_rate_total.eval()[ag25], rtol=1e-12)
    np.testing.assert_allclose(no_bf_fraction[ag25], ordinary_fraction[ag25], rtol=1e-12)


@pytest.mark.parametrize(
    "evaluator",
    [evaluate_agpd_co_bf_er_lh_ag10_no_bf, evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf],
)
def test_ag10_rate_is_independent_of_delta_g5_when_bf_is_disabled(evaluator):
    state = build_agpd_point_state(inputs=_inputs(), config=_config())
    less_favorable = evaluator(state=state, parameters=_parameters(deltaG5_0=0.10), temperature_K=293.15)
    more_favorable = evaluator(state=state, parameters=_parameters(deltaG5_0=-0.10), temperature_K=293.15)

    ag10 = state.material_index == 0
    ag25 = state.material_index == 1
    rate_low = less_favorable.log_rate_total.eval()
    rate_high = more_favorable.log_rate_total.eval()

    np.testing.assert_allclose(rate_low[ag10], rate_high[ag10], rtol=1e-12, atol=1e-12)
    assert np.any(np.abs(rate_low[ag25] - rate_high[ag25]) > 1e-6)


def test_uncapped_ag10_no_bf_ignores_configured_co_cap():
    parameters = _parameters()
    low_cap_state = build_agpd_point_state(inputs=_inputs(), config=_config(theta_CO_max=0.05))
    high_cap_state = build_agpd_point_state(inputs=_inputs(), config=_config(theta_CO_max=0.95))

    low_cap = evaluate_agpd_co_bf_er_lh_ag10_no_bf(
        state=low_cap_state,
        parameters=parameters,
        temperature_K=293.15,
    )
    high_cap = evaluate_agpd_co_bf_er_lh_ag10_no_bf(
        state=high_cap_state,
        parameters=parameters,
        temperature_K=293.15,
    )

    np.testing.assert_allclose(low_cap.log_rate_total.eval(), high_cap.log_rate_total.eval(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        low_cap.mechanism_result.pointwise["theta_CO"].eval(),
        high_cap.mechanism_result.pointwise["theta_CO"].eval(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert "theta_CO_site_occupation" not in low_cap.mechanism_result.pointwise


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_ag10_no_bf_models_reuse_base_configuration_and_are_all_material_only(model_name):
    config = {
        "prior_profiles": {"Ag10Pd90": {"CO_BF_ER_LH": {"parameters": {"sentinel": 1}}}},
        "composition_parameterizations": {
            "linear_xAg": {
                "x_reference": 0.5,
                "models": {"CO_BF_ER_LH": {"slopes": {}}},
            }
        },
        "surface_composition": {"Ag10Pd90": {"Ag_fraction": 0.10, "Pd_fraction": 0.90}},
    }

    assert model_name in available_agpd_models()
    assert model_name in available_agpd_all_material_models()
    assert get_agpd_model_definition(model_name).parameter_class is AgPdCOBFERRLHParameters
    assert get_agpd_prior_profile(config, "Ag10Pd90", model_name)["parameters"] == {"sentinel": 1}
    assert get_agpd_parameterization(config, model_name, "linear_xAg") == (0.5, {})

    with pytest.raises(ValueError, match="only for all-material fitting"):
        build_agpd_mechanism(model_name, "Ag10Pd90", config)
