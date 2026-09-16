import numpy as np
import pytensor
import yaml

from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLHParameters,
    build_agpd_point_state,
    evaluate_agpd_co_bf_er_lh_ag10_no_er,
)
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import (
    available_agpd_all_material_models,
    get_agpd_model_definition,
    get_agpd_parameterization,
)
from mkm.postprocessing.drc import transition_state_controls


def _config():
    return load_agpd_model_config(ProjectPaths.discover(__file__))


def test_ag10_no_er_masks_er_only_for_ag10():
    config = _config()
    materials = ("Ag10Pd90", "Ag50Pd50")
    point_inputs = ModelPointInputs(
        materials=materials,
        material_index=np.array([0, 1], dtype=np.int64),
        E_V_SHE=np.array([0.10, 0.10]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.1, 0.1])),
    )
    state = build_agpd_point_state(inputs=point_inputs, config=config)
    parameters = AgPdCOBFERRLHParameters(
        deltaG1_0=-0.12,
        deltaG4_0=0.09,
        deltaG5_0=-0.05,
        beta_2_BF=0.2,
        beta_2_ER=0.3,
        q=0.3,
        Gact1_0=0.56,
        Gact2_BF_0=0.73,
        Gact2_ER_0=0.73,
        Gact2_LH_0=0.76,
    )
    result = evaluate_agpd_co_bf_er_lh_ag10_no_er(
        state=state,
        parameters=parameters,
        temperature_K=config["temperature_K"],
    )
    function = pytensor.function(
        [],
        [
            result.mechanism_result.pointwise["rate_fraction_ER"],
            result.mechanism_result.pointwise["rate_fraction_BF"],
            result.mechanism_result.pointwise["rate_fraction_LH"],
        ],
    )
    er, bf, lh = function()
    assert er[0] == 0.0
    assert er[1] > 0.0
    assert bf[0] > 0.0
    assert lh[0] > 0.0
    np.testing.assert_allclose(er + bf + lh, 1.0, rtol=0.0, atol=1e-10)


def test_ag10_no_er_registry_prior_alias_and_drc_controls():
    config = _config()
    assert "CO_BF_ER_LH_Ag10_no_ER" in available_agpd_all_material_models()
    assert (
        get_agpd_model_definition("CO_BF_ER_LH_Ag10_no_ER").parameter_class
        is get_agpd_model_definition("CO_BF_ER_LH").parameter_class
    )
    assert get_agpd_parameterization(config, "CO_BF_ER_LH_Ag10_no_ER", "linear_xAg") == get_agpd_parameterization(
        config, "CO_BF_ER_LH", "linear_xAg"
    )
    assert tuple(control.name for control in transition_state_controls("CO_BF_ER_LH_Ag10_no_ER")) == (
        "CO_adsorption", "BF", "ER", "LH"
    )
