import numpy as np
import pytensor

from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLHParameters,
    AgPdPointState,
    evaluate_agpd_co_bf_er_lh,
)
from mkm.models.agpd_basic import available_agpd_composition_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import available_agpd_materials, validate_agpd_material


def test_composition_models_cover_pd100_through_non_bf_pathways():
    assert available_agpd_composition_models() == ("BF_LH", "CO_BF_ER_LH")


def test_available_materials_come_from_surface_composition():
    config = {
        "surface_composition": {
            "Pd100": {"Ag_fraction": 0.0, "Pd_fraction": 1.0},
            "Ag10Pd90": {"Ag_fraction": 0.1, "Pd_fraction": 0.9},
        }
    }

    assert available_agpd_materials(config) == ("Pd100", "Ag10Pd90")
    validate_agpd_material(config, "Pd100")


def test_composition_output_path_is_separate_from_individual_materials(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    result = paths.agpd_composition_posterior_output_dir(
        composition_model="shared",
        model_name="BF_LH",
        likelihood_name="setup_intercept",
    )

    assert result == (
        tmp_path / "results" / "AgPd_COOx_basic" / "posterior"
        / "composition" / "shared" / "setup_intercept" / "BF_LH"
    )


def test_co_bf_er_lh_accepts_pd100_when_ag_is_present_elsewhere():
    state = AgPdPointState(
        E_V_SHE=np.array([0.35, 0.35]),
        ln_a_OH=np.log(np.array([0.5, 0.5])),
        ln_a_CO=np.log(np.array([0.1, 0.1])),
        Ag_fraction=np.array([0.0, 0.1]),
        Pd_fraction=np.array([1.0, 0.9]),
    )
    parameters = AgPdCOBFERRLHParameters(
        deltaG1_0=-0.12,
        deltaG4_0=0.11,
        deltaG5_0=0.03,
        beta_2_BF=0.10,
        beta_2_ER=0.50,
        q=0.30,
        Gact1_0=0.70,
        Gact2_BF_0=0.67,
        Gact2_ER_0=0.72,
        Gact2_LH_0=0.73,
    )

    result = evaluate_agpd_co_bf_er_lh(state, parameters, temperature_K=293.15)
    function = pytensor.function([], result.mechanism_result.ln_rate)
    values = np.asarray(function(), dtype=float)

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
