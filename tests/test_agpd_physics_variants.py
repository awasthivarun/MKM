"""Canonical model names and the analytical CO-QEA variants."""

from dataclasses import fields

import numpy as np
import pymc as pm
import pytest

from mkm.mechanisms.agpd_basic import (
    AgPdBFERRLHParameters,
    AgPdERParameters,
    build_agpd_point_state,
    evaluate_agpd_bf_er_lh,
    evaluate_agpd_er,
)
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import build_agpd_all_material_mechanism, get_agpd_model_definition, get_agpd_prior_profile
from mkm.models.agpd_physics import parse_agpd_physics
from mkm.postprocessing.drc import transition_state_controls
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.workflows.agpd_fit import resolve_agpd_fit_specification


def _state():
    inputs = ModelPointInputs(
        materials=("Ag50Pd50",),
        material_index=np.array([0, 0], dtype=np.int64),
        E_V_SHE=np.array([0.0, 0.1]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 1.0])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.1])),
    )
    config = {
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
        "surface_composition": {"Ag50Pd50": {"Ag_fraction": 0.5, "Pd_fraction": 0.5}},
    }
    return build_agpd_point_state(inputs, config)


@pytest.mark.parametrize("name", (
    "ER", "LH", "BF", "BF_ER", "BF_LH", "ER_LH", "BF_ER_LH", "CO_BF_ER_LH",
    "BF_ER_LH_3", "CO_BF_ER_LH_3_pOH", "cxo_CO_BF_ER_LH_pOH",
    "cx_CO_BF_ER_sOH", "o_ER_pOH", "cxo_CO_BF_ER_LH_3_pOH_sOH_capped",
))
def test_canonical_physics_name_roundtrip(name):
    assert parse_agpd_physics(name).name == name


@pytest.mark.parametrize("name", (
    "", "CO", "ER_BF", "BF_LH_ER", "xc_ER", "cc_ER", "CO_BF_pOH_ER", "BF_sOH_pOH",
    "BF_3_CO", "BF_random", "c_x_ER", "co_", "BF_ER_BF",
))
def test_invalid_or_noncanonical_names_are_rejected(name):
    with pytest.raises(ValueError):
        parse_agpd_physics(name)


def test_lateral_interactions_are_available_but_modifiers_still_rejected():
    definition = get_agpd_model_definition("c_ER")
    assert {field.name for field in fields(definition.parameter_class)} & {"z_CO_CO_eV"}
    with pytest.raises(ValueError, match="not yet implemented"):
        get_agpd_model_definition("ER_capped")
    assert get_agpd_model_definition("BF_ER_LH_3").supports_individual


def test_qea_er_equals_legacy_qea_er():
    state = _state()
    legacy_parameters = AgPdERParameters(deltaG1_0=-0.3, deltaG4_0=0.05, beta_2_ER=0.4, Gact2_ER_0=0.72)
    definition = get_agpd_model_definition("ER")
    assert {f.name for f in fields(definition.parameter_class)} == {
        "deltaG1_0", "deltaG4_0", "beta_2_ER", "Gact2_ER_0"
    }
    new_parameters = definition.parameter_class(**vars(legacy_parameters))
    old = evaluate_agpd_er(state, legacy_parameters, 293.15)
    new = definition.evaluator(state, new_parameters, 293.15)
    np.testing.assert_allclose(new.mechanism_result.ln_rate.eval(), old.mechanism_result.ln_rate.eval())
    for name in ("theta_CO", "theta_OH_Pd", "theta_empty_Pd"):
        np.testing.assert_allclose(
            new.mechanism_result.pointwise[name].eval(), old.mechanism_result.pointwise[name].eval()
        )


def test_qea_three_path_equals_legacy_qea_three_path():
    state = _state()
    legacy_parameters = AgPdBFERRLHParameters(
        deltaG1_0=-0.25, deltaG4_0=0.05, deltaG5_0=0.12,
        beta_2_BF=0.45, beta_2_ER=0.5, q=0.65,
        Gact2_BF_0=0.68, Gact2_ER_0=0.74, Gact2_LH_0=0.80,
    )
    definition = get_agpd_model_definition("BF_ER_LH")
    new = definition.evaluator(state, definition.parameter_class(**vars(legacy_parameters)), 293.15)
    old = evaluate_agpd_bf_er_lh(state, legacy_parameters, 293.15)
    np.testing.assert_allclose(new.mechanism_result.ln_rate.eval(), old.mechanism_result.ln_rate.eval())
    for pathway in ("BF", "ER", "LH"):
        np.testing.assert_allclose(
            new.mechanism_result.pointwise[f"ln_rate_{pathway}"].eval(),
            old.mechanism_result.pointwise[f"ln_rate_{pathway}"].eval(),
        )


def test_qea_registry_projects_existing_priors_and_drc_controls():
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    profile = get_agpd_prior_profile(config, "Ag50Pd50", "ER_LH")
    assert set(profile["parameters"]) == {
        "deltaG1_0", "deltaG4_0", "Gact2_ER_0", "Gact2_LH_0", "beta_2_ER"
    }
    assert [control.parameter for control in transition_state_controls("ER_LH")] == [
        "Gact2_ER_0", "Gact2_LH_0"
    ]
    spec = resolve_agpd_fit_specification(config, model_name="ER_LH", material="Ag50Pd50")
    assert spec.model_name == "ER_LH"
    pooled = resolve_agpd_fit_specification(config, model_name="ER_LH", all_materials=True)
    assert pooled.model_name == "ER_LH"


def test_all_material_qea_builds_only_selected_parameters():
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    materials = tuple(config["surface_composition"])
    mechanism = build_agpd_all_material_mechanism("ER", materials, config, parameterization="shared")
    inputs = ModelPointInputs(
        materials=materials,
        material_index=np.arange(len(materials), dtype=np.int64),
        E_V_SHE=np.full(len(materials), 0.1),
        ln_electrolyte_concentration=np.log(np.full(len(materials), 0.5)),
        ln_CO_mole_fraction=np.log(np.full(len(materials), 0.1)),
    )
    with pm.Model() as model:
        prediction = mechanism(inputs)
    assert {rv.name for rv in model.free_RVs} == {
        "deltaG1_0", "deltaG4_0", "beta_2_ER", "Gact2_ER_0"
    }
    assert prediction.ln_rate.ndim == 1
