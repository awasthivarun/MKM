"""Integration checks for dynamic model priors and mechanism assembly."""

from dataclasses import fields

import numpy as np
import pytest

from mkm.mechanisms.base import MechanismResult, validate_mechanism_result
from mkm.model_inputs import ModelPointInputs
from mkm.models import agpd_basic
from mkm.models.agpd_registry import get_agpd_model_definition
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config


DEFAULTS = dict(
    deltaG1_0=-0.20, deltaG4_0=0.02, deltaG5_0=0.02, deltaG3_0=-0.08, q=0.5,
    Gact1_0=0.46, Gact4_0=0.65, Gact5_0=0.65, Gact2_BF_0=0.70,
    Gact2_ER_0=0.72, Gact2_LH_0=0.70, Gact3_0=0.68,
    beta_4=0.5, beta_5=0.5, beta_2_BF=0.5, beta_2_ER=0.5, beta_3=0.5,
    z_CO_CO_eV=0.01, z_CO_OH_eV=0.01, z_OH_OH_eV=0.01,
)


def _inputs(materials):
    return ModelPointInputs(
        materials=materials,
        material_index=np.arange(len(materials), dtype=np.int64),
        E_V_SHE=np.zeros(len(materials)),
        ln_electrolyte_concentration=np.full(len(materials), np.log(0.5)),
        ln_CO_mole_fraction=np.full(len(materials), np.log(0.1)),
    )


def test_lateral_prior_projection_and_model_metadata():
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    assert config["lateral_interactions"]["bep_coefficient"] == 0.65
    for name in ("c_ER", "xo_ER", "cxo_CO_BF_ER_LH_3_pOH_sOH"):
        definition = get_agpd_model_definition(name)
        profile = agpd_basic.get_agpd_prior_profile(config, "Ag50Pd50", name)
        assert set(profile["parameters"]) == {field.name for field in fields(definition.parameter_class)}
        for key, prior in profile["parameters"].items():
            if key == "z_CO_CO_eV":
                assert 0 <= prior["lower"] < prior["upper"]
            elif key.startswith("z_"):
                assert prior["lower"] < 0 < prior["upper"]


@pytest.mark.integration
def test_lateral_and_reversible_models_flow_through_individual_assembly(monkeypatch):
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    config["lateral_interactions"]["fixed_point_iterations"] = 64
    monkeypatch.setattr(agpd_basic, "build_named_priors", lambda specs: {name: DEFAULTS[name] for name in specs})
    for name in ("CO_ER_3", "x_ER", "cxo_CO_BF_ER_LH_3_pOH_sOH"):
        mechanism = agpd_basic.build_agpd_mechanism(name, "Ag50Pd50", config)
        evaluated = mechanism(_inputs(("Ag50Pd50",)))
        assert isinstance(evaluated, MechanismResult)
        validate_mechanism_result(evaluated)
        assert np.all(np.isfinite(evaluated.ln_rate.eval()))


def test_lateral_model_flows_through_all_material_assembly(monkeypatch):
    config = load_agpd_model_config(ProjectPaths.discover(__file__))
    config["lateral_interactions"]["fixed_point_iterations"] = 64
    monkeypatch.setattr(agpd_basic, "build_named_priors", lambda specs: {name: DEFAULTS[name] for name in specs})
    mechanism = agpd_basic.build_agpd_all_material_mechanism(
        "cxo_CO_BF_ER_LH_3_pOH_sOH", ("Pd100", "Ag50Pd50"), config,
        parameterization="shared", prediction_only=True,
    )
    evaluated = mechanism(_inputs(("Pd100", "Ag50Pd50")))
    assert isinstance(evaluated, MechanismResult)
    validate_mechanism_result(evaluated)
    assert np.all(np.isfinite(evaluated.ln_rate.eval()))
