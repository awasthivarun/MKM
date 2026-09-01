from copy import deepcopy

import numpy as np
import pymc as pm
import pytest
import yaml

from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import (
    available_agpd_all_material_models,
    available_agpd_models,
    available_agpd_parameterizations,
    build_agpd_all_material_mechanism,
    build_agpd_mechanism,
    get_agpd_all_material_parameter_specs,
    get_agpd_model_definition,
    get_agpd_parameterization,
)


def _config():
    with open("config/models/agpd_basic.yaml", "r") as file:
        return yaml.safe_load(file)


def _point_inputs(materials):
    return ModelPointInputs(
        materials=tuple(materials),
        material_index=np.arange(len(materials), dtype=np.int64),
        E_V_SHE=np.linspace(0.25, 0.35, len(materials)),
        ln_electrolyte_concentration=np.log(np.full(len(materials), 0.5)),
        ln_CO_mole_fraction=np.log(np.full(len(materials), 0.1)),
    )


def test_agpd_registry_keeps_all_chemical_mechanisms():
    assert available_agpd_models() == ("BF", "BF_LH", "CO_BF_ER_LH")
    assert available_agpd_all_material_models() == ("BF_LH", "CO_BF_ER_LH")
    for name in available_agpd_models():
        assert get_agpd_model_definition(name).parameter_class is not None


def test_parameterization_profiles_are_configuration_driven():
    config = _config()
    assert available_agpd_parameterizations(config, "BF_LH") == ("shared",)
    assert set(available_agpd_parameterizations(config, "CO_BF_ER_LH")) == {
        "shared",
        "linear_xAg",
    }

    x_reference, slopes = get_agpd_parameterization(
        config,
        "CO_BF_ER_LH",
        "linear_xAg",
    )
    assert x_reference == pytest.approx(0.5)
    assert set(slopes) == {
        "deltaG1_0",
        "deltaG4_0",
        "deltaG5_0",
        "Gact2_BF_0",
        "Gact2_ER_0",
    }


def test_arbitrary_parameter_subset_can_receive_xag_slopes():
    config = deepcopy(_config())
    config["composition_parameterizations"]["linear_beta_er"] = {
        "x_reference": 0.5,
        "models": {
            "CO_BF_ER_LH": {
                "slopes": {
                    "beta_2_ER": {
                        "distribution": "normal",
                        "mu": 0.0,
                        "sigma": 0.25,
                    }
                }
            }
        },
    }

    specs = get_agpd_all_material_parameter_specs(
        config,
        prior_material="Ag10Pd90",
        model_name="CO_BF_ER_LH",
        parameterization="linear_beta_er",
    )
    assert "beta_2_ER_xAg_slope" in specs
    assert not any(
        name.endswith("_xAg_slope") and name != "beta_2_ER_xAg_slope"
        for name in specs
    )

    mechanism = build_agpd_all_material_mechanism(
        "CO_BF_ER_LH",
        ("Ag10Pd90", "Pd100"),
        config,
        prior_material="Ag10Pd90",
        parameterization="linear_beta_er",
    )
    with pm.Model() as model:
        result = mechanism(_point_inputs(("Ag10Pd90", "Pd100")))

    assert result.ln_rate.ndim == 1
    assert "beta_2_ER_xAg_slope" in {variable.name for variable in model.free_RVs}


def test_individual_mechanism_rejects_different_material_set():
    mechanism = build_agpd_mechanism("BF_LH", "Ag10Pd90", _config())
    with pm.Model(), pytest.raises(ValueError, match="exactly that one material"):
        mechanism(_point_inputs(("Ag10Pd90", "Ag50Pd50")))


def test_unknown_model_or_parameterization_is_rejected():
    config = _config()
    with pytest.raises(ValueError, match="Unknown AgPd model"):
        get_agpd_model_definition("unknown")
    with pytest.raises(ValueError, match="Unknown AgPd parameterization"):
        get_agpd_parameterization(config, "CO_BF_ER_LH", "unknown")
