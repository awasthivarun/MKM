from copy import deepcopy
from dataclasses import fields

import numpy as np
import pymc as pm
import pytest
import yaml

from mkm.mechanisms.pd_basic import PdCOERLHParameters
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
    get_agpd_parameterization_metadata,
)


def _config():
    with open("config/models/agpd_basic.yaml", "r") as file:
        return yaml.safe_load(file)


def _point_inputs(materials):
    return ModelPointInputs(
        materials=tuple(materials),
        material_index=np.arange(len(materials), dtype=np.int64),
        E_V_SHE=np.linspace(0.25, 0.35, len(materials)),
        ln_electrolyte_concentration=np.log(
            np.full(len(materials), 0.5)
        ),
        ln_CO_mole_fraction=np.log(
            np.full(len(materials), 0.1)
        ),
    )


def test_agpd_registry_keeps_all_chemical_mechanisms_and_reduced_pd_model():
    assert available_agpd_models() == (
        "BF",
        "BF_LH",
        "CO_BF_ER_LH",
        "CO_BF_ER_LH_capped",
        "CO_BF_ER_LH_capped_Ag10_no_BF",
        "CO_ER_LH",
    )

    assert available_agpd_all_material_models() == (
        "BF_LH",
        "CO_BF_ER_LH",
        "CO_BF_ER_LH_capped",
        "CO_BF_ER_LH_capped_Ag10_no_BF",
    )

    for name in available_agpd_models():
        assert get_agpd_model_definition(name).parameter_class is not None

    assert (
        get_agpd_model_definition("CO_BF_ER_LH_capped").parameter_class
        is get_agpd_model_definition("CO_BF_ER_LH").parameter_class
    )

    pd_definition = get_agpd_model_definition("CO_ER_LH")
    assert pd_definition.parameter_class is PdCOERLHParameters
    assert {field.name for field in fields(PdCOERLHParameters)} == {
        "deltaG1_0",
        "deltaG4_0",
        "beta_2_ER",
        "Gact1_0",
        "Gact2_ER_0",
        "Gact2_LH_0",
    }


def test_parameterization_profiles_are_configuration_driven():
    config = _config()

    assert available_agpd_parameterizations(
        config,
        "BF_LH",
    ) == ("shared",)

    base_parameterizations = set(
        available_agpd_parameterizations(
            config,
            "CO_BF_ER_LH",
        )
    )
    capped_parameterizations = set(
        available_agpd_parameterizations(
            config,
            "CO_BF_ER_LH_capped",
        )
    )

    assert base_parameterizations == {
        "shared",
        "linear_dG1",
        "linear_dG4",
        "linear_dG5",
        "linear_thermo",
        "linear_GactBF",
        "linear_GactER",
        "linear_oxidation_barriers",
        "linear_selected_energies",
        "linear_energies",
        "linear_xAg",
    }
    assert capped_parameterizations == base_parameterizations

    selected_x_reference, selected_slopes = get_agpd_parameterization(
        config,
        "CO_BF_ER_LH",
        "linear_selected_energies",
    )
    assert selected_x_reference == pytest.approx(0.5)
    assert set(selected_slopes) == {
        "deltaG1_0",
        "deltaG4_0",
        "deltaG5_0",
        "Gact2_BF_0",
        "Gact2_ER_0",
    }

    x_reference, slopes = get_agpd_parameterization(
        config,
        "CO_BF_ER_LH",
        "linear_xAg",
    )
    capped_x_reference, capped_slopes = get_agpd_parameterization(
        config,
        "CO_BF_ER_LH_capped",
        "linear_xAg",
    )

    assert x_reference == pytest.approx(0.5)
    assert set(slopes) == {
        "deltaG1_0",
        "deltaG4_0",
        "deltaG5_0",
        "beta_2_BF",
        "beta_2_ER",
        "q",
        "Gact1_0",
        "Gact2_BF_0",
        "Gact2_ER_0",
        "Gact2_LH_0",
    }
    assert capped_x_reference == x_reference
    assert capped_slopes == slopes

    metadata = get_agpd_parameterization_metadata(
        config,
        "CO_BF_ER_LH_capped",
        "linear_xAg",
    )
    assert metadata["name"] == "linear_xAg"
    assert metadata["x_reference"] == pytest.approx(0.5)
    assert metadata["slopes"] == slopes


def test_capped_model_reuses_uncapped_prior_and_slope_specs():
    config = _config()

    base_specs = get_agpd_all_material_parameter_specs(
        config,
        prior_material="Ag10Pd90",
        model_name="CO_BF_ER_LH",
        parameterization="linear_xAg",
    )
    capped_specs = get_agpd_all_material_parameter_specs(
        config,
        prior_material="Ag10Pd90",
        model_name="CO_BF_ER_LH_capped",
        parameterization="linear_xAg",
    )

    assert capped_specs == base_specs


def test_arbitrary_parameter_subset_can_receive_xag_slopes_without_domain_enforcement():
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
        name.endswith("_xAg_slope")
        and name != "beta_2_ER_xAg_slope"
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
        result = mechanism(
            _point_inputs(("Ag10Pd90", "Pd100"))
        )

    assert result.ln_rate.ndim == 1
    assert "beta_2_ER_xAg_slope" in {
        variable.name
        for variable in model.free_RVs
    }


@pytest.mark.parametrize(
    "model_name",
    ("CO_BF_ER_LH", "CO_BF_ER_LH_capped"),
)
def test_prediction_only_full_model_accepts_pure_pd_state(model_name):
    config = _config()

    mechanism = build_agpd_all_material_mechanism(
        model_name,
        ("Pd100",),
        config,
        prior_material="Ag10Pd90",
        parameterization="linear_xAg",
        prediction_only=True,
    )

    with pm.Model() as model:
        result = mechanism(
            _point_inputs(("Pd100",))
        )

    assert result.ln_rate.ndim == 1
    assert "rate_fraction_BF" not in result.pointwise
    assert {
        "rate_fraction_ER",
        "rate_fraction_LH",
    }.issubset(result.pointwise)

    if model_name == "CO_BF_ER_LH_capped":
        assert "theta_CO_site_occupation" in result.pointwise
    else:
        assert "theta_CO_site_occupation" not in result.pointwise

    free_names = {
        variable.name
        for variable in model.free_RVs
    }
    assert "Gact2_BF_0" in free_names
    assert "Gact2_BF_0_xAg_slope" in free_names


def test_reduced_pd_individual_mechanism_contains_no_bf_parameters():
    mechanism = build_agpd_mechanism(
        "CO_ER_LH",
        "Pd100",
        _config(),
    )
    with pm.Model() as model:
        result = mechanism(
            _point_inputs(("Pd100",))
        )

    free_names = {
        variable.name
        for variable in model.free_RVs
    }
    assert result.ln_rate.ndim == 1
    assert free_names == {
        "deltaG1_0",
        "deltaG4_0",
        "beta_2_ER",
        "Gact1_0",
        "Gact2_ER_0",
        "Gact2_LH_0",
    }


def test_individual_mechanism_rejects_different_material_set():
    mechanism = build_agpd_mechanism(
        "BF_LH",
        "Ag10Pd90",
        _config(),
    )
    with pm.Model(), pytest.raises(
        ValueError,
        match="exactly that one material",
    ):
        mechanism(
            _point_inputs(("Ag10Pd90", "Ag50Pd50"))
        )


def test_unknown_model_or_parameterization_is_rejected():
    config = _config()

    with pytest.raises(
        ValueError,
        match="Unknown AgPd model",
    ):
        get_agpd_model_definition("unknown")

    with pytest.raises(
        ValueError,
        match="Unknown AgPd parameterization",
    ):
        get_agpd_parameterization(
            config,
            "CO_BF_ER_LH",
            "unknown",
        )
