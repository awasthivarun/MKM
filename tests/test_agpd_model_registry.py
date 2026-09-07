from copy import deepcopy
from dataclasses import fields

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
    get_agpd_parameterization_metadata,
    get_agpd_prior_profile,
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


def test_agpd_registry_keeps_individual_co_pathway_subsets_and_all_material_models():
    assert available_agpd_models() == (
        "CO_LH",
        "CO_ER",
        "CO_BF",
        "CO_ER_LH",
        "CO_BF_LH",
        "CO_BF_ER",
        "CO_BF_ER_LH",
        "CO_BF_ER_LH_Ag10_no_BF",
        "CO_BF_ER_LH_capped",
        "CO_BF_ER_LH_capped_Ag10_no_BF",
        "CO_BF_ER_LH_fitted_caps_Ag10_no_BF",
    )

    assert available_agpd_all_material_models() == (
        "CO_BF_ER_LH",
        "CO_BF_ER_LH_Ag10_no_BF",
        "CO_BF_ER_LH_capped",
        "CO_BF_ER_LH_capped_Ag10_no_BF",
        "CO_BF_ER_LH_fitted_caps_Ag10_no_BF",
    )

    for name in available_agpd_models():
        assert get_agpd_model_definition(name).parameter_class is not None

    assert (
        get_agpd_model_definition("CO_BF_ER_LH_capped").parameter_class
        is get_agpd_model_definition("CO_BF_ER_LH").parameter_class
    )


@pytest.mark.parametrize(
    ("model_name", "expected_parameters"),
    [
        ("CO_LH", {"deltaG1_0", "deltaG4_0", "Gact1_0", "Gact2_LH_0"}),
        ("CO_ER", {"deltaG1_0", "deltaG4_0", "beta_2_ER", "Gact1_0", "Gact2_ER_0"}),
        (
            "CO_BF",
            {"deltaG1_0", "deltaG4_0", "deltaG5_0", "beta_2_BF", "q", "Gact1_0", "Gact2_BF_0"},
        ),
        (
            "CO_ER_LH",
            {"deltaG1_0", "deltaG4_0", "beta_2_ER", "Gact1_0", "Gact2_ER_0", "Gact2_LH_0"},
        ),
        (
            "CO_BF_LH",
            {
                "deltaG1_0",
                "deltaG4_0",
                "deltaG5_0",
                "beta_2_BF",
                "q",
                "Gact1_0",
                "Gact2_BF_0",
                "Gact2_LH_0",
            },
        ),
        (
            "CO_BF_ER",
            {
                "deltaG1_0",
                "deltaG4_0",
                "deltaG5_0",
                "beta_2_BF",
                "beta_2_ER",
                "q",
                "Gact1_0",
                "Gact2_BF_0",
                "Gact2_ER_0",
            },
        ),
    ],
)
def test_individual_co_subset_parameter_classes_contain_only_active_parameters(model_name, expected_parameters):
    definition = get_agpd_model_definition(model_name)
    assert {field.name for field in fields(definition.parameter_class)} == expected_parameters


@pytest.mark.parametrize(
    ("model_name", "expected_parameters"),
    [
        ("CO_LH", {"deltaG1_0", "deltaG4_0", "Gact1_0", "Gact2_LH_0"}),
        ("CO_ER", {"deltaG1_0", "deltaG4_0", "beta_2_ER", "Gact1_0", "Gact2_ER_0"}),
        (
            "CO_BF",
            {"deltaG1_0", "deltaG4_0", "deltaG5_0", "beta_2_BF", "q", "Gact1_0", "Gact2_BF_0"},
        ),
        (
            "CO_ER_LH",
            {"deltaG1_0", "deltaG4_0", "beta_2_ER", "Gact1_0", "Gact2_ER_0", "Gact2_LH_0"},
        ),
        (
            "CO_BF_LH",
            {
                "deltaG1_0",
                "deltaG4_0",
                "deltaG5_0",
                "beta_2_BF",
                "q",
                "Gact1_0",
                "Gact2_BF_0",
                "Gact2_LH_0",
            },
        ),
        (
            "CO_BF_ER",
            {
                "deltaG1_0",
                "deltaG4_0",
                "deltaG5_0",
                "beta_2_BF",
                "beta_2_ER",
                "q",
                "Gact1_0",
                "Gact2_BF_0",
                "Gact2_ER_0",
            },
        ),
        (
            "CO_BF_ER_LH",
            {
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
            },
        ),
    ],
)
def test_alloy_individual_co_models_build_only_their_active_parameters(model_name, expected_parameters):
    mechanism = build_agpd_mechanism(model_name, "Ag50Pd50", _config())
    with pm.Model() as model:
        result = mechanism(_point_inputs(("Ag50Pd50",)))

    assert result.ln_rate.ndim == 1
    assert {variable.name for variable in model.free_RVs} == expected_parameters


def test_individual_co_priors_use_one_canonical_parent_per_material_family():
    config = _config()
    alloy_parent = config["prior_profiles"]["Ag50Pd50"]["CO_BF_ER_LH"]["parameters"]
    pd_parent = config["prior_profiles"]["Pd100"]["CO_ER_LH"]["parameters"]

    for model_name in ("CO_LH", "CO_ER", "CO_BF", "CO_ER_LH", "CO_BF_LH", "CO_BF_ER", "CO_BF_ER_LH"):
        profile = get_agpd_prior_profile(config, "Ag50Pd50", model_name)
        assert all(profile["parameters"][name] == alloy_parent[name] for name in profile["parameters"])

    for model_name in ("CO_LH", "CO_ER", "CO_ER_LH"):
        profile = get_agpd_prior_profile(config, "Pd100", model_name)
        assert all(profile["parameters"][name] == pd_parent[name] for name in profile["parameters"])


def test_yaml_keeps_only_canonical_individual_co_prior_profiles():
    config = _config()
    for material in ("Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10"):
        assert set(config["prior_profiles"][material]) == {"CO_BF_ER_LH"}
    assert set(config["prior_profiles"]["Pd100"]) == {"CO_ER_LH"}


def test_parameterization_profiles_are_configuration_driven():
    config = _config()

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

    assert base_parameterizations == {"shared", "linear_xAg"}
    assert capped_parameterizations == base_parameterizations
    assert available_agpd_parameterizations(
        config,
        "CO_BF_ER_LH_fitted_caps_Ag10_no_BF",
    ) == ("linear_xAg",)

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


def test_fitted_caps_model_builds_one_free_cap_per_configured_material():
    config = _config()
    model_name = "CO_BF_ER_LH_fitted_caps_Ag10_no_BF"
    materials = tuple(config["surface_composition"])

    specs = get_agpd_all_material_parameter_specs(
        config,
        prior_material="Ag10Pd90",
        model_name=model_name,
        parameterization="linear_xAg",
    )
    cap_names = {f"theta_CO_max_{material}" for material in materials}
    assert cap_names.issubset(specs)
    assert all(specs[name] == config["fitted_cap_calibration"]["theta_CO_max_prior"] for name in cap_names)

    mechanism = build_agpd_all_material_mechanism(
        model_name,
        materials,
        config,
        prior_material="Ag10Pd90",
        parameterization="linear_xAg",
    )
    with pm.Model() as model:
        result = mechanism(_point_inputs(materials))

    free_names = {variable.name for variable in model.free_RVs}
    assert cap_names.issubset(free_names)
    assert "theta_CO_site_occupation" in result.pointwise


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
    ("CO_BF_ER_LH", "CO_BF_ER_LH_capped", "CO_BF_ER_LH_fitted_caps_Ag10_no_BF"),
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

    if model_name in {"CO_BF_ER_LH_capped", "CO_BF_ER_LH_fitted_caps_Ag10_no_BF"}:
        assert "theta_CO_site_occupation" in result.pointwise
    else:
        assert "theta_CO_site_occupation" not in result.pointwise

    free_names = {
        variable.name
        for variable in model.free_RVs
    }
    assert "Gact2_BF_0" in free_names
    assert "Gact2_BF_0_xAg_slope" in free_names


def test_pd_individual_co_er_lh_contains_no_bf_parameters():
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
        "CO_BF_ER",
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

    for retired_model in ("BF", "BF_LH"):
        with pytest.raises(ValueError, match="Unknown AgPd model"):
            get_agpd_model_definition(retired_model)

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
