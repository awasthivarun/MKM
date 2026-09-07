import yaml

from mkm.inference.likelihoods import RATE_NORMAL, available_error_structures
from mkm.models.agpd_basic import (
    available_agpd_all_material_models,
    available_agpd_parameterizations,
    get_agpd_all_material_parameter_specs,
)


def _load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def test_agpd_preprocessing_and_model_metadata_are_consistent():
    preprocessing = _load_yaml("config/preprocessing/agpd_basic.yaml")
    model = _load_yaml("config/models/agpd_basic.yaml")

    assert preprocessing["dataset_name"] == model["dataset_name"]
    assert float(preprocessing["temperature_K"]) == float(model["temperature_K"])


def test_agpd_config_exposes_only_current_rate_likelihood_contract():
    config = _load_yaml("config/models/agpd_basic.yaml")
    likelihood = config["likelihood"]

    assert likelihood["name"] == RATE_NORMAL
    assert likelihood["sigma_form"] == "sigma_abs + sigma_rel * model_rate"
    assert tuple(
        likelihood["supported_error_structures"]
    ) == available_error_structures()
    assert likelihood["sigma_abs_prior_median_s_inv"] > 0
    assert likelihood["sigma_rel_prior_median"] > 0


def test_capped_co_coverage_configuration_is_complete():
    config = _load_yaml("config/models/agpd_basic.yaml")

    caps = config["co_coverage_cap"]

    assert set(caps) == set(config["surface_composition"])

    for value in caps.values():
        value = float(value)
        assert 0.0 < value <= 1.0



def test_every_configured_all_material_profile_has_complete_parameter_specs():
    config = _load_yaml("config/models/agpd_basic.yaml")

    for model_name in available_agpd_all_material_models():
        for parameterization in available_agpd_parameterizations(
            config,
            model_name,
        ):
            specs = get_agpd_all_material_parameter_specs(
                config,
                prior_material="Ag10Pd90",
                model_name=model_name,
                parameterization=parameterization,
            )
            assert specs


def test_pd100_reduced_model_prior_matches_reduced_parameter_set():
    from dataclasses import fields

    from mkm.mechanisms.pd_basic import PdCOERLHParameters
    from mkm.models.agpd_basic import get_agpd_prior_profile

    config = _load_yaml("config/models/agpd_basic.yaml")
    profile = get_agpd_prior_profile(
        config,
        "Pd100",
        "CO_ER_LH",
    )

    assert set(profile["parameters"]) == {
        field.name
        for field in fields(PdCOERLHParameters)
    }


def test_active_composition_parameterizations_have_expected_slope_sets():
    config = _load_yaml("config/models/agpd_basic.yaml")
    profiles = config["composition_parameterizations"]

    assert set(profiles) == {"shared", "linear_xAg"}
    assert profiles["shared"]["models"]["CO_BF_ER_LH"]["slopes"] == {}

    expected_slopes = {
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
    slopes = profiles["linear_xAg"]["models"]["CO_BF_ER_LH"]["slopes"]
    assert set(slopes) == expected_slopes


def test_prior_profiles_keep_only_canonical_co_sources():
    config = _load_yaml("config/models/agpd_basic.yaml")
    profiles = config["prior_profiles"]

    for material in ("Ag10Pd90", "Ag25Pd75", "Ag50Pd50", "Ag75Pd25", "Ag90Pd10"):
        assert set(profiles[material]) == {"CO_BF_ER_LH"}

    assert set(profiles["Pd100"]) == {"CO_ER_LH"}


def test_full_linear_xag_contains_all_mechanism_slopes():
    config = _load_yaml("config/models/agpd_basic.yaml")

    slopes = config[
        "composition_parameterizations"
    ]["linear_xAg"]["models"]["CO_BF_ER_LH"]["slopes"]

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

    for parameter_name in {
        "beta_2_BF",
        "beta_2_ER",
        "q",
    }:
        prior = slopes[parameter_name]
        assert prior["distribution"] == "truncated_normal"
        assert float(prior["lower"]) == -1.0
        assert float(prior["upper"]) == 1.0
