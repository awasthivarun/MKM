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
    assert tuple(likelihood["supported_error_structures"]) == available_error_structures()
    assert likelihood["sigma_abs_prior_median_s_inv"] > 0
    assert likelihood["sigma_rel_prior_median"] > 0


def test_every_configured_all_material_profile_has_complete_parameter_specs():
    config = _load_yaml("config/models/agpd_basic.yaml")
    for model_name in available_agpd_all_material_models():
        for parameterization in available_agpd_parameterizations(config, model_name):
            specs = get_agpd_all_material_parameter_specs(
                config,
                prior_material="Ag10Pd90",
                model_name=model_name,
                parameterization=parameterization,
            )
            assert specs
