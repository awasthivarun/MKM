from copy import deepcopy

import pytest
import yaml

from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    rate_normal_likelihood_kwargs,
    resolve_agpd_fit_specification,
)


def _config():
    with open("config/models/agpd_basic.yaml", "r") as file:
        return yaml.safe_load(file)


def test_individual_fit_uses_one_material_labelled_error_pair():
    specification = resolve_agpd_fit_specification(
        _config(),
        model_name="BF_LH",
        material="Ag10Pd90",
        error_structure="material",
    )

    assert specification.fit_scope == "individual"
    assert specification.material == "Ag10Pd90"
    assert specification.parameterization is None
    assert specification.error_structure == "material"


def test_individual_fit_rejects_shared_error_option():
    with pytest.raises(ValueError, match="material-indexed"):
        resolve_agpd_fit_specification(
            _config(),
            model_name="BF_LH",
            material="Ag10Pd90",
            error_structure="shared",
        )


def test_all_material_fit_supports_shared_or_material_error():
    for error_structure in ("shared", "material"):
        specification = resolve_agpd_fit_specification(
            _config(),
            model_name="CO_BF_ER_LH",
            all_materials=True,
            parameterization="linear_xAg",
            error_structure=error_structure,
        )
        assert specification.fit_scope == "all_materials"
        assert specification.error_structure == error_structure


def test_profile_name_and_arbitrary_slope_are_preserved_in_specification():
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
    specification = resolve_agpd_fit_specification(
        config,
        model_name="CO_BF_ER_LH",
        all_materials=True,
        parameterization="linear_beta_er",
        error_structure="shared",
    )
    specs = all_parameter_specs(specification, config)

    assert specification.parameterization == "linear_beta_er"
    assert "beta_2_ER_xAg_slope" in specs
    assert {"sigma_rate_abs", "sigma_rate_rel"}.issubset(specs)


def test_likelihood_configuration_requires_exact_current_sigma_form():
    config = _config()
    kwargs = rate_normal_likelihood_kwargs(config)
    assert kwargs["sigma_abs_prior_median_s_inv"] > 0
    assert kwargs["sigma_rel_prior_median"] > 0

    config["likelihood"]["sigma_form"] = "sqrt(sigma_abs**2 + sigma_rel**2 * model_rate**2)"
    with pytest.raises(ValueError, match="Configured sigma form"):
        rate_normal_likelihood_kwargs(config)
