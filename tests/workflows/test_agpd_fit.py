from copy import deepcopy

import pytest
import yaml

from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    rate_normal_likelihood_kwargs,
    resolved_parameterization_metadata,
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


def test_pd100_individual_fit_requires_reduced_pd_model():
    config = _config()
    for model_name in ("BF", "BF_LH", "CO_BF_ER_LH"):
        with pytest.raises(ValueError, match="reduced Pd-only model 'CO_ER_LH'"):
            resolve_agpd_fit_specification(
                config,
                model_name=model_name,
                material="Pd100",
                error_structure="material",
            )

    specification = resolve_agpd_fit_specification(
        config,
        model_name="CO_ER_LH",
        material="Pd100",
        error_structure="material",
    )
    assert specification.material == "Pd100"
    assert specification.model_name == "CO_ER_LH"


def test_reduced_pd_model_is_not_available_for_alloys_or_all_material_fit():
    config = _config()
    with pytest.raises(ValueError, match="reserved for individual Pd100"):
        resolve_agpd_fit_specification(
            config,
            model_name="CO_ER_LH",
            material="Ag50Pd50",
            error_structure="material",
        )

    with pytest.raises(ValueError, match="not available for all-material fitting"):
        resolve_agpd_fit_specification(
            config,
            model_name="CO_ER_LH",
            all_materials=True,
            parameterization="shared",
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
    metadata = resolved_parameterization_metadata(specification, config)

    assert specification.parameterization == "linear_beta_er"
    assert "beta_2_ER_xAg_slope" in specs
    assert {"sigma_rate_abs", "sigma_rate_rel"}.issubset(specs)
    assert metadata["name"] == "linear_beta_er"
    assert set(metadata["slopes"]) == {"beta_2_ER"}


def test_likelihood_configuration_requires_exact_current_sigma_form():
    config = _config()
    kwargs = rate_normal_likelihood_kwargs(config)
    assert kwargs["sigma_abs_prior_median_s_inv"] > 0
    assert kwargs["sigma_rel_prior_median"] > 0

    config["likelihood"]["sigma_form"] = "sqrt(sigma_abs**2 + sigma_rel**2 * model_rate**2)"
    with pytest.raises(ValueError, match="Configured sigma form"):
        rate_normal_likelihood_kwargs(config)


@pytest.mark.parametrize(
    ("parameterization", "expected_slopes"),
    [
        ("linear_dG1", {"deltaG1_0_xAg_slope"}),
        ("linear_dG4", {"deltaG4_0_xAg_slope"}),
        ("linear_dG5", {"deltaG5_0_xAg_slope"}),
        (
            "linear_thermo",
            {"deltaG1_0_xAg_slope", "deltaG4_0_xAg_slope", "deltaG5_0_xAg_slope"},
        ),
        ("linear_GactBF", {"Gact2_BF_0_xAg_slope"}),
        ("linear_GactER", {"Gact2_ER_0_xAg_slope"}),
        (
            "linear_oxidation_barriers",
            {
                "Gact2_BF_0_xAg_slope",
                "Gact2_ER_0_xAg_slope",
                "Gact2_LH_0_xAg_slope",
            },
        ),
        (
            "linear_selected_energies",
            {
                "deltaG1_0_xAg_slope",
                "deltaG4_0_xAg_slope",
                "deltaG5_0_xAg_slope",
                "Gact2_BF_0_xAg_slope",
                "Gact2_ER_0_xAg_slope",
            },
        ),
        (
            "linear_xAg",
            {
                "deltaG1_0_xAg_slope",
                "deltaG4_0_xAg_slope",
                "deltaG5_0_xAg_slope",
                "beta_2_BF_xAg_slope",
                "beta_2_ER_xAg_slope",
                "q_xAg_slope",
                "Gact1_0_xAg_slope",
                "Gact2_BF_0_xAg_slope",
                "Gact2_ER_0_xAg_slope",
                "Gact2_LH_0_xAg_slope",
            },
        ),
    ],
)
def test_named_composition_profiles_resolve_for_shared_error(
    parameterization,
    expected_slopes,
):
    config = _config()
    specification = resolve_agpd_fit_specification(
        config,
        model_name="CO_BF_ER_LH",
        all_materials=True,
        parameterization=parameterization,
        error_structure="shared",
        prior_material="Ag10Pd90",
    )
    specs = all_parameter_specs(specification, config)
    metadata = resolved_parameterization_metadata(specification, config)

    assert specification.error_structure == "shared"
    assert metadata["name"] == parameterization
    assert {f"{name}_xAg_slope" for name in metadata["slopes"]} == expected_slopes
    assert expected_slopes.issubset(specs)
    assert {"sigma_rate_abs", "sigma_rate_rel"}.issubset(specs)
