from dataclasses import fields

import pytest

from mkm.models.ptru_registry import get_ptru_model_definition
from mkm.project_paths import ProjectPaths
from mkm.workflows.pt100_fit import (
    load_pt100_config,
    pt100_likelihood_kwargs,
    pt100_parameter_specs,
    resolve_pt100_fit_specification,
    resolve_pt100_parameter_specs,
)


@pytest.fixture
def configured_fit():
    paths = ProjectPaths.discover(__file__)
    config, prior_config = load_pt100_config(paths)
    return config, prior_config


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model_name", ["CO_LH", "CO_ER", "CO_ER_LH"])
def test_pt100_fit_specification_matches_registered_model(configured_fit, medium, model_name):
    config, prior_config = configured_fit
    specification = resolve_pt100_fit_specification(config, medium, model_name)

    assert specification.medium == medium
    assert specification.model_name == model_name
    assert specification.material == "Pt100"
    assert specification.fit_scope == "individual"
    assert specification.error_structure == "shared"
    assert specification.parameterization is None
    assert not specification.is_all_materials

    definition = get_ptru_model_definition(model_name)
    physical = resolve_pt100_parameter_specs(prior_config, medium, model_name)
    expected_names = {field.name for field in fields(definition.parameter_class)}

    assert set(physical) == expected_names

    all_specs = pt100_parameter_specs(prior_config, specification, config)
    assert set(all_specs) == expected_names | {"sigma_rate_abs", "sigma_rate_rel"}


def test_pt100_uses_existing_linear_rate_normal_likelihood(configured_fit):
    config, _ = configured_fit
    kwargs = pt100_likelihood_kwargs(config)

    assert set(kwargs) == {
        "sigma_abs_prior_median_s_inv",
        "sigma_abs_prior_log_sd",
        "sigma_rel_prior_median",
        "sigma_rel_prior_log_sd",
    }
    assert all(value > 0 for value in kwargs.values())


@pytest.mark.parametrize(
    "medium,model_name",
    [("neutral", "CO_LH"), ("acidic", "CO_BF")],
)
def test_pt100_rejects_unsupported_fit(configured_fit, medium, model_name):
    config, _ = configured_fit

    with pytest.raises(ValueError):
        resolve_pt100_fit_specification(config, medium, model_name)


def test_pt100_rejects_unimplemented_all_material_fitting(configured_fit):
    config, _ = configured_fit

    with pytest.raises(ValueError, match="individual-material"):
        resolve_pt100_fit_specification(
            config, "acidic", "CO_LH", fit_scope="all_materials",
        )


def test_pt100_rejects_unconfigured_error_structure(configured_fit):
    config, _ = configured_fit

    with pytest.raises(ValueError, match="not enabled"):
        resolve_pt100_fit_specification(
            config, "basic", "CO_LH", error_structure="material",
        )