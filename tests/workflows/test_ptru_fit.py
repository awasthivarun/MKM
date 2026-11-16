"""Pure Pt uses the same PtRu fit specification and prior registry as alloys."""

import pytest

from mkm.models.ptru import ptru_parameter_names, ptru_parameter_specs
from mkm.project_paths import ProjectPaths
from mkm.workflows.likelihood_configuration import error_parameter_specs, rate_normal_likelihood_kwargs
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification


@pytest.fixture
def config_and_priors():
    return load_ptru_config(ProjectPaths.discover(__file__))


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model", ["CO_LH", "CO_ER", "CO_ER_LH"])
def test_pt100_individual_fit_is_registered(config_and_priors, medium, model):
    config, priors = config_and_priors
    spec = resolve_ptru_fit_specification(config, medium, model, material="Pt100")
    assert spec.medium == medium and spec.model_name == model
    assert spec.material == "Pt100" and spec.materials == ("Pt100",)
    assert spec.fit_scope == "individual" and not spec.is_all_materials
    assert spec.parameterization is None and spec.error_structure == "shared"
    assert set(ptru_parameter_specs(priors, medium, model, "Pt100")) == set(ptru_parameter_names(model, "Pt100"))


def test_linear_rate_likelihood_is_unchanged(config_and_priors):
    config, _ = config_and_priors
    kwargs = rate_normal_likelihood_kwargs(config)
    assert len(kwargs) == 4 and all(value > 0 for value in kwargs.values())
    assert set(error_parameter_specs(config)) == {"sigma_rate_abs", "sigma_rate_rel"}


@pytest.mark.parametrize("medium,model", [("neutral", "CO_LH"), ("acidic", "CO_BF")])
def test_pt100_rejects_unsupported_models(config_and_priors, medium, model):
    config, _ = config_and_priors
    with pytest.raises(ValueError):
        resolve_ptru_fit_specification(config, medium, model, material="Pt100")


def test_pooled_fit_is_available_and_error_structure_configured(config_and_priors):
    config, _ = config_and_priors
    spec = resolve_ptru_fit_specification(
        config, "basic", "CO_ER_LH", all_materials=True, parameterization="shared", error_structure="material",
    )
    assert spec.is_all_materials and spec.fit_scope == "all_materials"
    assert spec.materials == ("Pt100", "Pt50Ru50", "Pt66Ru33")
    assert spec.error_structure == "material"
