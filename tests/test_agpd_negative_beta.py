from copy import deepcopy

import numpy as np
import pytest
import pytensor.tensor as pt
import yaml

from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import load_agpd_model_config
from mkm.models.agpd_basic import (
    _bounded_linear_max_abs_slope,
    available_agpd_all_material_models,
    get_agpd_all_material_parameter_specs,
    get_agpd_parameterization,
    get_agpd_prior_profile,
)


def _config():
    return load_agpd_model_config(ProjectPaths.discover(__file__))


def test_negative_beta_models_are_available_for_all_material_fitting():
    models = set(available_agpd_all_material_models())
    assert {"CO_BF_ER_LH_neg", "CO_BF_ER_LH_Ag10_no_BF_neg"}.issubset(models)


@pytest.mark.parametrize(
    "model_name",
    ("CO_BF_ER_LH_neg", "CO_BF_ER_LH_Ag10_no_BF_neg"),
)
def test_negative_beta_models_change_only_beta_bf_base_prior(model_name):
    config = _config()
    base = get_agpd_prior_profile(config, "Ag10Pd90", "CO_BF_ER_LH")
    neg = get_agpd_prior_profile(config, "Ag10Pd90", model_name)

    assert neg["parameters"]["beta_2_BF"] == {
        "distribution": "truncated_normal",
        "mu": 0,
        "sigma": 1,
        "lower": -1,
        "upper": 1,
    }
    for parameter_name, specification in base["parameters"].items():
        if parameter_name != "beta_2_BF":
            assert neg["parameters"][parameter_name] == specification


@pytest.mark.parametrize(
    "model_name",
    ("CO_BF_ER_LH_neg", "CO_BF_ER_LH_Ag10_no_BF_neg"),
)
def test_negative_beta_models_reuse_parent_linear_xag_slope_priors(model_name):
    config = _config()
    base_x_reference, base_slopes = get_agpd_parameterization(config, "CO_BF_ER_LH", "linear_xAg")
    neg_x_reference, neg_slopes = get_agpd_parameterization(config, model_name, "linear_xAg")

    assert neg_x_reference == pytest.approx(base_x_reference)
    assert neg_slopes == base_slopes


def test_bounded_linear_slope_scale_generalizes_unit_interval_without_changing_it():
    config = _config()
    base_specs = get_agpd_all_material_parameter_specs(
        config,
        prior_material="Ag10Pd90",
        model_name="CO_BF_ER_LH",
        parameterization="linear_xAg",
    )
    beta = pt.scalar("beta")
    scale = _bounded_linear_max_abs_slope("beta_2_BF", beta, base_specs, 0.5)

    assert float(scale.eval({beta: 0.2})) == pytest.approx(0.4)
    assert float(scale.eval({beta: 0.5})) == pytest.approx(1.0)
    assert float(scale.eval({beta: 0.8})) == pytest.approx(0.4)


def test_bounded_linear_slope_scale_uses_negative_beta_prior_bounds():
    config = _config()
    neg_specs = get_agpd_all_material_parameter_specs(
        config,
        prior_material="Ag10Pd90",
        model_name="CO_BF_ER_LH_neg",
        parameterization="linear_xAg",
    )
    beta = pt.scalar("beta")
    scale = _bounded_linear_max_abs_slope("beta_2_BF", beta, neg_specs, 0.5)

    assert neg_specs["beta_2_BF"]["lower"] == -1
    assert neg_specs["beta_2_BF"]["upper"] == 1
    assert float(scale.eval({beta: 0.0})) == pytest.approx(2.0)
    assert float(scale.eval({beta: -0.5})) == pytest.approx(1.0)
    assert float(scale.eval({beta: 0.5})) == pytest.approx(1.0)


def test_bounded_linear_slope_scale_rejects_invalid_bounds():
    specs = {"beta_2_BF": {"lower": 1.0, "upper": -1.0}}
    with pytest.raises(ValueError, match="invalid prior bounds"):
        _bounded_linear_max_abs_slope("beta_2_BF", pt.scalar(), specs, 0.5)
