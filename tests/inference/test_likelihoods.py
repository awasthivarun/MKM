import numpy as np
import pymc as pm
import pytest

from mkm.inference.likelihoods import (
    RATE_NORMAL,
    add_likelihood,
    add_rate_normal_likelihood,
    available_error_structures,
    available_likelihoods,
    get_observation_material_index,
)
from mkm.model_inputs import ModelInputArrays, build_model_coords


def _inputs():
    return ModelInputArrays(
        materials=("M1", "M2"),
        condition_material_index=np.array([0, 1], dtype=np.int64),
        condition_ln_electrolyte_concentration=np.log([0.25, 1.0]),
        condition_ln_CO_mole_fraction=np.log([0.01, 0.10]),
        model_point_condition_index=np.array([0, 1], dtype=np.int64),
        model_point_E_V_SHE=np.array([0.20, 0.30]),
        observation_model_point_index=np.array([0, 0, 1], dtype=np.int64),
        observation_rate=np.array([1.0, 1.1, 2.0]),
    )


def _likelihood_kwargs():
    return {
        "sigma_abs_prior_median_s_inv": 2.0e-4,
        "sigma_abs_prior_log_sd": 1.0,
        "sigma_rel_prior_median": 0.18,
        "sigma_rel_prior_log_sd": 0.75,
    }


def test_only_rate_normal_likelihood_is_registered():
    assert available_likelihoods() == (RATE_NORMAL,)
    assert available_error_structures() == ("shared", "material")


def test_observation_material_index_follows_condition_and_point_mappings():
    np.testing.assert_array_equal(
        get_observation_material_index(_inputs()),
        np.array([0, 0, 1], dtype=np.int64),
    )


def test_material_error_structure_uses_exact_additive_sigma_formula():
    inputs = _inputs()
    rate_model_values = np.array([1.25, 2.50])

    with pm.Model(coords=build_model_coords(inputs)) as model:
        rate_model = pm.Data("rate_model", rate_model_values, dims="model_point")
        likelihood = add_rate_normal_likelihood(
            rate_model,
            inputs,
            error_structure="material",
            **_likelihood_kwargs(),
        )
        sigma_abs, sigma_rel, sigma_observation = pm.draw(
            [likelihood.sigma_abs, likelihood.sigma_rel, likelihood.sigma_observation],
            draws=5,
            random_seed=21,
        )

    material_index = np.array([0, 0, 1])
    expected = (
        sigma_abs[:, material_index]
        + sigma_rel[:, material_index] * rate_model_values[[0, 0, 1]]
    )
    np.testing.assert_allclose(sigma_observation, expected)
    assert model.named_vars_to_dims["sigma_rate_abs"] == ("material",)
    assert model.named_vars_to_dims["sigma_rate_rel"] == ("material",)
    assert model.named_vars_to_dims["rate_observed"] == ("observation",)


def test_shared_error_structure_uses_one_sigma_pair_for_all_materials():
    inputs = _inputs()
    rate_model_values = np.array([1.25, 2.50])

    with pm.Model(coords=build_model_coords(inputs)):
        rate_model = pm.Data("rate_model", rate_model_values, dims="model_point")
        likelihood = add_rate_normal_likelihood(
            rate_model,
            inputs,
            error_structure="shared",
            **_likelihood_kwargs(),
        )
        sigma_abs, sigma_rel, sigma_observation = pm.draw(
            [likelihood.sigma_abs, likelihood.sigma_rel, likelihood.sigma_observation],
            draws=5,
            random_seed=22,
        )

    expected = sigma_abs[:, None] + sigma_rel[:, None] * rate_model_values[[0, 0, 1]]
    np.testing.assert_allclose(sigma_observation, expected)


def test_rate_normal_initial_logp_is_finite():
    inputs = _inputs()
    with pm.Model(coords=build_model_coords(inputs)) as model:
        rate_model = pm.Data("rate_model", np.array([1.0, 2.0]), dims="model_point")
        add_likelihood(
            RATE_NORMAL,
            rate_model=rate_model,
            inputs=inputs,
            error_structure="material",
            likelihood_kwargs=_likelihood_kwargs(),
        )

    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_unknown_likelihood_and_error_structure_are_rejected():
    inputs = _inputs()
    with pm.Model(coords=build_model_coords(inputs)):
        rate_model = pm.Data("rate_model", np.array([1.0, 2.0]), dims="model_point")
        with pytest.raises(ValueError, match="Unknown likelihood"):
            add_likelihood("retired", rate_model=rate_model, inputs=inputs)
        with pytest.raises(ValueError, match="Unknown error structure"):
            add_rate_normal_likelihood(
                rate_model,
                inputs,
                error_structure="retired",
                **_likelihood_kwargs(),
            )
