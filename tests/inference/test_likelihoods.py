import numpy as np
import pandas as pd
import pymc as pm
import pytest

from mkm.inference.likelihoods import (
    add_log_rate_likelihood,
    add_material_log_rate_likelihood,
    add_rate_normal_likelihood,
    get_observation_curve_structure,
    get_observation_material_index,
)
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_coords, build_model_input_arrays


def _build_test_inputs(setup_group_columns=None, setup_zero_sum_columns=None):
    records = []

    for replicate, offset in zip(["A", "B", "C"], [-0.1, 0.0, 0.1]):
        for grid_index in [0, 1]:
            ln_rate = 1.0 + 0.2 * grid_index + offset

            records.append(
                {
                    "material": "M1",
                    "C_KOH_M": 0.25,
                    "CO_mole_fraction": 0.01,
                    "replicate": replicate,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": 0.01 * grid_index,
                    "rate_s_inv": np.exp(ln_rate),
                    "ln_rate": ln_rate,
                }
            )

    for replicate, offset in zip(["A", "B", "C"], [-0.2, 0.0, 0.2]):
        ln_rate = 2.0 + offset

        records.append(
            {
                "material": "M2",
                "C_KOH_M": 1.0,
                "CO_mole_fraction": 0.10,
                "replicate": replicate,
                "analysis_grid_index": 2,
                "E_V_SHE": 0.02,
                "rate_s_inv": np.exp(ln_rate),
                "ln_rate": ln_rate,
            }
        )

    data = pd.DataFrame(records)

    model_data = build_model_data(selected_replicates=data, electrolyte_concentration_column="C_KOH_M")

    return build_model_input_arrays(
        model_data,
        setup_group_columns=setup_group_columns,
        setup_zero_sum_columns=setup_zero_sum_columns,
    )


def test_observation_material_index():
    inputs = _build_test_inputs()

    material_index = get_observation_material_index(inputs)

    assert len(material_index) == len(inputs.observation_ln_rate)
    assert set(material_index) == {0, 1}


def test_likelihood_prediction_mapping():
    inputs = _build_test_inputs()

    coords = build_model_coords(inputs)
    ln_rate_model_values = np.array([1.0, 1.2, 2.0])

    with pm.Model(coords=coords) as model:
        ln_rate_model = pm.Data("ln_rate_model", ln_rate_model_values, dims="model_point")
        likelihood = add_log_rate_likelihood(ln_rate_model=ln_rate_model, inputs=inputs)

    assert likelihood.mu_observation.ndim == 1
    assert likelihood.sigma_observation.ndim == 1
    assert "ln_rate_observed" in model.named_vars


def test_likelihood_initial_logp_is_finite():
    inputs = _build_test_inputs()

    coords = build_model_coords(inputs)
    ln_rate_model_values = np.array([1.0, 1.2, 2.0])

    with pm.Model(coords=coords) as model:
        ln_rate_model = pm.Data("ln_rate_model", ln_rate_model_values, dims="model_point")
        add_log_rate_likelihood(ln_rate_model=ln_rate_model, inputs=inputs)

    initial_point = model.initial_point()
    logp = model.compile_logp()(initial_point)

    assert np.isfinite(logp)


def test_material_sigma_has_one_value_per_material():
    inputs = _build_test_inputs()

    coords = build_model_coords(inputs)

    with pm.Model(coords=coords) as model:
        ln_rate_model = pm.Data("ln_rate_model", np.array([1.0, 1.2, 2.0]), dims="model_point")
        add_log_rate_likelihood(ln_rate_model=ln_rate_model, inputs=inputs)
        prior = pm.sample_prior_predictive(draws=20, random_seed=123)

    assert prior.prior["sigma_ln_rate_material"].sizes["material"] == 2


def test_setup_intercept_likelihood_structure():
    inputs = _build_test_inputs(
        setup_group_columns=["material", "electrolyte_concentration_M", "replicate"],
        setup_zero_sum_columns=["material", "electrolyte_concentration_M"],
    )

    coords = build_model_coords(inputs)
    ln_rate_model_values = np.array([1.0, 1.2, 2.0])

    with pm.Model(coords=coords) as model:
        ln_rate_model = pm.Data("ln_rate_model", ln_rate_model_values, dims="model_point")

        likelihood = add_log_rate_likelihood(
            ln_rate_model=ln_rate_model,
            inputs=inputs,
            setup_intercept=True,
        )

        prior = pm.sample_prior_predictive(draws=20, random_seed=123)

    assert likelihood.setup_sigma_material is not None
    assert likelihood.setup_z is not None
    assert likelihood.setup_offset is not None

    assert prior.prior["sigma_ln_rate_setup_material"].sizes["material"] == 2
    assert prior.prior["z_ln_rate_setup"].sizes["setup"] == 6
    assert prior.prior["ln_rate_setup_offset"].sizes["setup"] == 6


def test_setup_intercept_requires_setup_indexed_inputs():
    inputs = _build_test_inputs()

    coords = build_model_coords(inputs)

    with pm.Model(coords=coords):
        ln_rate_model = pm.Data(
            "ln_rate_model",
            np.array([1.0, 1.2, 2.0]),
            dims="model_point",
        )

        with pytest.raises(ValueError, match="setup-indexed"):
            add_log_rate_likelihood(
                ln_rate_model=ln_rate_model,
                inputs=inputs,
                setup_intercept=True,
            )


def test_setup_offsets_sum_to_zero_within_experiment():
    inputs = _build_test_inputs(
        setup_group_columns=["material", "electrolyte_concentration_M", "replicate"],
        setup_zero_sum_columns=["material", "electrolyte_concentration_M"],
    )

    coords = build_model_coords(inputs)

    with pm.Model(coords=coords):
        ln_rate_model = pm.Data(
            "ln_rate_model",
            np.array([1.0, 1.2, 2.0]),
            dims="model_point",
        )

        add_log_rate_likelihood(
            ln_rate_model=ln_rate_model,
            inputs=inputs,
            setup_intercept=True,
        )

        prior = pm.sample_prior_predictive(
            draws=100,
            random_seed=123,
        )

    offsets = prior.prior["ln_rate_setup_offset"].values

    for experiment_id in range(len(inputs.setup_experiment_size)):
        setup_indices = np.flatnonzero(
            inputs.setup_experiment_index == experiment_id
        )

        summed_offsets = offsets[..., setup_indices].sum(axis=-1)

        np.testing.assert_allclose(
            summed_offsets,
            0.0,
            rtol=0,
            atol=1e-12,
        )


def test_likelihood_backward_compatible_alias():
    assert add_material_log_rate_likelihood is add_log_rate_likelihood

def test_potential_curve_structure_groups_condition_and_replicate():
    inputs = _build_test_inputs()

    structure = get_observation_curve_structure(inputs)

    assert structure.n_curves == 6
    assert int(structure.is_transition.sum()) == 3
    np.testing.assert_allclose(
        structure.delta_E_V[structure.is_transition],
        0.01,
        rtol=0,
        atol=1e-12,
    )


def test_mvn_likelihood_has_material_correlation_lengths_and_finite_logp():
    inputs = _build_test_inputs()
    coords = build_model_coords(inputs)

    with pm.Model(coords=coords) as model:
        ln_rate_model = pm.Data(
            "ln_rate_model",
            np.array([1.0, 1.2, 2.0]),
            dims="model_point",
        )
        likelihood = add_log_rate_likelihood(
            ln_rate_model=ln_rate_model,
            inputs=inputs,
            correlated_potential=True,
            correlation_length_prior_median_V=0.02,
            correlation_length_prior_log_sd=1.0,
        )
        prior = pm.sample_prior_predictive(draws=20, random_seed=123)

    assert likelihood.correlation_length_material is not None
    assert likelihood.rho_observation is not None
    assert prior.prior["ell_E_V_material"].sizes["material"] == 2
    assert prior.prior["ln_rate_observation_rho"].sizes["observation"] == len(
        inputs.observation_ln_rate
    )

    initial_point = model.initial_point()
    logp = model.compile_logp()(initial_point)
    assert np.isfinite(logp)


def test_mvn_and_setup_intercept_are_mutually_exclusive():
    inputs = _build_test_inputs(
        setup_group_columns=["material", "electrolyte_concentration_M", "replicate"],
        setup_zero_sum_columns=["material", "electrolyte_concentration_M"],
    )
    coords = build_model_coords(inputs)

    with pm.Model(coords=coords):
        ln_rate_model = pm.Data(
            "ln_rate_model",
            np.array([1.0, 1.2, 2.0]),
            dims="model_point",
        )
        with pytest.raises(ValueError, match="cannot be enabled together"):
            add_log_rate_likelihood(
                ln_rate_model=ln_rate_model,
                inputs=inputs,
                setup_intercept=True,
                correlated_potential=True,
            )



def test_rate_normal_likelihood_has_two_global_error_terms_and_finite_logp():
    inputs = _build_test_inputs()
    coords = build_model_coords(inputs)
    rate_model_values = np.exp(np.array([1.0, 1.2, 2.0]))

    with pm.Model(coords=coords) as model:
        rate_model = pm.Data(
            "rate_model",
            rate_model_values,
            dims="model_point",
        )
        likelihood = add_rate_normal_likelihood(
            rate_model=rate_model,
            inputs=inputs,
            sigma_abs_prior_median_s_inv=2.0e-4,
            sigma_abs_prior_log_sd=1.0,
            sigma_rel_prior_median=0.18,
            sigma_rel_prior_log_sd=0.75,
        )
        prior = pm.sample_prior_predictive(draws=20, random_seed=123)

    assert likelihood.sigma_abs.ndim == 0
    assert likelihood.sigma_rel.ndim == 0
    assert "sigma_rate_abs" in prior.prior
    assert "sigma_rate_rel" in prior.prior
    assert "rate_observation_sigma" in prior.prior
    assert "rate_observed" in model.named_vars
    assert "material" not in prior.prior["sigma_rate_abs"].dims
    assert "material" not in prior.prior["sigma_rate_rel"].dims

    initial_point = model.initial_point()
    logp = model.compile_logp()(initial_point)
    assert np.isfinite(logp)


def test_rate_normal_observation_sigma_is_absolute_plus_relative_rate():
    inputs = _build_test_inputs()
    coords = build_model_coords(inputs)
    rate_model_values = np.exp(np.array([1.0, 1.2, 2.0]))

    with pm.Model(coords=coords):
        rate_model = pm.Data(
            "rate_model",
            rate_model_values,
            dims="model_point",
        )
        add_rate_normal_likelihood(
            rate_model=rate_model,
            inputs=inputs,
            sigma_abs_prior_median_s_inv=2.0e-4,
            sigma_abs_prior_log_sd=0.01,
            sigma_rel_prior_median=0.18,
            sigma_rel_prior_log_sd=0.01,
        )
        prior = pm.sample_prior_predictive(draws=5, random_seed=123)

    sigma_abs = prior.prior["sigma_rate_abs"].values[..., None]
    sigma_rel = prior.prior["sigma_rate_rel"].values[..., None]
    observation_model_point_index = inputs.observation_model_point_index
    expected = sigma_abs + sigma_rel * rate_model_values[observation_model_point_index]
    actual = prior.prior["rate_observation_sigma"].values

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
