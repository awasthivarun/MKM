import numpy as np
import pandas as pd
import pymc as pm

from mkm.inference.prior_predictive import (
    _flatten_draw_dimensions,
    _summarize_draws,
    sample_prior_predictive,
    summarize_prior_model_points,
    summarize_prior_linear_observable
)

from mkm.observable_maps import LinearObservableMap

def test_flatten_draw_dimensions():
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    flattened = _flatten_draw_dimensions(values)

    assert flattened.shape == (6, 4)


def test_summarize_draws_preserves_point_axis():
    values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    summary = _summarize_draws(values)

    np.testing.assert_allclose(summary["mean"], np.array([4.0, 5.0]))


def test_sample_prior_predictive_returns_prior():
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)

    class Built:
        pass

    built = Built()
    built.model = model

    result = sample_prior_predictive(built_model=built, draws=10, random_seed=123)

    assert hasattr(result, "prior") or "prior" in result


def test_model_point_summary_aligns_to_rows():
    with pm.Model(coords={"model_point": [0, 1]}) as model:
        x = pm.Normal("x", mu=0.0, sigma=1.0)
        pm.Deterministic("ln_rate_model", x + np.array([0.0, 1.0]), dims="model_point")

        prior = pm.sample_prior_predictive(draws=20, random_seed=123, return_inferencedata=True)

    model_points = pd.DataFrame({"model_point_id": [0, 1], "E_V_SHE": [0.0, 0.1]})
    summary = summarize_prior_model_points(prior_predictive=prior, model_points=model_points)

    assert len(summary) == 2
    assert "ln_rate_model_mean" in summary.columns

    difference = summary.loc[1, "ln_rate_model_mean"] - summary.loc[0, "ln_rate_model_mean"]

    np.testing.assert_allclose(difference, 1.0, rtol=1e-12, atol=1e-12)


def test_parameter_summary_includes_length_one_vector():
    with pm.Model(coords={"material": ["Ag10Pd90"]}):
        pm.LogNormal("sigma_ln_rate_material", mu=np.log(0.2), sigma=0.75, dims="material")
        prior = pm.sample_prior_predictive(draws=20, random_seed=123, return_inferencedata=True)

    from mkm.inference.prior_predictive import summarize_prior_parameters

    summary = summarize_prior_parameters(prior)

    assert "sigma_ln_rate_material" in set(summary["parameter"])


def test_prior_linear_observable_summary_preserves_metadata():

    with pm.Model(coords={"model_point": [0, 1, 2]}):
        offset = pm.Normal("offset", mu=0.0, sigma=1.0)
        pm.Deterministic("ln_rate_model", offset + np.array([0.0, 1.0, 2.0]), dims="model_point")
        prior = pm.sample_prior_predictive(draws=20, random_seed=123, return_inferencedata=True)

    observable_map = LinearObservableMap(
        outputs=pd.DataFrame({"observable_id": [0], "E_V_SHE": [0.1]}),
        terms=pd.DataFrame(
            {
                "observable_id": [0, 0],
                "model_point_id": [0, 2],
                "coefficient": [-0.5, 0.5],
            }
        ),
    )

    summary = summarize_prior_linear_observable(prior, observable_map)

    assert len(summary) == 1
    assert summary.loc[0, "E_V_SHE"] == 0.1
    np.testing.assert_allclose(summary.loc[0, "mean"], 1.0, atol=1e-12)
    np.testing.assert_allclose(summary.loc[0, "sd"], 0.0, atol=1e-12)