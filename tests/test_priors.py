import numpy as np
import pymc as pm
import pytest

from mkm.inference.priors import (
    build_named_priors,
    build_prior,
)


def test_build_named_priors_supports_required_distributions():
    specs = {
        "normal_parameter": {
            "distribution": "normal",
            "mu": 0.0,
            "sigma": 0.2,
        },
        "uniform_parameter": {
            "distribution": "uniform",
            "lower": 0.0,
            "upper": 1.0,
        },
        "truncated_parameter": {
            "distribution": (
                "truncated_normal"
            ),
            "mu": 0.7,
            "sigma": 0.2,
            "lower": 0.0,
            "upper": 1.0,
        },
    }

    with pm.Model() as model:
        priors = build_named_priors(
            specs
        )

    assert set(priors) == set(
        specs
    )

    assert all(
        name in model.named_vars
        for name in specs
    )


def test_prior_model_has_finite_initial_logp():
    with pm.Model() as model:
        build_named_priors(
            {
                "x": {
                    "distribution": (
                        "truncated_normal"
                    ),
                    "mu": -0.4,
                    "sigma": 0.2,
                    "lower": -0.4,
                    "upper": 0.0,
                }
            }
        )

    logp = model.compile_logp()(
        model.initial_point()
    )

    assert np.isfinite(logp)


def test_prior_rejects_unknown_distribution():
    with pm.Model():
        with pytest.raises(
            ValueError,
            match="Unsupported",
        ):
            build_prior(
                "x",
                {
                    "distribution": "magic",
                },
            )


def test_prior_rejects_invalid_bounds():
    with pm.Model():
        with pytest.raises(
            ValueError,
            match="lower < upper",
        ):
            build_prior(
                "x",
                {
                    "distribution": "uniform",
                    "lower": 1.0,
                    "upper": 0.0,
                },
            )


