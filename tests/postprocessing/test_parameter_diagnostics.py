from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from mkm.postprocessing.composition_parameters import (
    build_agpd_composition_parameter_trends,
)
from mkm.postprocessing.diagnostics import (
    ERROR_VARIABLES,
    build_posterior_parameter_summary,
    prior_statistics,
)
from mkm.postprocessing.sampling import (
    build_sampling_datatree,
    build_sampling_health,
    sampling_parameter_names,
)


def _posterior():
    return xr.Dataset(
        {
            "x": (
                ("chain", "draw"),
                np.array([[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]]),
            ),
            "sigma_rate_abs": (
                ("chain", "draw", "material"),
                np.array(
                    [
                        [[0.10, 0.20], [0.11, 0.21], [0.12, 0.22], [0.13, 0.23]],
                        [[0.09, 0.19], [0.10, 0.20], [0.11, 0.21], [0.12, 0.22]],
                    ]
                ),
            ),
            "sigma_rate_rel": (
                ("chain", "draw", "material"),
                np.full((2, 4, 2), 0.18),
            ),
        },
        coords={"material": ["Ag10Pd90", "Pd100"]},
    )


def _specs():
    return {
        "x": {"distribution": "normal", "mu": 0.0, "sigma": 1.0},
        "sigma_rate_abs": {
            "distribution": "lognormal",
            "median": 0.0002,
            "log_sd": 1.0,
        },
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": 0.18,
            "log_sd": 0.75,
        },
    }


def test_combined_parameter_table_contains_estimates_diagnostics_and_named_errors():
    inference_data = xr.DataTree.from_dict({"/posterior": _posterior()})
    result = build_posterior_parameter_summary(inference_data, _specs())

    assert {
        "parameter",
        "variable",
        "parameter_type",
        "mean",
        "sd",
        "median",
        "ess_bulk",
        "ess_tail",
        "rhat",
        "prior_sd",
    }.issubset(result.columns)
    assert set(result.loc[result["parameter_type"] == "error", "variable"]) == set(
        ERROR_VARIABLES
    )
    parameter_labels = " ".join(result["parameter"].astype(str))
    assert "Ag10Pd90" in parameter_labels
    assert "Pd100" in parameter_labels


def test_sampling_expands_material_errors_for_trace_and_excludes_them_from_pairs():
    inference_data = SimpleNamespace(posterior=_posterior())
    data = build_sampling_datatree(
        inference_data,
        ("x", "sigma_rate_abs", "sigma_rate_rel"),
    )

    assert set(data.posterior.data_vars) == {
        "x",
        "sigma_rate_abs[material=Ag10Pd90]",
        "sigma_rate_abs[material=Pd100]",
        "sigma_rate_rel[material=Ag10Pd90]",
        "sigma_rate_rel[material=Pd100]",
    }
    assert sampling_parameter_names(_posterior(), _specs()) == ("x",)


def test_sampling_health_extracts_bfmi_from_datatree(monkeypatch):
    posterior = xr.Dataset(
        {
            "x": (
                ("chain", "draw"),
                np.array([[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]]),
            )
        }
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (
                ("chain", "draw"),
                np.zeros((2, 4), dtype=bool),
            ),
            "energy": (
                ("chain", "draw"),
                np.array([[1.0, 1.1, 0.9, 1.2], [1.2, 1.0, 1.1, 0.8]]),
            ),
        }
    )
    inference_data = xr.DataTree.from_dict(
        {
            "/posterior": posterior,
            "/sample_stats": sample_stats,
        }
    )

    monkeypatch.setattr(
        "mkm.postprocessing.sampling.azs.summary",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "ess_bulk": [500.0],
                "ess_tail": [450.0],
                "r_hat": [1.001],
            },
            index=["x"],
        ),
    )
    monkeypatch.setattr(
        "mkm.postprocessing.sampling.azs.bfmi",
        lambda *args, **kwargs: xr.DataTree.from_dict(
            {"/": xr.Dataset({"energy": ("chain", [0.72, 0.81])})}
        ),
    )

    health = build_sampling_health(inference_data, parameter_names=("x",))

    assert health["n_divergent"] == 0
    assert health["min_bfmi"] == 0.72
    assert health["max_rhat"] == 1.001
    assert health["min_ess_bulk"] == 500.0
    assert health["min_ess_tail"] == 450.0


def test_lognormal_prior_statistics_use_median_parameterization():
    result = prior_statistics(
        {"distribution": "lognormal", "median": 0.18, "log_sd": 0.75}
    )
    assert result["prior_q50"] == 0.18
    assert result["prior_mean"] > result["prior_q50"]
    assert result["lower"] == 0.0


def test_shared_all_material_parameter_trends_do_not_invent_slopes():
    with open("config/models/agpd_basic.yaml", "r") as file:
        config = yaml.safe_load(file)

    parameter_values = {
        "deltaG1_0": -0.4,
        "deltaG4_0": 0.0,
        "deltaG5_0": 0.0,
        "beta_2": 0.5,
        "q": 0.5,
        "Gact2_BF_0": 0.7,
        "Gact2_LH_0": 0.7,
    }
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.full((2, 3), value))
            for name, value in parameter_values.items()
        }
    )
    trends = build_agpd_composition_parameter_trends(
        SimpleNamespace(posterior=posterior),
        config,
        model_name="BF_LH",
        parameterization="shared",
        n_grid=11,
    )

    assert not trends["x_dependent"].any()
    for _, frame in trends.groupby("parameter"):
        assert frame["median"].nunique() == 1
