from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from mkm.postprocessing.composition_parameters import (
    build_agpd_composition_parameter_trends,
    build_agpd_material_noise_summary,
)


CONFIG = {
    "surface_composition": {
        "Pd100": {"Ag_fraction": 0.0, "Pd_fraction": 1.0},
        "Ag50Pd50": {"Ag_fraction": 0.5, "Pd_fraction": 0.5},
        "Ag90Pd10": {"Ag_fraction": 0.9, "Pd_fraction": 0.1},
    },
    "composition_parameterizations": {
        "linear_xAg": {
            "x_reference": 0.5,
            "models": {
                "CO_BF_ER_LH": {
                    "slopes": {
                        "deltaG1_0": {"distribution": "normal", "mu": 0.0, "sigma": 0.1},
                    }
                }
            },
        }
    },
}


def _idata():
    scalar = {
        "deltaG1_0": [1.0, 1.0],
        "deltaG4_0": [0.1, 0.1],
        "deltaG5_0": [0.0, 0.0],
        "beta_2_BF": [0.01, 0.01],
        "beta_2_ER": [0.3, 0.3],
        "q": [0.4, 0.4],
        "Gact1_0": [0.6, 0.6],
        "Gact2_BF_0": [0.7, 0.7],
        "Gact2_ER_0": [0.75, 0.75],
        "Gact2_LH_0": [0.72, 0.72],
        "deltaG1_0_xAg_slope": [2.0, 2.0],
    }
    data_vars = {
        name: (("chain", "draw"), np.asarray(values, dtype=float).reshape(1, 2))
        for name, values in scalar.items()
    }
    data_vars["sigma_ln_rate_material"] = (
        ("chain", "draw", "material"),
        np.array([[[0.3, 0.4, 0.5], [0.3, 0.4, 0.5]]]),
    )
    posterior = xr.Dataset(
        data_vars,
        coords={
            "chain": [0],
            "draw": [0, 1],
            "material": ["Pd100", "Ag50Pd50", "Ag90Pd10"],
        },
    )
    return SimpleNamespace(posterior=posterior)


def test_linear_xag_parameter_trend_uses_reference_plus_slope():
    trends = build_agpd_composition_parameter_trends(
        inference_data=_idata(),
        config=CONFIG,
        model_name="CO_BF_ER_LH",
        composition_model="linear_xAg",
        n_grid=3,
    )

    delta_g1 = trends.loc[trends["parameter"] == "deltaG1_0"].sort_values("xAg")
    medians = dict(zip(delta_g1["xAg"], delta_g1["median"]))

    assert medians[0.0] == pytest.approx(0.0)
    assert medians[0.5] == pytest.approx(1.0)
    assert medians[0.9] == pytest.approx(1.8)
    assert delta_g1["x_dependent"].all()


def test_shared_parameter_is_flat_across_composition():
    trends = build_agpd_composition_parameter_trends(
        inference_data=_idata(),
        config=CONFIG,
        model_name="CO_BF_ER_LH",
        composition_model="linear_xAg",
        n_grid=5,
    )

    beta = trends.loc[trends["parameter"] == "beta_2_ER"]
    assert beta["median"].tolist() == pytest.approx([0.3] * len(beta))
    assert not beta["x_dependent"].any()


def test_material_noise_summary_preserves_composition_mapping():
    noise = build_agpd_material_noise_summary(_idata(), CONFIG)

    assert noise["material"].tolist() == ["Pd100", "Ag50Pd50", "Ag90Pd10"]
    assert noise["xAg"].tolist() == pytest.approx([0.0, 0.5, 0.9])
    assert noise["median"].tolist() == pytest.approx([0.3, 0.4, 0.5])
