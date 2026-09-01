from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.postprocessing.calibration import compute_normal_loo_pit
from mkm.postprocessing.materials import (
    summarize_observation_diagnostics_by_material,
    summarize_residual_structure_by_material,
)
from mkm.postprocessing.predictions import (
    build_observation_diagnostics,
    build_observation_distribution_draws,
)
from mkm.postprocessing.residuals import (
    summarize_residual_curves,
    summarize_shared_replicate_residuals,
)


def _model_data_and_inputs():
    records = []
    for material, koh, co, base in (
        ("M1", 0.25, 0.01, 1.0),
        ("M2", 1.00, 0.10, 2.0),
    ):
        for replicate, offset in (("A", -0.05), ("B", 0.05)):
            for grid_index, potential in enumerate((0.20, 0.30)):
                rate = base + 0.2 * grid_index + offset
                records.append(
                    {
                        "material": material,
                        "C_KOH_M": koh,
                        "CO_mole_fraction": co,
                        "replicate": replicate,
                        "analysis_grid_index": grid_index,
                        "E_V_SHE": potential,
                        "rate_s_inv": rate,
                        "ln_rate": np.log(rate),
                    }
                )
    model_data = build_model_data(
        pd.DataFrame(records),
        electrolyte_concentration_column="C_KOH_M",
    )
    return model_data, build_model_input_arrays(model_data)


def _inference_data(model_data):
    point_rate = np.array([1.0, 1.2, 2.0, 2.2])
    ln_rate = np.broadcast_to(
        np.log(point_rate),
        (2, 3, len(point_rate)),
    ).copy()
    sigma_abs = np.empty((2, 3, 2))
    sigma_abs[..., 0] = 0.10
    sigma_abs[..., 1] = 0.20
    sigma_rel = np.empty((2, 3, 2))
    sigma_rel[..., 0] = 0.10
    sigma_rel[..., 1] = 0.20

    posterior = xr.Dataset(
        {
            "ln_rate_model": (("chain", "draw", "model_point"), ln_rate),
            "sigma_rate_abs": (("chain", "draw", "material"), sigma_abs),
            "sigma_rate_rel": (("chain", "draw", "material"), sigma_rel),
        },
        coords={
            "chain": [0, 1],
            "draw": [0, 1, 2],
            "model_point": model_data.model_points["model_point_id"],
            "material": ["M1", "M2"],
        },
    )
    return SimpleNamespace(posterior=posterior)


def test_observation_distribution_uses_model_rate_and_exact_sigma():
    model_data, inputs = _model_data_and_inputs()
    draws = build_observation_distribution_draws(
        _inference_data(model_data),
        inputs,
        sample_predictive=False,
    )

    expected_model_rate = np.array([1.0, 1.0, 1.2, 1.2, 2.0, 2.0, 2.2, 2.2])
    expected_sigma = np.array([0.2, 0.2, 0.22, 0.22, 0.6, 0.6, 0.64, 0.64])
    np.testing.assert_allclose(draws.model_rate[0, 0], expected_model_rate)
    np.testing.assert_allclose(draws.sigma_rate[0, 0], expected_sigma)
    assert draws.predictive_rate is None


def test_observation_diagnostics_have_one_residual_definition_only():
    model_data, inputs = _model_data_and_inputs()
    result = build_observation_diagnostics(
        _inference_data(model_data),
        model_data,
        inputs,
        random_seed=7,
    )

    assert {
        "rate_model_median",
        "sigma_rate_median",
        "rate_predictive_median",
        "residual",
        "standardized_residual",
        "observed_inside_predictive_95_hdi",
    }.issubset(result.columns)
    assert not any("conditional" in column for column in result.columns)
    np.testing.assert_allclose(
        result["residual"],
        result["rate"] - result["rate_model_median"],
    )


def test_residual_and_material_summaries_use_consolidated_columns():
    model_data, inputs = _model_data_and_inputs()
    observations = build_observation_diagnostics(
        _inference_data(model_data),
        model_data,
        inputs,
        random_seed=8,
    )

    curves = summarize_residual_curves(observations)
    shared = summarize_shared_replicate_residuals(observations)
    observation_by_material = summarize_observation_diagnostics_by_material(observations)
    residual_by_material = summarize_residual_structure_by_material(curves, shared)

    assert set(curves["material"]) == {"M1", "M2"}
    assert set(shared["material"]) == {"M1", "M2"}
    assert set(observation_by_material["material"]) == {"M1", "M2"}
    assert set(residual_by_material["material"]) == {"M1", "M2"}


def test_normal_loo_pit_is_computed_in_linear_rate_space():
    model_data, inputs = _model_data_and_inputs()
    inference_data = _inference_data(model_data)
    log_weights = xr.DataArray(
        np.zeros((2, 3, len(model_data.observations))),
        dims=("chain", "draw", "observation"),
    )
    loo_result = SimpleNamespace(log_weights=log_weights)

    result = compute_normal_loo_pit(
        inference_data,
        loo_result,
        model_data.observations,
        inputs,
    )

    assert result.pointwise["loo_pit"].between(0.0, 1.0).all()
    assert result.summary.loc[0, "n_points"] == len(model_data.observations)
