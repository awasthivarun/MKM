from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
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



def test_residual_curve_summary_preserves_lag1_and_other_statistics():
    rows = []
    for replicate, residuals, standardized in (
        ("A", [1.0, 2.0, 3.0, 2.5], [-1.0, 0.5, 1.0, -2.0]),
        ("B", [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]),
    ):
        for potential, residual, scaled in zip((0.0, 0.1, 0.2, 0.3), residuals, standardized):
            rows.append({
                "material": "M1", "electrolyte_concentration_M": 0.5,
                "CO_mole_fraction": 0.1, "replicate": replicate, "E_V_SHE": potential,
                "residual": residual, "standardized_residual": scaled,
            })

    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=3)
    actual = summarize_residual_curves(frame).set_index("replicate")
    for replicate, (residual, standardized) in {
        "A": (np.array([1.0, 2.0, 3.0, 2.5]), np.array([-1.0, 0.5, 1.0, -2.0])),
        "B": (np.ones(4), np.full(4, 0.5)),
    }.items():
        row = actual.loc[replicate]
        expected_lag1 = np.corrcoef(residual[:-1], residual[1:])[0, 1] if replicate == "A" else np.nan
        expected_standardized_lag1 = (
            np.corrcoef(standardized[:-1], standardized[1:])[0, 1] if replicate == "A" else np.nan
        )
        np.testing.assert_allclose(row["lag1_residual_correlation"], expected_lag1)
        np.testing.assert_allclose(row["standardized_lag1_residual_correlation"], expected_standardized_lag1)
        np.testing.assert_allclose(row["residual_slope_per_V"], np.polyfit([0.0, 0.1, 0.2, 0.3], residual, 1)[0])
        np.testing.assert_allclose(row["mean_residual"], residual.mean())
        np.testing.assert_allclose(row["rms_residual"], np.sqrt(np.mean(residual**2)))


def test_residual_curve_summary_without_standardized_or_sufficient_points():
    rows = []
    for replicate, residuals in (("A", [1.0]), ("B", [1.0, 2.0])):
        for potential, residual in zip((0.0, 0.1), residuals):
            rows.append({
                "material": "M1", "electrolyte_concentration_M": 0.5,
                "CO_mole_fraction": 0.1, "replicate": replicate,
                "E_V_SHE": potential, "residual": residual,
            })
    actual = summarize_residual_curves(pd.DataFrame(rows)).set_index("replicate")
    assert actual["standardized_lag1_residual_correlation"].isna().all()
    assert actual["lag1_residual_correlation"].isna().all()
    assert np.isnan(actual.loc["A", "residual_slope_per_V"])
    np.testing.assert_allclose(actual.loc["B", "residual_slope_per_V"], 10.0)

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


def test_residual_material_summary_preserves_order_and_missing_material_products():
    curves = pd.DataFrame(
        {
            "material": ["M2", "M1", "M2", "M4"],
            "lag1_residual_correlation": [0.4, -0.1, 0.8, np.nan],
            "residual_slope_per_V": [-2.0, 3.0, 4.0, 5.0],
        }
    )
    shared = pd.DataFrame(
        {
            "material": ["M3", "M1", "M3"],
            "shared_fraction_squared_residual": [0.1, 0.6, 0.3],
        }
    )
    result = summarize_residual_structure_by_material(curves, shared)
    assert result["material"].tolist() == ["M2", "M1", "M4", "M3"]
    np.testing.assert_allclose(result["median_lag1_residual_correlation"].iloc[:3], [0.6, -0.1, np.nan])
    np.testing.assert_allclose(result["median_abs_residual_slope_per_V"].iloc[:3], [3.0, 3.0, 5.0])
    np.testing.assert_allclose(
        result["median_shared_squared_residual_fraction"], [np.nan, 0.6, np.nan, 0.2]
    )


def test_residual_material_summary_handles_empty_and_missing_values():
    curves = pd.DataFrame(
        {
            "material": ["M1", "M1", np.nan],
            "lag1_residual_correlation": [np.nan, 0.5, 0.2],
            "residual_slope_per_V": [1.0, -3.0, 10.0],
        }
    )
    empty_shared = pd.DataFrame(columns=["material", "shared_fraction_squared_residual"])
    actual = summarize_residual_structure_by_material(curves, empty_shared)
    assert actual["material"].iloc[0] == "M1"
    assert pd.isna(actual["material"].iloc[1])
    np.testing.assert_allclose(actual["median_lag1_residual_correlation"], [0.5, np.nan])
    np.testing.assert_allclose(actual["median_abs_residual_slope_per_V"], [2.0, np.nan])
    assert "median_shared_squared_residual_fraction" not in actual
    assert summarize_residual_structure_by_material(pd.DataFrame(), pd.DataFrame()).empty


def test_residual_material_summary_uses_shared_only_when_curves_missing():
    shared = pd.DataFrame(
        {
            "material": ["M1", "M2", "M1"],
            "shared_fraction_squared_residual": [0.25, 0.75, 0.55],
        }
    )
    actual = summarize_residual_structure_by_material(pd.DataFrame(columns=["material"]), shared)
    assert actual["material"].tolist() == ["M1", "M2"]
    np.testing.assert_allclose(actual["median_shared_squared_residual_fraction"], [0.4, 0.75])
    assert "median_lag1_residual_correlation" not in actual


def test_residual_material_summary_preserves_missing_material_column_error():
    shared = pd.DataFrame(
        {"material": ["M1"], "shared_fraction_squared_residual": [0.4]}
    )
    with pytest.raises(KeyError, match="material"):
        summarize_residual_structure_by_material(pd.DataFrame(), shared)
