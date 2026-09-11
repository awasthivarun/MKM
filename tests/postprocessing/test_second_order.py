from types import SimpleNamespace

import numpy as np
import pandas as pd

from mkm.constants import F_C_mol, R_J_mol_K
from mkm.observable_maps import (
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
    evaluate_linear_observable_map,
)
from mkm.postprocessing.second_order import build_second_order_outputs


def _make_model_points():
    records = []
    model_point_id = 0
    condition_id = 0
    for material in ("Ag50Pd50",):
        for concentration in (0.25, 0.50, 1.00):
            for co_fraction in (0.001, 0.01, 0.10, 1.00):
                for grid_index in range(7):
                    records.append(
                        {
                            "model_point_id": model_point_id,
                            "condition_id": condition_id,
                            "material": material,
                            "electrolyte_concentration_M": concentration,
                            "CO_mole_fraction": co_fraction,
                            "analysis_grid_index": grid_index,
                            "E_V_SHE": 0.01 * grid_index,
                        }
                    )
                    model_point_id += 1
                condition_id += 1
    return pd.DataFrame(records)


def _conditions(points):
    return (
        points[["condition_id", "material", "electrolyte_concentration_M", "CO_mole_fraction"]]
        .drop_duplicates()
        .sort_values("condition_id")
        .reset_index(drop=True)
    )


def _experimental_observable_points(points, log_rate, temperature_K):
    alpha_map = build_alpha_map(points, temperature_K)
    oh_map = build_log_slope_order_map(
        points,
        "electrolyte_concentration_M",
        [0.25, 0.50, 1.00],
        ["material", "CO_mole_fraction"],
    )
    co_map = build_adjacent_log_order_map(
        points,
        "CO_mole_fraction",
        [0.001, 0.01, 0.10, 1.00],
        ["material", "electrolyte_concentration_M"],
        "lower_CO_mole_fraction",
        "upper_CO_mole_fraction",
    )

    condition_metadata = _conditions(points).rename(columns={"electrolyte_concentration_M": "C_KOH_M"})
    alpha = alpha_map.outputs.merge(condition_metadata, on="condition_id", validate="many_to_one")
    alpha["alpha_mean"] = evaluate_linear_observable_map(log_rate, alpha_map)
    alpha.insert(0, "observable", "alpha")

    oh = oh_map.outputs.copy()
    oh["delta_OH"] = evaluate_linear_observable_map(log_rate, oh_map)
    oh.insert(0, "observable", "delta_OH")

    co = co_map.outputs.rename(
        columns={
            "electrolyte_concentration_M": "C_KOH_M",
            "lower_CO_mole_fraction": "CO_lower_mole_fraction",
            "upper_CO_mole_fraction": "CO_upper_mole_fraction",
        }
    )
    co["delta_CO"] = evaluate_linear_observable_map(log_rate, co_map)
    co.insert(0, "observable", "delta_CO")
    return pd.concat([alpha, oh, co], ignore_index=True, sort=False)


def test_second_order_outputs_recover_known_cross_derivatives():
    points = _make_model_points()
    conditions = _conditions(points)
    temperature_K = 293.15
    E = points["E_V_SHE"].to_numpy(dtype=float)
    ln_oh = np.log(points["electrolyte_concentration_M"].to_numpy(dtype=float))
    ln_co = np.log(points["CO_mole_fraction"].to_numpy(dtype=float))

    curvature = 6.0
    oh_cross = 1.7
    co_cross = -0.8
    log_rate = 1.2 + 4.0 * E + curvature * E**2 + 0.4 * ln_oh + oh_cross * E * ln_oh
    log_rate = log_rate - 0.2 * ln_co + co_cross * E * ln_co

    observable_points = _experimental_observable_points(points, log_rate, temperature_K)
    posterior = np.stack([log_rate, log_rate], axis=0)[None, ...]
    inference_data = SimpleNamespace(posterior={"ln_rate_model": posterior})

    result = build_second_order_outputs(
        inference_data=inference_data,
        model_points=points,
        conditions=conditions,
        observable_points=observable_points,
        temperature_K=temperature_K,
        koh_values=[0.25, 0.50, 1.00],
        co_values=[0.001, 0.01, 0.10, 1.00],
        potential_step_V=0.01,
    )

    factor = R_J_mol_K * temperature_K / F_C_mol
    expected_dalpha = 2.0 * curvature * factor
    expected_doh = oh_cross
    expected_dco = co_cross
    expected_delta2 = expected_dalpha - expected_doh

    expected = {
        "dalpha_dE": expected_dalpha,
        "ddelta_OH_dE": expected_doh,
        "ddelta_CO_dE": expected_dco,
        "delta2": expected_delta2,
    }
    for observable, value in expected.items():
        frame = result.points.loc[result.points["observable"] == observable]
        assert not frame.empty
        np.testing.assert_allclose(frame["experimental"], value, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(frame["median"], value, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(frame["residual_median"], 0.0, rtol=0, atol=1e-9)

    assert not result.summary["experimental_uncertainty_propagated"].any()
