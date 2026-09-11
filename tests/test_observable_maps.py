import numpy as np
import pandas as pd

from mkm.observable_maps import (
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
    build_potential_derivative_map,
    evaluate_linear_observable_map,
)


def _make_model_points():
    records = []
    model_point_id = 0
    condition_id = 0

    concentrations = [0.25, 0.50, 1.00]
    CO_values = [0.001, 0.01, 0.10, 1.00]

    for concentration in concentrations:
        for CO_fraction in CO_values:
            for grid_index in range(5):
                records.append(
                    {
                        "model_point_id": model_point_id,
                        "condition_id": condition_id,
                        "material": "test",
                        "electrolyte_concentration_M": concentration,
                        "CO_mole_fraction": CO_fraction,
                        "analysis_grid_index": grid_index,
                        "E_V_SHE": 0.01 * grid_index,
                    }
                )

                model_point_id += 1

            condition_id += 1

    return pd.DataFrame(records)


def test_alpha_map_recovers_linear_log_rate():
    points = _make_model_points()

    slope = 8.0
    log_rate = 1.5 + slope * points["E_V_SHE"].to_numpy()
    temperature_K = 293.15

    alpha_map = build_alpha_map(model_points=points, temperature_K=temperature_K)
    alpha = evaluate_linear_observable_map(log_rate=log_rate, observable_map=alpha_map)

    from mkm.constants import F_C_mol, R_J_mol_K

    expected = R_J_mol_K * temperature_K / F_C_mol * slope

    np.testing.assert_allclose(alpha, expected, rtol=1e-12, atol=1e-12)


def test_alpha_map_recovers_quadratic_derivative():
    points = _make_model_points()

    E = points["E_V_SHE"].to_numpy()
    log_rate = 2.0 + 4.0 * E + 6.0 * E**2
    temperature_K = 293.15

    alpha_map = build_alpha_map(points, temperature_K)
    alpha = evaluate_linear_observable_map(log_rate, alpha_map)

    from mkm.constants import F_C_mol, R_J_mol_K

    output_E = alpha_map.outputs["E_V_SHE"].to_numpy()
    expected = R_J_mol_K * temperature_K / F_C_mol * (4.0 + 12.0 * output_E)

    np.testing.assert_allclose(alpha, expected, rtol=1e-11, atol=1e-11)


def test_log_slope_map_recovers_known_order():
    points = _make_model_points()

    true_order = 1.3
    log_rate = (
        2.0
        + true_order * np.log(points["electrolyte_concentration_M"].to_numpy())
        + 3.0 * points["E_V_SHE"].to_numpy()
    )

    order_map = build_log_slope_order_map(
        model_points=points,
        varying_column="electrolyte_concentration_M",
        varying_values=[0.25, 0.50, 1.00],
        group_columns=["material", "CO_mole_fraction"],
    )

    order = evaluate_linear_observable_map(log_rate, order_map)

    np.testing.assert_allclose(order, true_order, rtol=1e-12, atol=1e-12)


def test_adjacent_order_map_recovers_known_order():
    points = _make_model_points()

    true_order = -0.45
    log_rate = 1.0 + true_order * np.log(points["CO_mole_fraction"].to_numpy()) + 2.0 * points["E_V_SHE"].to_numpy()

    order_map = build_adjacent_log_order_map(
        model_points=points,
        varying_column="CO_mole_fraction",
        varying_values=[0.001, 0.01, 0.10, 1.00],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="CO_lower",
        upper_value_column="CO_upper",
    )

    order = evaluate_linear_observable_map(log_rate, order_map)

    np.testing.assert_allclose(order, true_order, rtol=1e-12, atol=1e-12)


def test_adjacent_order_map_has_three_co_ranges():
    points = _make_model_points()

    order_map = build_adjacent_log_order_map(
        model_points=points,
        varying_column="CO_mole_fraction",
        varying_values=[0.001, 0.01, 0.10, 1.00],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="CO_lower",
        upper_value_column="CO_upper",
    )

    pairs = set(zip(order_map.outputs["CO_lower"], order_map.outputs["CO_upper"]))

    assert pairs == {(0.001, 0.01), (0.01, 0.10), (0.10, 1.00)}


def test_potential_derivative_map_recovers_alpha_curvature():
    points = _make_model_points()
    E = points["E_V_SHE"].to_numpy()
    log_rate = 2.0 + 4.0 * E + 6.0 * E**2
    temperature_K = 293.15

    alpha_map = build_alpha_map(points, temperature_K)
    derivative_map = build_potential_derivative_map(alpha_map, ["condition_id"], potential_step_V=0.01)
    derivative = evaluate_linear_observable_map(log_rate, derivative_map)

    from mkm.constants import F_C_mol, R_J_mol_K

    expected = (R_J_mol_K * temperature_K / F_C_mol) * 12.0
    np.testing.assert_allclose(derivative, expected, rtol=1e-10, atol=1e-10)
    assert np.allclose(derivative_map.outputs["potential_step_V"], 0.01)


def test_potential_derivative_map_requires_symmetric_support():
    points = _make_model_points()
    alpha_map = build_alpha_map(points, 293.15)
    derivative_map = build_potential_derivative_map(alpha_map, ["condition_id"], potential_step_V=0.02)

    counts = derivative_map.outputs.groupby("condition_id").size()
    assert set(counts.index) == set(alpha_map.outputs["condition_id"].unique())
    assert (counts == 1).all()
    assert np.allclose(derivative_map.outputs["E_V_SHE"], 0.02)
