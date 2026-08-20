import numpy as np
import pandas as pd

from mkm.preprocessing.observables import (
    F_C_mol,
    R_J_mol_K,
    add_transfer_coefficients,
    calculate_adjacent_log_orders,
    calculate_log_slope_order,
    calculate_transfer_coefficient,
    summarize_transfer_coefficients,
)


TEMPERATURE_K = 293.15


def test_constant_transfer_coefficient():
    potential = np.arange(-0.20, 0.21, 0.01)

    alpha_true = 0.50

    slope = (
        alpha_true
        * F_C_mol
        / (R_J_mol_K * TEMPERATURE_K)
    )

    ln_rate = slope * potential

    alpha = calculate_transfer_coefficient(
        potential_V=potential,
        ln_rate=ln_rate,
        temperature_K=TEMPERATURE_K,
    )

    np.testing.assert_allclose(
        alpha,
        alpha_true,
        rtol=1e-12,
        atol=1e-12,
    )


def test_transfer_coefficient_for_quadratic_log_rate():
    potential = np.arange(-0.20, 0.21, 0.01)

    a = 4.0
    b = 20.0

    ln_rate = a + b * potential**2

    expected = (
        R_J_mol_K
        * TEMPERATURE_K
        / F_C_mol
        * 2.0
        * b
        * potential
    )

    alpha = calculate_transfer_coefficient(
        potential_V=potential,
        ln_rate=ln_rate,
        temperature_K=TEMPERATURE_K,
    )

    np.testing.assert_allclose(
        alpha,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_add_transfer_coefficients_by_curve():
    potential = np.arange(-0.20, 0.21, 0.01)

    alpha_A = 0.40
    alpha_B = 0.60

    slope_A = (
        alpha_A
        * F_C_mol
        / (R_J_mol_K * TEMPERATURE_K)
    )

    slope_B = (
        alpha_B
        * F_C_mol
        / (R_J_mol_K * TEMPERATURE_K)
    )

    import pandas as pd

    data = pd.DataFrame(
        {
            "condition": (
                ["test"] * len(potential)
                + ["test"] * len(potential)
            ),
            "replicate": (
                ["A"] * len(potential)
                + ["B"] * len(potential)
            ),
            "E_V_SHE": np.concatenate(
                [potential, potential]
            ),
            "ln_rate": np.concatenate(
                [
                    slope_A * potential,
                    slope_B * potential,
                ]
            ),
        }
    )

    result = add_transfer_coefficients(
        data=data,
        group_columns=[
            "condition",
            "replicate",
        ],
        temperature_K=TEMPERATURE_K,
    )

    result_A = result[
        result["replicate"] == "A"
    ]

    result_B = result[
        result["replicate"] == "B"
    ]

    np.testing.assert_allclose(
        result_A["alpha"],
        alpha_A,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result_B["alpha"],
        alpha_B,
        rtol=1e-12,
        atol=1e-12,
    )


def test_summarize_transfer_coefficients():
    import pandas as pd

    data = pd.DataFrame(
        {
            "condition": [
                "test",
                "test",
                "test",
            ],
            "analysis_grid_index": [
                0,
                0,
                0,
            ],
            "replicate": [
                "A",
                "B",
                "C",
            ],
            "alpha": [
                0.4,
                0.5,
                0.6,
            ],
        }
    )

    summary = summarize_transfer_coefficients(
        data=data,
        group_columns=[
            "condition",
            "analysis_grid_index",
        ],
    )

    assert len(summary) == 1

    assert summary.loc[
        0,
        "n_replicates",
    ] == 3

    np.testing.assert_allclose(
        summary.loc[0, "alpha_mean"],
        0.5,
    )

    np.testing.assert_allclose(
        summary.loc[0, "alpha_sd"],
        0.1,
    )


def test_single_replicate_alpha_sd_is_nan():
    import pandas as pd

    data = pd.DataFrame(
        {
            "condition": ["test"],
            "analysis_grid_index": [0],
            "replicate": ["A"],
            "alpha": [0.5],
        }
    )

    summary = summarize_transfer_coefficients(
        data=data,
        group_columns=[
            "condition",
            "analysis_grid_index",
        ],
    )

    assert summary.loc[
        0,
        "n_replicates",
    ] == 1

    assert np.isnan(
        summary.loc[
            0,
            "alpha_sd",
        ]
    )


def test_log_slope_order_recovers_known_power_law():
    concentrations = np.array([
        0.25,
        0.50,
        1.00,
    ])

    potential = np.array([
        -0.10,
        0.00,
        0.10,
    ])

    true_order = 1.5

    records = []

    for concentration in concentrations:
        for grid_index, E in enumerate(potential):
            base_log_rate = 2.0 + 3.0 * E

            records.append(
                {
                    "material": "test",
                    "CO_mole_fraction": 0.01,
                    "C_KOH_M": concentration,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": E,
                    "ln_rate_mean": (
                        base_log_rate
                        + true_order
                        * np.log(concentration)
                    ),
                    "ln_rate_sd": 0.1,
                }
            )

    data = pd.DataFrame(records)

    result = calculate_log_slope_order(
        data=data,
        varying_column="C_KOH_M",
        varying_values=concentrations,
        group_columns=[
            "material",
            "CO_mole_fraction",
        ],
        output_column="delta_OH",
        output_sd_column="delta_OH_sd",
    )

    np.testing.assert_allclose(
        result["delta_OH"].to_numpy(),
        true_order,
        rtol=1e-12,
        atol=1e-12,
    )


def test_log_slope_order_uses_largest_common_grid():
    concentrations = [
        0.25,
        0.50,
        1.00,
    ]

    grids = {
        0.25: [-2, -1, 0, 1, 2],
        0.50: [-1, 0, 1, 2],
        1.00: [0, 1, 2, 3],
    }

    true_order = 0.75

    records = []

    for concentration in concentrations:
        for grid_index in grids[concentration]:
            E = 0.01 * grid_index

            records.append(
                {
                    "material": "test",
                    "CO_mole_fraction": 0.01,
                    "C_KOH_M": concentration,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": E,
                    "ln_rate_mean": (
                        1.0
                        + true_order
                        * np.log(concentration)
                    ),
                    "ln_rate_sd": 0.1,
                }
            )

    data = pd.DataFrame(records)

    result = calculate_log_slope_order(
        data=data,
        varying_column="C_KOH_M",
        varying_values=concentrations,
        group_columns=[
            "material",
            "CO_mole_fraction",
        ],
        output_column="delta_OH",
        output_sd_column="delta_OH_sd",
    )

    np.testing.assert_array_equal(
        result["analysis_grid_index"].to_numpy(),
        np.array([0, 1, 2]),
    )

    np.testing.assert_allclose(
        result["delta_OH"],
        true_order,
    )


def test_log_slope_order_uncertainty():
    concentrations = np.array([
        0.25,
        0.50,
        1.00,
    ])

    log_rate_sds = np.array([
        0.10,
        0.20,
        0.30,
    ])

    records = []

    for concentration, sd in zip(
        concentrations,
        log_rate_sds,
    ):
        records.append(
            {
                "material": "test",
                "CO_mole_fraction": 0.01,
                "C_KOH_M": concentration,
                "analysis_grid_index": 0,
                "E_V_SHE": 0.0,
                "ln_rate_mean": (
                    1.2 * np.log(concentration)
                ),
                "ln_rate_sd": sd,
            }
        )

    data = pd.DataFrame(records)

    result = calculate_log_slope_order(
        data=data,
        varying_column="C_KOH_M",
        varying_values=concentrations,
        group_columns=[
            "material",
            "CO_mole_fraction",
        ],
        output_column="delta_OH",
        output_sd_column="delta_OH_sd",
    )

    log_C = np.log(concentrations)

    centered = (
        log_C
        - np.mean(log_C)
    )

    weights = (
        centered
        / np.sum(centered**2)
    )

    expected_sd = np.sqrt(
        np.sum(
            (weights * log_rate_sds)**2
        )
    )

    np.testing.assert_allclose(
        result.loc[0, "delta_OH_sd"],
        expected_sd,
    )


def test_adjacent_log_orders_recover_known_power_law():
    CO_values = np.array([
        0.001,
        0.01,
        0.10,
        1.00,
    ])

    potential = np.array([
        -0.10,
        0.00,
        0.10,
    ])

    true_order = -0.4

    records = []

    for CO_fraction in CO_values:
        for grid_index, E in enumerate(potential):
            records.append(
                {
                    "material": "test",
                    "C_KOH_M": 0.50,
                    "CO_mole_fraction": CO_fraction,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": E,
                    "ln_rate_mean": (
                        2.0
                        + 4.0 * E
                        + true_order
                        * np.log(CO_fraction)
                    ),
                    "ln_rate_sd": 0.1,
                }
            )

    data = pd.DataFrame(records)

    result = calculate_adjacent_log_orders(
        data=data,
        varying_column="CO_mole_fraction",
        varying_values=CO_values,
        group_columns=[
            "material",
            "C_KOH_M",
        ],
        output_column="delta_CO",
        output_sd_column="delta_CO_sd",
        lower_value_column="CO_lower",
        upper_value_column="CO_upper",
    )

    np.testing.assert_allclose(
        result["delta_CO"],
        true_order,
        rtol=1e-12,
        atol=1e-12,
    )

    assert set(
        zip(
            result["CO_lower"],
            result["CO_upper"],
        )
    ) == {
        (0.001, 0.01),
        (0.01, 0.10),
        (0.10, 1.00),
    }


def test_adjacent_log_orders_use_pair_specific_common_grids():
    grids = {
        0.001: [0, 1, 2],
        0.01: [-1, 0, 1, 2, 3],
        0.10: [-2, -1, 0, 1, 2, 3, 4],
        1.00: [1, 2, 3, 4],
    }

    records = []

    for CO_fraction, grid_indices in grids.items():
        for grid_index in grid_indices:
            records.append(
                {
                    "material": "test",
                    "C_KOH_M": 0.50,
                    "CO_mole_fraction": CO_fraction,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": 0.01 * grid_index,
                    "ln_rate_mean": (
                        0.5
                        + 0.25
                        * np.log(CO_fraction)
                    ),
                    "ln_rate_sd": 0.1,
                }
            )

    data = pd.DataFrame(records)

    result = calculate_adjacent_log_orders(
        data=data,
        varying_column="CO_mole_fraction",
        varying_values=[
            0.001,
            0.01,
            0.10,
            1.00,
        ],
        group_columns=[
            "material",
            "C_KOH_M",
        ],
        output_column="delta_CO",
        output_sd_column="delta_CO_sd",
        lower_value_column="CO_lower",
        upper_value_column="CO_upper",
    )

    pair_1 = result[
        (result["CO_lower"] == 0.001)
        & (result["CO_upper"] == 0.01)
    ]

    pair_2 = result[
        (result["CO_lower"] == 0.01)
        & (result["CO_upper"] == 0.10)
    ]

    pair_3 = result[
        (result["CO_lower"] == 0.10)
        & (result["CO_upper"] == 1.00)
    ]

    np.testing.assert_array_equal(
        pair_1["analysis_grid_index"],
        np.array([0, 1, 2]),
    )

    np.testing.assert_array_equal(
        pair_2["analysis_grid_index"],
        np.array([-1, 0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(
        pair_3["analysis_grid_index"],
        np.array([1, 2, 3, 4]),
    )


def test_adjacent_log_order_uncertainty():
    lower = 0.001
    upper = 0.01

    lower_sd = 0.2
    upper_sd = 0.3

    data = pd.DataFrame(
        {
            "material": [
                "test",
                "test",
            ],
            "C_KOH_M": [
                0.50,
                0.50,
            ],
            "CO_mole_fraction": [
                lower,
                upper,
            ],
            "analysis_grid_index": [
                0,
                0,
            ],
            "E_V_SHE": [
                0.0,
                0.0,
            ],
            "ln_rate_mean": [
                1.0,
                2.0,
            ],
            "ln_rate_sd": [
                lower_sd,
                upper_sd,
            ],
        }
    )

    result = calculate_adjacent_log_orders(
        data=data,
        varying_column="CO_mole_fraction",
        varying_values=[
            lower,
            upper,
        ],
        group_columns=[
            "material",
            "C_KOH_M",
        ],
        output_column="delta_CO",
        output_sd_column="delta_CO_sd",
        lower_value_column="CO_lower",
        upper_value_column="CO_upper",
    )

    expected_sd = (
        np.sqrt(
            lower_sd**2
            + upper_sd**2
        )
        / np.log(
            upper / lower
        )
    )

    np.testing.assert_allclose(
        result.loc[0, "delta_CO_sd"],
        expected_sd,
    )