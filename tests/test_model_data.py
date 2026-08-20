import numpy as np
import pandas as pd

from mkm.model_data import (
    build_model_data,
)


def _make_incomplete_test_data():
    records = []

    for replicate in [
        "A",
        "B",
        "C",
    ]:
        for grid_index in [
            0,
            1,
        ]:
            rate = (
                1.0
                + 0.1 * grid_index
            )

            records.append(
                {
                    "material": "M1",
                    "C_KOH_M": 0.25,
                    "CO_mole_fraction": 0.001,
                    "replicate": replicate,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": 0.01 * grid_index,
                    "rate_s_inv": rate,
                    "ln_rate": np.log(rate),
                }
            )

    rate = 2.0

    records.append(
        {
            "material": "M1",
            "C_KOH_M": 1.00,
            "CO_mole_fraction": 0.10,
            "replicate": "A",
            "analysis_grid_index": 3,
            "E_V_SHE": 0.03,
            "rate_s_inv": rate,
            "ln_rate": np.log(rate),
        }
    )

    return pd.DataFrame(
        records
    )


def test_model_data_preserves_only_observed_conditions():
    data = _make_incomplete_test_data()

    result = build_model_data(
        selected_replicates=data,
        electrolyte_concentration_column="C_KOH_M",
    )

    assert len(
        result.conditions
    ) == 2

    observed_conditions = set(
        zip(
            result.conditions[
                "electrolyte_concentration_M"
            ],
            result.conditions[
                "CO_mole_fraction"
            ],
        )
    )

    assert observed_conditions == {
        (0.25, 0.001),
        (1.00, 0.10),
    }


def test_model_data_separates_points_from_replicates():
    data = _make_incomplete_test_data()

    result = build_model_data(
        selected_replicates=data,
        electrolyte_concentration_column="C_KOH_M",
    )

    assert len(
        result.model_points
    ) == 3

    assert len(
        result.observations
    ) == 7


def test_replicates_share_model_point():
    data = _make_incomplete_test_data()

    result = build_model_data(
        selected_replicates=data,
        electrolyte_concentration_column="C_KOH_M",
    )

    point = result.model_points[
        (
            result.model_points[
                "electrolyte_concentration_M"
            ]
            == 0.25
        )
        & (
            result.model_points[
                "CO_mole_fraction"
            ]
            == 0.001
        )
        & (
            result.model_points[
                "analysis_grid_index"
            ]
            == 0
        )
    ].iloc[0]

    observations = result.observations[
        result.observations[
            "model_point_id"
        ]
        == point["model_point_id"]
    ]

    assert len(
        observations
    ) == 3

    assert set(
        observations[
            "replicate"
        ]
    ) == {
        "A",
        "B",
        "C",
    }


def test_model_data_allows_variable_replicate_counts():
    data = _make_incomplete_test_data()

    result = build_model_data(
        selected_replicates=data,
        electrolyte_concentration_column="C_KOH_M",
    )

    replicate_counts = (
        result.observations
        .groupby(
            "model_point_id"
        )
        .size()
        .sort_values()
        .to_numpy()
    )

    np.testing.assert_array_equal(
        replicate_counts,
        np.array([
            1,
            3,
            3,
        ]),
    )


def test_model_data_ids_are_contiguous():
    data = _make_incomplete_test_data()

    result = build_model_data(
        selected_replicates=data,
        electrolyte_concentration_column="C_KOH_M",
    )

    np.testing.assert_array_equal(
        result.conditions[
            "condition_id"
        ].to_numpy(),
        np.arange(
            len(
                result.conditions
            )
        ),
    )

    np.testing.assert_array_equal(
        result.model_points[
            "model_point_id"
        ].to_numpy(),
        np.arange(
            len(
                result.model_points
            )
        ),
    )

    np.testing.assert_array_equal(
        result.observations[
            "observation_id"
        ].to_numpy(),
        np.arange(
            len(
                result.observations
            )
        ),
    )



