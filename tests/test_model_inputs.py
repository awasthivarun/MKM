import numpy as np
import pandas as pd

from mkm.model_data import (
    build_model_data,
)
from mkm.model_inputs import (
    build_model_input_arrays,
    build_model_point_inputs,
)


def _make_test_data():
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

    records.append(
        {
            "material": "M2",
            "C_KOH_M": 1.00,
            "CO_mole_fraction": 0.10,
            "replicate": "A",
            "analysis_grid_index": 3,
            "E_V_SHE": 0.03,
            "rate_s_inv": 2.0,
            "ln_rate": np.log(2.0),
        }
    )

    return pd.DataFrame(
        records
    )


def _build_inputs():
    data = _make_test_data()

    model_data = build_model_data(
        selected_replicates=data,
        electrolyte_concentration_column="C_KOH_M",
    )

    return (
        model_data,
        build_model_input_arrays(
            model_data
        ),
    )


def test_model_inputs_preserve_materials():
    _, inputs = _build_inputs()

    assert inputs.materials == (
        "M1",
        "M2",
    )

    np.testing.assert_array_equal(
        inputs.condition_material_index,
        np.array([
            0,
            1,
        ]),
    )


def test_model_inputs_use_log_condition_variables():
    model_data, inputs = _build_inputs()

    expected_ln_C = np.log(
        model_data.conditions[
            "electrolyte_concentration_M"
        ].to_numpy()
    )

    expected_ln_CO = np.log(
        model_data.conditions[
            "CO_mole_fraction"
        ].to_numpy()
    )

    np.testing.assert_allclose(
        inputs.condition_ln_electrolyte_concentration,
        expected_ln_C,
    )

    np.testing.assert_allclose(
        inputs.condition_ln_CO_mole_fraction,
        expected_ln_CO,
    )


def test_model_point_condition_mapping():
    model_data, inputs = _build_inputs()

    for row in (
        model_data.model_points
        .itertuples(index=False)
    ):
        assert (
            inputs.model_point_condition_index[
                row.model_point_id
            ]
            == row.condition_id
        )

        np.testing.assert_allclose(
            inputs.model_point_E_V_SHE[
                row.model_point_id
            ],
            row.E_V_SHE,
        )


def test_observation_model_point_mapping():
    model_data, inputs = _build_inputs()

    for row in (
        model_data.observations
        .itertuples(index=False)
    ):
        assert (
            inputs.observation_model_point_index[
                row.observation_id
            ]
            == row.model_point_id
        )

        np.testing.assert_allclose(
            inputs.observation_ln_rate[
                row.observation_id
            ],
            row.ln_rate,
        )

        assert (
            inputs.observation_replicate[
                row.observation_id
            ]
            == row.replicate
        )


def test_shared_replicates_share_prediction_index():
    model_data, inputs = _build_inputs()

    point = model_data.model_points[
        (
            model_data.model_points[
                "material"
            ]
            == "M1"
        )
        & (
            model_data.model_points[
                "analysis_grid_index"
            ]
            == 0
        )
    ].iloc[0]

    observations = model_data.observations[
        model_data.observations[
            "model_point_id"
        ]
        == point["model_point_id"]
    ]

    indices = (
        inputs.observation_model_point_index[
            observations[
                "observation_id"
            ].to_numpy()
        ]
    )

    np.testing.assert_array_equal(
        indices,
        np.full(
            3,
            point["model_point_id"],
        ),
    )


def test_model_point_inputs_expand_condition_variables():
    model_data, inputs = _build_inputs()

    point_inputs = build_model_point_inputs(
        inputs
    )

    assert (
        len(point_inputs.E_V_SHE)
        == len(model_data.model_points)
    )

    for point in (
        model_data.model_points
        .itertuples(index=False)
    ):
        condition = (
            model_data.conditions[
                model_data.conditions[
                    "condition_id"
                ]
                == point.condition_id
            ]
            .iloc[0]
        )

        np.testing.assert_allclose(
            point_inputs.E_V_SHE[
                point.model_point_id
            ],
            point.E_V_SHE,
        )

        np.testing.assert_allclose(
            point_inputs.ln_electrolyte_concentration[
                point.model_point_id
            ],
            np.log(
                condition[
                    "electrolyte_concentration_M"
                ]
            ),
        )

        np.testing.assert_allclose(
            point_inputs.ln_CO_mole_fraction[
                point.model_point_id
            ],
            np.log(
                condition[
                    "CO_mole_fraction"
                ]
            ),
        )


def test_model_point_material_mapping():
    model_data, inputs = _build_inputs()

    point_inputs = build_model_point_inputs(
        inputs
    )

    material_to_index = {
        material: index
        for index, material in enumerate(
            point_inputs.materials
        )
    }

    for point in (
        model_data.model_points
        .itertuples(index=False)
    ):
        expected = material_to_index[
            point.material
        ]

        assert (
            point_inputs.material_index[
                point.model_point_id
            ]
            == expected
        )


