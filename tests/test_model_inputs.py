import numpy as np
import pandas as pd
import pytest

from mkm.model_data import build_model_data
from mkm.model_inputs import (
    build_model_coords,
    build_model_input_arrays,
    build_model_point_inputs,
)


def _selected_data():
    records = []
    for material, koh, co, base in (
        ("M1", 0.25, 0.001, 1.0),
        ("M2", 1.00, 0.10, 2.0),
    ):
        for replicate in ("A", "B", "C"):
            for grid_index, potential in enumerate((0.0, 0.01)):
                rate = base + 0.1 * grid_index
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
    return pd.DataFrame(records)


def _model_data_and_inputs():
    model_data = build_model_data(
        selected_replicates=_selected_data(),
        electrolyte_concentration_column="C_KOH_M",
    )
    return model_data, build_model_input_arrays(model_data)


def test_model_inputs_preserve_nonrectangular_table_mappings():
    model_data, inputs = _model_data_and_inputs()

    assert inputs.materials == ("M1", "M2")
    np.testing.assert_array_equal(
        inputs.observation_model_point_index,
        model_data.observations["model_point_id"].to_numpy(dtype=np.int64),
    )
    np.testing.assert_allclose(
        inputs.observation_rate,
        model_data.observations["rate_s_inv"].to_numpy(dtype=float),
    )


def test_model_inputs_store_condition_activities_in_log_space():
    model_data, inputs = _model_data_and_inputs()

    np.testing.assert_allclose(
        inputs.condition_ln_electrolyte_concentration,
        np.log(model_data.conditions["electrolyte_concentration_M"]),
    )
    np.testing.assert_allclose(
        inputs.condition_ln_CO_mole_fraction,
        np.log(model_data.conditions["CO_mole_fraction"]),
    )


def test_model_point_inputs_expand_condition_values_without_rectangularization():
    model_data, inputs = _model_data_and_inputs()
    point_inputs = build_model_point_inputs(inputs)

    assert len(point_inputs.E_V_SHE) == len(model_data.model_points)
    for point in model_data.model_points.itertuples(index=False):
        condition = model_data.conditions.loc[
            model_data.conditions["condition_id"] == point.condition_id
        ].iloc[0]
        assert point_inputs.materials[point_inputs.material_index[point.model_point_id]] == point.material
        assert point_inputs.E_V_SHE[point.model_point_id] == pytest.approx(point.E_V_SHE)
        assert point_inputs.ln_electrolyte_concentration[point.model_point_id] == pytest.approx(
            np.log(condition["electrolyte_concentration_M"])
        )
        assert point_inputs.ln_CO_mole_fraction[point.model_point_id] == pytest.approx(
            np.log(condition["CO_mole_fraction"])
        )


def test_model_coords_only_include_current_model_axes():
    _, inputs = _model_data_and_inputs()
    coords = build_model_coords(inputs)

    assert set(coords) == {"material", "condition", "model_point", "observation"}
    assert coords["material"] == ["M1", "M2"]


def test_nonpositive_observed_rate_is_rejected():
    selected = _selected_data()
    selected.loc[0, "rate_s_inv"] = 0.0
    selected.loc[0, "ln_rate"] = -np.inf

    with pytest.raises(ValueError, match="non-finite|positive"):
        build_model_data(
            selected_replicates=selected,
            electrolyte_concentration_column="C_KOH_M",
        )
