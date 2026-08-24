from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.model_inputs import ModelPointInputs
from mkm.postprocessing.drc import compute_transition_state_drc


MATERIAL = "Ag10Pd90"

CONFIG = {
    "temperature_K": 293.15,
    "gas": {
        "standard_state_pressure_bar": 1.0,
        "total_pressure_bar": 1.0,
    },
    "electrolyte": {
        "activity_model": "ideal_molarity",
        "standard_state_concentration_M": 1.0,
    },
    "surface_composition": {
        MATERIAL: {
            "Ag_fraction": 0.10,
            "Pd_fraction": 0.90,
        }
    },
}


def _point_inputs():
    return ModelPointInputs(
        materials=(MATERIAL,),
        material_index=np.array([0, 0], dtype=np.int64),
        E_V_SHE=np.array([0.30, 0.45]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 1.0])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.10])),
    )


def _model_points():
    return pd.DataFrame(
        {
            "model_point_id": [0, 1],
            "condition_id": [0, 1],
            "material": [MATERIAL, MATERIAL],
            "electrolyte_concentration_M": [0.5, 1.0],
            "CO_mole_fraction": [0.01, 0.10],
            "analysis_grid_index": [0, 1],
            "E_V_SHE": [0.30, 0.45],
        }
    )


def _idata(parameters):
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.asarray(values, dtype=float).reshape(1, -1))
            for name, values in parameters.items()
        },
        coords={"chain": [0], "draw": np.arange(len(next(iter(parameters.values()))))},
    )
    return SimpleNamespace(posterior=posterior)


def test_bf_transition_state_drc_is_one():
    idata = _idata(
        {
            "deltaG1_0": [-0.12, -0.13],
            "deltaG4_0": [0.11, 0.12],
            "deltaG5_0": [0.03, 0.04],
            "beta_2": [0.10, 0.15],
            "q": [0.30, 0.35],
            "Gact2_0": [0.66, 0.68],
        }
    )

    result = compute_transition_state_drc(
        inference_data=idata,
        model_name="BF",
        point_inputs=_point_inputs(),
        model_points=_model_points(),
        config=CONFIG,
        step_eV=1e-4,
    )

    np.testing.assert_allclose(result.draws["X_TS"].values, 1.0, rtol=0, atol=1e-9)
    assert result.checks.loc[0, "max_abs_sum_error"] < 1e-9


def test_bf_lh_transition_state_drcs_sum_to_one():
    idata = _idata(
        {
            "deltaG1_0": [-0.12, -0.13],
            "deltaG4_0": [0.11, 0.12],
            "deltaG5_0": [0.03, 0.04],
            "beta_2": [0.08, 0.12],
            "q": [0.30, 0.35],
            "Gact2_BF_0": [0.66, 0.68],
            "Gact2_LH_0": [0.72, 0.74],
        }
    )

    result = compute_transition_state_drc(
        inference_data=idata,
        model_name="BF_LH",
        point_inputs=_point_inputs(),
        model_points=_model_points(),
        config=CONFIG,
        step_eV=1e-4,
    )

    summed = result.draws["X_TS"].sum("control").values
    np.testing.assert_allclose(summed, 1.0, rtol=0, atol=2e-6)
    assert result.checks.loc[0, "max_abs_sum_error"] < 2e-6


def test_transition_state_drc_summary_preserves_model_point_metadata():
    idata = _idata(
        {
            "deltaG1_0": [-0.12, -0.13],
            "deltaG4_0": [0.11, 0.12],
            "deltaG5_0": [0.03, 0.04],
            "beta_2": [0.10, 0.15],
            "q": [0.30, 0.35],
            "Gact2_0": [0.66, 0.68],
        }
    )

    result = compute_transition_state_drc(
        inference_data=idata,
        model_name="BF",
        point_inputs=_point_inputs(),
        model_points=_model_points(),
        config=CONFIG,
    )

    assert result.summary["model_point_id"].tolist() == [0, 1]
    assert result.summary["control"].unique().tolist() == ["BF"]
    assert result.summary["q50"].tolist() == pytest.approx([1.0, 1.0])
