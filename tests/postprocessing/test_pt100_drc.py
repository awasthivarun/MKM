"""Six Pt100 TS DRC regression cases using no posterior sampling."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.model_inputs import ModelPointInputs
from mkm.postprocessing.drc import (
    compare_transition_state_drc_steps, compute_pt100_transition_state_drc,
)


@pytest.mark.parametrize("medium", ("acidic", "basic"))
@pytest.mark.parametrize("mechanism", ("CO_LH", "CO_ER", "CO_ER_LH"))
def test_pt100_transition_state_drc_sums_and_half_step(medium, mechanism):
    parameters = {
        "deltaG1_0": -0.12, "deltaG4_0": 0.20,
        "Gact1_0": 0.61, "Gact2_ER_0": 0.73,
        "Gact2_LH_0": 0.68, "beta_2_ER": 0.40,
    }
    if mechanism == "CO_LH":
        selected = ("deltaG1_0", "deltaG4_0", "Gact1_0", "Gact2_LH_0")
    elif mechanism == "CO_ER":
        selected = ("deltaG1_0", "deltaG4_0", "Gact1_0", "Gact2_ER_0", "beta_2_ER")
    else:
        selected = tuple(parameters)
    posterior = xr.Dataset({
        name: (("chain", "draw"), np.array([[parameters[name], parameters[name] + 1e-3]]))
        for name in selected
    }, coords={"chain": [0], "draw": [0, 1]})
    inputs = ModelPointInputs(
        materials=("Pt100",), material_index=np.array([0, 0]),
        E_V_SHE=np.array([0.70, 0.85]) if medium == "acidic" else np.array([0.20, 0.40]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 1.0])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.1])),
    )
    points = pd.DataFrame({
        "model_point_id": [0, 1], "condition_id": [0, 1],
        "material": ["Pt100", "Pt100"],
        "electrolyte_concentration_M": [0.5, 1.0], "CO_mole_fraction": [0.01, 0.1],
        "analysis_grid_index": [0, 1], "E_V_SHE": inputs.E_V_SHE,
    })
    kwargs = dict(
        inference_data=SimpleNamespace(posterior=posterior), model_name=mechanism,
        medium=medium, point_inputs=inputs, model_points=points, temperature_K=293.15,
    )
    coarse = compute_pt100_transition_state_drc(**kwargs, step_eV=1e-4)
    fine = compute_pt100_transition_state_drc(**kwargs, step_eV=5e-5)
    np.testing.assert_allclose(coarse.draws["X_TS"].sum("control"), 1.0, atol=2e-5, rtol=0)
    assert coarse.checks.iloc[0]["max_abs_sum_error"] < 2e-5
    assert set(coarse.summary["material"]) == {"Pt100"}
    assert len(coarse.summary["control"].unique()) == (3 if mechanism == "CO_ER_LH" else 2)
    steps = compare_transition_state_drc_steps(coarse, fine)
    assert np.isfinite(steps["max_abs_difference"]).all()
    assert (steps["max_abs_difference"] < 2e-4).all()
