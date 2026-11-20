"""Transition-state DRC checks for new reversible and interacting AgPd models."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.constants import K_B_EV_K
from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_registry import get_agpd_model_definition
from mkm.postprocessing.drc import (
    _compile_log_rate_evaluator, compare_transition_state_drc_steps,
    compute_transition_state_drc, transition_state_controls,
)


PARAMETERS = {
    "deltaG1_0": -0.20, "deltaG3_0": -0.08, "deltaG4_0": 0.02, "deltaG5_0": 0.02,
    "q": 0.5, "beta_2_BF": 0.5, "Gact2_BF_0": 0.70, "Gact2_LH_0": 0.70,
    "Gact4_0": 0.65, "Gact5_0": 0.65, "beta_4": 0.5, "beta_5": 0.5,
    "beta_2_ER": 0.5, "beta_3": 0.5, "Gact1_0": 0.46,
    "Gact2_ER_0": 0.72, "Gact3_0": 0.68, "z_CO_CO_eV": 0.035,
    "z_CO_OH_eV": 0.010, "z_OH_OH_eV": 0.010,
}
CONFIG = {
    "temperature_K": 293.15,
    "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
    "electrolyte": {"standard_state_concentration_M": 1.0, "activity_model": "ideal_molarity"},
    "surface_composition": {"Ag50Pd50": {"Ag_fraction": 0.5, "Pd_fraction": 0.5}},
    "reversible_ssa": {"affinity_iterations": 48, "affinity_damping": 1.0,
                       "relative_residual_tolerance": 1e-8},
    "lateral_interactions": {"fixed_point_iterations": 192, "adaptive_convergence": True,
                             "fixed_point_damping": 0.5, "relative_residual_tolerance": 1e-7,
                             "branch_tolerance": 1e-5, "bep_coefficient": 0.65},
}


def _inputs():
    points = ModelPointInputs(
        materials=("Ag50Pd50",), material_index=np.zeros(2, dtype=np.int64),
        E_V_SHE=np.array([-0.05, 0.05]),
        ln_electrolyte_concentration=np.log(np.array([0.5, 0.5])),
        ln_CO_mole_fraction=np.log(np.array([0.01, 0.10])),
    )
    metadata = pd.DataFrame({"model_point_id": [0, 1], "condition_id": [0, 1],
                             "material": ["Ag50Pd50", "Ag50Pd50"], "E_V_SHE": [-0.05, 0.05]})
    return points, metadata


def _posterior(model_name):
    definition = get_agpd_model_definition(model_name)
    names = (field.name for field in fields(definition.parameter_class))
    offsets = {"deltaG1_0": -0.006, "Gact3_0": 0.007, "z_CO_CO_eV": 0.008,
               "z_CO_OH_eV": 0.004, "beta_2_ER": 0.02}
    dataset = xr.Dataset({name: (("chain", "draw"), np.array([
        [PARAMETERS[name], PARAMETERS[name] + offsets.get(name, 0.0)]
    ])) for name in names}, coords={"chain": [0], "draw": [0, 1]})
    return SimpleNamespace(posterior=dataset)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ("CO_ER_3", "c_CO_ER", "c_CO_ER_3", "CO_BF_ER_LH_3_pOH_sOH", "cxo_BF_ER_LH"))
def test_dynamic_transition_state_drc_sum_and_half_step(model_name):
    point_inputs, model_points = _inputs()
    kwargs = dict(inference_data=_posterior(model_name), model_name=model_name, point_inputs=point_inputs,
                  model_points=model_points, config=CONFIG)
    coarse = compute_transition_state_drc(**kwargs, step_eV=1e-4)
    fine = compute_transition_state_drc(**kwargs, step_eV=5e-5)

    actual = fine.draws["X_TS"].values
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual.sum(axis=2), 1.0, atol=5e-4, rtol=0)
    assert set(fine.draws.coords["control"].values) == {
        control.name for control in transition_state_controls(model_name)
    }
    difference = np.max(np.abs(coarse.draws["X_TS"].values - actual))
    assert difference < 5e-4
    assert fine.checks.loc[0, "max_abs_sum_error"] < 5e-4


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ("CO_ER_3", "c_CO_ER_3", "CO_BF_ER_LH_3_pOH_sOH", "cxo_BF_ER_LH"))
def test_uniform_transition_state_shift_scales_rate_without_changing_equilibria(model_name):
    point_inputs, _ = _inputs()
    names, evaluate = _compile_log_rate_evaluator(model_name, point_inputs, CONFIG)
    base = {name: PARAMETERS[name] for name in names}
    barriers = {control.parameter for control in transition_state_controls(model_name)}
    shift = 2e-3
    shifted = {name: value + shift if name in barriers else value for name, value in base.items()}
    ln_before = evaluate(*(base[name] for name in names))
    ln_after = evaluate(*(shifted[name] for name in names))
    np.testing.assert_allclose(ln_after - ln_before, -shift / (K_B_EV_K * CONFIG["temperature_K"]),
                               rtol=0, atol=5e-4)
