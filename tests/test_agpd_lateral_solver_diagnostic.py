"""Selection tests for the read-only Frumkin failed-draw diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_agpd_lateral_solver.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diagnose_agpd_lateral_solver", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_draw_groups_preserves_draw_and_model_point_axes():
    values = np.zeros((2, 3, 5))
    values[0, 2, 1] = np.nan
    values[0, 2, 4] = -np.inf
    values[1, 1, 0] = np.inf
    groups = _load_script().failed_draw_groups(values)
    assert [(c, d) for c, d, _ in groups] == [(0, 2), (1, 1)]
    np.testing.assert_array_equal(groups[0][2], [1, 4])
    np.testing.assert_array_equal(groups[1][2], [0])


def test_failed_draw_groups_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="chain, draw, model_point"):
        _load_script().failed_draw_groups(np.ones((3, 5)))
    assert _load_script().failed_draw_groups(np.ones((1, 2, 4))) == []


def test_replay_uses_saved_parameter_values_and_selected_points():
    import xarray as xr

    from mkm.mechanisms.agpd_basic.state import AgPdPointState

    values = {"deltaG1_0": -0.20, "deltaG4_0": 0.02, "beta_2_ER": 0.5,
              "Gact2_ER_0": 0.72, "z_CO_CO_eV": -0.01}
    prior = xr.Dataset({name: (("chain", "draw"), np.array([[value]]))
                        for name, value in values.items()})
    state = AgPdPointState(
        E_V_SHE=np.array([-0.05, 0.0, 0.05]),
        ln_a_CO=np.log(np.array([0.01, 0.1, 0.5])), ln_a_OH=np.log(np.full(3, 0.5)),
        Pd_fraction=np.full(3, 0.5), Ag_fraction=np.full(3, 0.5),
        material_index=np.zeros(3, dtype=np.int64), materials=("Ag50Pd50",),
    )
    solver = dict(fixed_point_iterations=32, fixed_point_damping=0.5,
                  relative_residual_tolerance=1e-7, branch_tolerance=1e-5,
                  bep_coefficient=0.65, diagnose_branches=True)
    observed, replay = _load_script().replay_failed_draw(
        prior, "c_ER", state, np.array([0, 2]), 0, 0, 293.15, solver,
    )
    assert observed == values
    assert replay["ln_rate_model"].shape == (2,)
    assert replay["coverage_branch_disagreement"].shape == (2,)
    assert replay["theta_CO"].shape == (2,)
    assert replay["theta_empty_Pd_from_CO_QEA"].shape == (2,)
    assert np.all(replay["theta_empty_Pd_from_CO_QEA"] > 0.0)
    assert np.all(np.isfinite(replay["coverage_solver_relative_residual"]))
    for branch in (0, 1):
        assert replay[f"branch{branch}_update_error"].shape == (2,)
        assert replay[f"branch{branch}_balance_residual"].shape == (2,)
        assert np.all(np.isfinite(replay[f"branch{branch}_update_error"]))
        for species in ("CO", "pOH", "COOH", "sOH", "empty_Pd"):
            assert replay[f"branch{branch}_theta_{species}"].shape == (2,)
    np.testing.assert_allclose(replay["theta_CO"], replay["branch0_theta_CO"])
    np.testing.assert_allclose(replay["coverage_branch_disagreement"],
                               np.maximum.reduce([abs(replay[f"branch0_theta_{name}"] -
                                                      replay[f"branch1_theta_{name}"])
                                                  for name in ("CO", "pOH", "COOH", "sOH")]))


def test_saturated_co_qea_uses_stable_vacancy():
    """Explicit vacancy retains the CO equilibrium when almost every primary site contains CO."""
    import xarray as xr

    from mkm.mechanisms.agpd_basic.state import AgPdPointState

    values = {"deltaG1_0": -0.45461, "deltaG4_0": -0.24207, "beta_2_ER": 0.0099815,
              "Gact2_ER_0": 0.78187, "z_CO_CO_eV": -0.0062272}
    prior = xr.Dataset({name: (("chain", "draw"), np.array([[value]]))
                        for name, value in values.items()})
    state = AgPdPointState(
        E_V_SHE=np.array([-0.25]), ln_a_CO=np.array([0.0]), ln_a_OH=np.log(np.array([0.25])),
        Pd_fraction=np.array([0.5]), Ag_fraction=np.array([0.5]),
        material_index=np.array([0], dtype=np.int64), materials=("Ag50Pd50",),
    )
    solver = dict(fixed_point_iterations=64, fixed_point_damping=0.5,
                  relative_residual_tolerance=1e-7, branch_tolerance=1e-5,
                  bep_coefficient=0.65, diagnose_branches=True)
    _, result = _load_script().replay_failed_draw(prior, "c_ER", state, np.array([0]), 0, 0, 293.15, solver)
    vacant = result["theta_empty_Pd"][0]
    vacant_qea = result["theta_empty_Pd_from_CO_QEA"][0]
    relative_difference = abs(vacant - vacant_qea) / vacant_qea
    assert vacant_qea > 0.0
    assert result["branch0_update_error"][0] < 1e-10
    assert np.isfinite(result["ln_rate_model"][0])
    assert result["coverage_solver_relative_residual"][0] < solver["relative_residual_tolerance"]
    assert relative_difference < 1e-9
