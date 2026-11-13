"""Regression tests for PtRu validation and prior-predictive workflows."""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import xarray as xr

from mkm.inference.prior_predictive import summarize_prior_model_points
from mkm.models.ptru_registry import available_ptru_models
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_validation import (
    build_ptru_loco_fit,
    ptru_validation_output_dir,
    split_ptru_loco,
)


def synthetic_selected():
    rows = []
    for electrolyte in (0.1, 0.5):
        for co in (0.01, 0.1):
            for replicate in ("1", "2"):
                for index, potential in enumerate((0.25, 0.26, 0.27)):
                    rate = np.exp(-3.0 + potential + np.log(electrolyte) + np.log(co))
                    rows.append({
                        "medium": "acidic",
                        "material": "Pt100",
                        "C_electrolyte_M": electrolyte,
                        "CO_mole_fraction": co,
                        "replicate": replicate,
                        "analysis_grid_index": index,
                        "E_V_SHE": potential,
                        "rate_s_inv": rate,
                        "ln_rate": np.log(rate),
                    })
    return pd.DataFrame(rows)


def test_ptru_loco_excludes_all_replicates_and_potentials():
    data = synthetic_selected()
    training, heldout = split_ptru_loco(data, 0.1, 0.01)
    assert len(training) + len(heldout) == len(data)
    assert len(heldout) == 6
    assert set(heldout["replicate"]) == {"1", "2"}
    assert heldout["analysis_grid_index"].nunique() == 3
    assert set(heldout["C_electrolyte_M"]) == {0.1}
    assert set(heldout["CO_mole_fraction"]) == {0.01}
    assert not ((training["C_electrolyte_M"] == 0.1) & (training["CO_mole_fraction"] == 0.01)).any()


def test_ptru_loco_rejects_missing_condition_and_lomo():
    selected = synthetic_selected()
    with pytest.raises(ValueError, match="No Pt100 observations"):
        split_ptru_loco(selected, 0.2, 0.01)
    with pytest.raises(ValueError, match="only Pt100"):
        split_ptru_loco(selected.assign(material="Pt66Ru33"), 0.1, 0.01)
    with pytest.raises(ValueError, match="training condition"):
        split_ptru_loco(selected.iloc[:6], 0.1, 0.01)


def test_ptru_validation_path_is_separate_and_material_restricted(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    output = ptru_validation_output_dir(paths, "acidic", "CO_LH", 0.1, 0.01)
    assert output == (
        tmp_path / "results" / "PtRu_COOx_acidic" / "validation" / "loco" / "CO_LH"
        / "electrolyte_0.1M__CO_0.01"
    )
    assert available_ptru_models(fit_scope="all_materials") == ()
    with pytest.raises(ValueError, match="Unknown Pt100 mechanism"):
        ptru_validation_output_dir(paths, "acidic", "CO_BF", 0.1, 0.01)


def test_ptru_loco_fit_uses_training_points_only():
    paths = ProjectPaths.discover(__file__)
    training, _ = split_ptru_loco(synthetic_selected(), 0.1, 0.01)
    output = ptru_validation_output_dir(paths, "acidic", "CO_LH", 0.1, 0.01)
    fit = build_ptru_loco_fit(paths, "acidic", "CO_LH", training, output)
    assert fit.inputs.materials == ("Pt100",)
    assert len(fit.inputs.observation_rate) == len(training)
    assert len(fit.model_data.conditions) == 3
    assert fit.built_model.error_structure == "shared"
    assert set(fit.parameter_specs) >= {"sigma_rate_abs", "sigma_rate_rel"}


def test_prior_model_point_summary_contains_pt100_coverages():
    prior = xr.Dataset({
        "ln_rate_model": (("chain", "draw", "model_point"), np.log(np.full((1, 3, 2), 1.0))),
        "theta_OH": (("chain", "draw", "model_point"), np.full((1, 3, 2), 0.2)),
        "theta_empty": (("chain", "draw", "model_point"), np.full((1, 3, 2), 0.5)),
    })
    points = pd.DataFrame({"model_point_id": [0, 1]})
    summary = summarize_prior_model_points(SimpleNamespace(prior=prior), points)
    assert {"theta_OH_q50", "theta_empty_q50", "rate_model_q50"} <= set(summary.columns)
    np.testing.assert_allclose(summary["theta_OH_q50"], 0.2)
    np.testing.assert_allclose(summary["theta_empty_q50"], 0.5)
