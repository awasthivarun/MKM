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


def test_ptru_loco_checkpoint_tracks_training_and_heldout_content(monkeypatch):
    from mkm.model_data import build_model_data
    from mkm.workflows import ptru_validation as workflow

    paths = ProjectPaths.discover(__file__)
    training, heldout = split_ptru_loco(synthetic_selected(), 0.1, 0.01)
    training_data = build_model_data(training, electrolyte_concentration_column="C_electrolyte_M")
    heldout_data = build_model_data(heldout, electrolyte_concentration_column="C_electrolyte_M")
    fit = SimpleNamespace(model_data=training_data)

    # Test the LOCO-specific fingerprint contract separately from full-fit provenance.
    monkeypatch.setattr(workflow, "build_pt100_run_metadata", lambda *args: {})
    monkeypatch.setattr(workflow, "validate_pt100_run_metadata", lambda *args, **kwargs: None)

    metadata = workflow.ptru_loco_metadata(paths, fit, {}, 0.1, 0.01, heldout_data)
    assert len(metadata["training_observations_sha256"]) == 64
    assert len(metadata["heldout_observations_sha256"]) == 64
    workflow.validate_ptru_loco_metadata(
        metadata, paths, fit, 0.1, 0.01, heldout_data, allowed_statuses=("complete",),
    )

    # Canonical model-data ordering makes the fingerprints independent of raw row order.
    shuffled = training.sample(frac=1, random_state=7)
    shuffled_data = build_model_data(shuffled, electrolyte_concentration_column="C_electrolyte_M")
    assert workflow._model_data_observation_sha256(shuffled_data) == metadata["training_observations_sha256"]

    modified_training = training.copy()
    modified_training.loc[modified_training.index[0], "rate_s_inv"] *= 1.01
    modified_training["ln_rate"] = np.log(modified_training["rate_s_inv"])
    modified_data = build_model_data(modified_training, electrolyte_concentration_column="C_electrolyte_M")
    changed_fit = SimpleNamespace(model_data=modified_data)
    with pytest.raises(ValueError, match="training_observations_sha256"):
        workflow.validate_ptru_loco_metadata(
            metadata, paths, changed_fit, 0.1, 0.01, heldout_data, allowed_statuses=("complete",),
        )

    modified_heldout = heldout.copy()
    modified_heldout.loc[modified_heldout.index[0], "rate_s_inv"] *= 1.01
    modified_heldout["ln_rate"] = np.log(modified_heldout["rate_s_inv"])
    changed_heldout = build_model_data(modified_heldout, electrolyte_concentration_column="C_electrolyte_M")
    with pytest.raises(ValueError, match="heldout_observations_sha256"):
        workflow.validate_ptru_loco_metadata(
            metadata, paths, fit, 0.1, 0.01, changed_heldout, allowed_statuses=("complete",),
        )

    with pytest.raises(ValueError, match="training_observations_sha256"):
        workflow.validate_ptru_loco_metadata(
            {key: value for key, value in metadata.items() if key != "training_observations_sha256"},
            paths, fit, 0.1, 0.01, heldout_data, allowed_statuses=("complete",),
        )


def test_ptru_loco_and_full_fit_share_the_model_building_pipeline(monkeypatch, tmp_path):
    from mkm.workflows import pt100_fit

    paths = ProjectPaths.discover(__file__)
    selected = synthetic_selected()
    monkeypatch.setattr(pt100_fit.pd, "read_parquet", lambda _: selected)

    full = pt100_fit.build_pt100_fit(paths, "acidic", "CO_LH")
    validation = build_ptru_loco_fit(paths, "acidic", "CO_LH", selected, tmp_path / "loco")

    assert validation.output_dir == tmp_path / "loco"
    assert validation.data_path == full.data_path
    assert validation.specification == full.specification
    assert validation.parameter_specs == full.parameter_specs
    assert set(validation.built_model.model.named_vars) == set(full.built_model.model.named_vars)
    np.testing.assert_allclose(validation.inputs.observation_rate, full.inputs.observation_rate)
    np.testing.assert_allclose(validation.inputs.model_point_E_V_SHE, full.inputs.model_point_E_V_SHE)
