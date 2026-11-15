"""Shared AgPd/PtRu postprocessing contracts and compatibility imports."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mkm.workflows import postprocessing_common as shared
from mkm.workflows.agpd_postprocess import products as agpd_products
from mkm.workflows.agpd_postprocess import status as agpd_status


def test_agpd_imports_retain_shared_implementations():
    assert agpd_products._compute_loo_stage is shared.compute_loo_stage
    assert agpd_products._compute_loo_pit_stage is shared.compute_loo_pit_stage
    assert agpd_products._merge_material_summaries is shared.merge_material_summaries
    assert agpd_products._overall_postprocessing_status is shared.overall_postprocessing_status
    assert agpd_status.status_row is shared.status_row
    assert agpd_status.safe_plot is shared.safe_plot
    assert agpd_status.write_postprocessing_status is shared.write_postprocessing_status


def test_common_material_summary_merges_without_duplicate_columns():
    first = pd.DataFrame({"material": ["Pt100"], "rmse": [0.20]})
    second = pd.DataFrame({"material": ["Pt100"], "rmse": [0.20], "loo": [-4.0]})
    combined = shared.merge_material_summaries(first, second)

    assert combined.to_dict("records") == [{"material": "Pt100", "rmse": 0.20, "loo": -4.0}]

    conflicting = second.assign(rmse=0.30)
    with pytest.raises(ValueError, match="Conflicting material-summary column"):
        shared.merge_material_summaries(first, conflicting)


@pytest.mark.parametrize("max_k,reason", [
    (np.inf, "non-finite"),
    (1.2, "LOO-PIT not attempted"),
])
def test_common_loo_pit_skips_unreliable_psis(tmp_path, max_k, reason):
    loo = SimpleNamespace(summary=pd.DataFrame({"max_pareto_k": [max_k]}))
    status = []

    assert shared.compute_loo_pit_stage(None, loo, tmp_path, status) is None
    assert status[-1]["stage"] == "loo_pit"
    assert status[-1]["status"] == "skipped_unreliable_psis"
    assert reason in status[-1]["message"]
    assert shared.overall_postprocessing_status([
        shared.status_row("core", "complete"), *status,
    ]) == "partial"


def test_common_loo_stage_writes_products_and_status(tmp_path, monkeypatch):
    summary = pd.DataFrame({
        "warning": [True], "good_k": [0.7], "max_pareto_k": [1.1],
        "n_pareto_k_above_good_k": [2],
    })
    pointwise = pd.DataFrame({"material": ["Pt100"], "pareto_k": [1.1]})
    loo = SimpleNamespace(summary=summary, pointwise=pointwise)
    run = SimpleNamespace(
        inference_data=object(), model_data=SimpleNamespace(observations=pd.DataFrame()),
        specification=SimpleNamespace(model_name="CO_ER_LH"),
    )
    monkeypatch.setattr(shared, "compute_loo_diagnostics", lambda *args, **kwargs: loo)
    monkeypatch.setattr(shared, "summarize_pointwise_loo_by_material", lambda frame: frame)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *args, **kwargs: None)
    status = []

    computed, material = shared.compute_loo_stage(run, tmp_path, status)
    assert computed is loo
    assert material is pointwise
    assert (tmp_path / "loo_summary.csv").is_file()
    assert status[-1]["stage"] == "loo"
    assert status[-1]["status"] == "warning"
    assert status[-1]["n_pareto_k_above_good_k"] == 2
