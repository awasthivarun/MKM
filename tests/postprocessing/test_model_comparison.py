from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mkm.postprocessing.model_comparison import (
    build_pointwise_elpd_differences,
    build_pointwise_elpd_table,
    compute_loo_results,
    summarize_pointwise_elpd_differences,
)


def _fake_loo(elpd_i):
    return SimpleNamespace(elpd_i=np.asarray(elpd_i, dtype=float))


def test_pointwise_elpd_table_preserves_observation_mapping():
    observations = pd.DataFrame(
        {
            "observation_id": [0, 1, 2],
            "E_V_SHE": [0.2, 0.3, 0.4],
        }
    )
    results = {
        "A": _fake_loo([-1.0, -2.0, -3.0]),
        "B": _fake_loo([-0.5, -2.5, -2.0]),
    }

    pointwise = build_pointwise_elpd_table(results, observations)

    assert pointwise["elpd_A"].tolist() == pytest.approx([-1.0, -2.0, -3.0])
    assert pointwise["elpd_B"].tolist() == pytest.approx([-0.5, -2.5, -2.0])


def test_pointwise_elpd_difference_uses_later_minus_earlier_convention():
    frame = pd.DataFrame(
        {
            "observation_id": [0, 1, 2],
            "elpd_A": [-1.0, -2.0, -3.0],
            "elpd_B": [-0.5, -2.5, -2.0],
        }
    )

    differences = build_pointwise_elpd_differences(frame, ["A", "B"])
    summary = summarize_pointwise_elpd_differences(differences)

    assert differences["comparison"].unique().tolist() == ["B_minus_A"]
    assert differences["elpd_difference"].tolist() == pytest.approx([0.5, -0.5, 1.0])
    assert summary.loc[0, "elpd_difference"] == pytest.approx(1.0)
    assert summary.loc[0, "fraction_points_favoring_numerator"] == pytest.approx(2.0 / 3.0)

def test_compute_loo_results_uses_canonical_loo_helper(monkeypatch):
    calls = []

    def fake_compute_loo_result(inference_data, var_name, pointwise):
        calls.append((inference_data, var_name, pointwise))
        return SimpleNamespace(n_data_points=3), 0.75

    monkeypatch.setattr(
        "mkm.postprocessing.model_comparison.compute_loo_result",
        fake_compute_loo_result,
    )

    inputs = {"A": object(), "B": object()}
    results = compute_loo_results(inputs, var_name="rate_observed")

    assert list(results) == ["A", "B"]
    assert len(calls) == 2
    assert all(var_name == "rate_observed" for _, var_name, _ in calls)
    assert all(pointwise is True for _, _, pointwise in calls)
