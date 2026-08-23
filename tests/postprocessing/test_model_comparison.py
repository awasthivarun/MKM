from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mkm.postprocessing.model_comparison import (
    build_loo_summary,
    build_pareto_k_table,
    build_pointwise_elpd_differences,
    build_pointwise_elpd_table,
    summarize_pointwise_elpd_differences,
)


def _fake_loo(elpd, elpd_i, pareto_k):
    return SimpleNamespace(
        elpd=elpd,
        se=1.2,
        p=3.4,
        n_samples=1000,
        n_data_points=len(elpd_i),
        good_k=0.7,
        warning=False,
        elpd_i=np.asarray(elpd_i, dtype=float),
        pareto_k=np.asarray(pareto_k, dtype=float),
    )


def test_build_loo_summary_reports_pareto_diagnostics():
    results = {
        "A": _fake_loo(-10.0, [-4.0, -6.0], [0.2, 0.8]),
        "B": _fake_loo(-8.0, [-3.0, -5.0], [0.1, 0.3]),
    }

    summary = build_loo_summary(results)

    assert summary["model"].tolist() == ["B", "A"]
    assert summary.loc[1, "max_pareto_k"] == pytest.approx(0.8)
    assert summary.loc[1, "n_pareto_k_above_good_k"] == 1


def test_pointwise_elpd_tables_preserve_observation_mapping():
    observations = pd.DataFrame(
        {
            "observation_id": [0, 1, 2],
            "E_V_SHE": [0.2, 0.3, 0.4],
        }
    )
    results = {
        "A": _fake_loo(-6.0, [-1.0, -2.0, -3.0], [0.1, 0.2, 0.3]),
        "B": _fake_loo(-5.0, [-0.5, -2.5, -2.0], [0.2, 0.2, 0.2]),
    }

    pointwise = build_pointwise_elpd_table(results, observations)
    pareto = build_pareto_k_table(results, observations)

    assert pointwise["elpd_A"].tolist() == pytest.approx([-1.0, -2.0, -3.0])
    assert pointwise["elpd_B"].tolist() == pytest.approx([-0.5, -2.5, -2.0])
    assert len(pareto) == 6


def test_pointwise_elpd_difference_sign_and_summary():
    frame = pd.DataFrame(
        {
            "observation_id": [0, 1, 2],
            "elpd_A": [-1.0, -2.0, -3.0],
            "elpd_B": [-0.5, -2.5, -2.0],
        }
    )

    differences = build_pointwise_elpd_differences(frame, ["A", "B"])
    summary = summarize_pointwise_elpd_differences(differences)

    assert differences["comparison"].unique().tolist() == ["A_minus_B"]
    assert differences["elpd_difference"].tolist() == pytest.approx([-0.5, 0.5, -1.0])
    assert summary.loc[0, "elpd_difference"] == pytest.approx(-1.0)
    assert summary.loc[0, "fraction_points_favoring_numerator"] == pytest.approx(1.0 / 3.0)