from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mkm.postprocessing.model_comparison import (
    build_pointwise_elpd_differences,
    build_pointwise_elpd_table,
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