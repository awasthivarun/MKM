import numpy as np
import pandas as pd
import pytest

from mkm.postprocessing.residuals import (
    summarize_residual_curves,
    summarize_shared_replicate_residuals,
)


def test_summarize_residual_curves_detects_smooth_curve():
    frame = pd.DataFrame(
        {
            "material": ["M"] * 4,
            "electrolyte_concentration_M": [1.0] * 4,
            "CO_mole_fraction": [0.1] * 4,
            "replicate": ["A"] * 4,
            "E_V_SHE": [0.1, 0.2, 0.3, 0.4],
            "residual_conditional": [0.1, 0.2, 0.3, 0.4],
        }
    )

    result = summarize_residual_curves(frame)

    assert len(result) == 1
    assert result.loc[
        0,
        "lag1_residual_correlation",
    ] == pytest.approx(1.0)

    assert result.loc[
        0,
        "residual_slope_per_V",
    ] == pytest.approx(1.0)


def test_shared_replicate_decomposition_is_exact():
    rows = []

    potentials = [0.1, 0.2, 0.3]

    residuals = {
        "A": [1.1, 2.1, 3.1],
        "B": [1.0, 2.0, 3.0],
        "C": [0.9, 1.9, 2.9],
    }

    for replicate, values in residuals.items():
        for potential, residual in zip(
            potentials,
            values,
        ):
            rows.append(
                {
                    "material": "M",
                    "electrolyte_concentration_M": 1.0,
                    "CO_mole_fraction": 0.1,
                    "replicate": replicate,
                    "E_V_SHE": potential,
                    "residual_conditional": residual,
                }
            )

    frame = pd.DataFrame(rows)

    result = summarize_shared_replicate_residuals(
        frame
    )

    assert len(result) == 1
    assert result.loc[0, "n_replicates"] == 3

    shared_fraction = result.loc[
        0,
        "shared_fraction_squared_residual",
    ]

    replicate_fraction = result.loc[
        0,
        "replicate_fraction_squared_residual",
    ]

    assert (
        shared_fraction + replicate_fraction
    ) == pytest.approx(1.0)

    assert shared_fraction > 0.99