import numpy as np
import pandas as pd
import pytest

from mkm.postprocessing.observable_comparison import (
    build_experimental_observable_comparison,
    summarize_experimental_observable,
)


def test_experimental_observable_comparison_residuals_and_chain_spread():
    pooled = pd.DataFrame(
        {
            "point": [0, 1],
            "q025": [0.8, 1.8],
            "q50": [1.0, 2.0],
            "q975": [1.2, 2.2],
        }
    )
    by_chain = pd.DataFrame(
        {
            "chain": [0, 1, 0, 1],
            "point": [0, 0, 1, 1],
            "q025": [0.8, 0.9, 1.8, 1.9],
            "q50": [0.9, 1.1, 1.9, 2.1],
            "q975": [1.0, 1.2, 2.0, 2.2],
        }
    )
    experimental = pd.DataFrame({"point": [0, 1], "observed": [1.05, 1.95], "observed_sd": [0.1, 0.2]})

    result = build_experimental_observable_comparison(
        pooled, by_chain, experimental, ["point"], "observed", "observed_sd"
    )

    assert result.pooled["residual_q50"].tolist() == pytest.approx([-0.05, 0.05])
    assert result.chain_spread["chain_q50_range"].tolist() == pytest.approx([0.2, 0.2])
    assert result.chain_spread["chain_q50_range_over_exp_sd"].tolist() == pytest.approx([2.0, 1.0])

    summary = summarize_experimental_observable("test", result)

    assert summary["observable"] == "test"
    assert summary["n_points"] == 2
    assert np.isfinite(summary["median_abs_standardized_residual_q50"])