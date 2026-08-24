import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.materials import (
    summarize_material_noise,
    summarize_observation_diagnostics_by_material,
    summarize_pointwise_loo_by_material,
)


def test_summarize_material_noise_preserves_material_labels():
    posterior = xr.Dataset(
        {
            "sigma_ln_rate_material": (
                ("chain", "draw", "material"),
                np.array(
                    [
                        [[0.2, 0.5], [0.2, 0.5]],
                        [[0.2, 0.5], [0.2, 0.5]],
                    ]
                ),
            )
        },
        coords={"chain": [0, 1], "draw": [0, 1], "material": ["Pd100", "Ag10Pd90"]},
    )

    result = summarize_material_noise(posterior)

    assert result["material"].tolist() == ["Pd100", "Ag10Pd90"]
    assert result["median"].tolist() == pytest.approx([0.2, 0.5])


def test_summarize_observation_diagnostics_by_material():
    frame = pd.DataFrame(
        {
            "material": ["A", "A", "B"],
            "residual_mechanism": [1.0, -1.0, 2.0],
            "residual_conditional": [0.5, -0.5, 1.0],
            "standardized_residual_conditional": [1.0, -1.0, 2.0],
            "observed_inside_predictive_95_hdi": [True, False, True],
        }
    )

    result = summarize_observation_diagnostics_by_material(frame).set_index("material")

    assert result.loc["A", "conditional_residual_rms"] == pytest.approx(0.5)
    assert result.loc["A", "predictive_95_hdi_coverage"] == pytest.approx(0.5)
    assert result.loc["B", "median_abs_standardized_residual"] == pytest.approx(2.0)


def test_summarize_pointwise_loo_by_material_is_explicitly_additive():
    frame = pd.DataFrame(
        {
            "material": ["A", "A", "B"],
            "elpd_loo": [-1.0, -2.0, -4.0],
            "pareto_k": [0.1, 0.2, 0.8],
        }
    )

    result = summarize_pointwise_loo_by_material(frame).set_index("material")

    assert result.loc["A", "elpd_loo_contribution"] == pytest.approx(-3.0)
    assert result.loc["B", "n_pareto_k_above_0p7"] == 1
