from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.postprocessing.materials import (
    summarize_material_noise,
    summarize_observation_diagnostics_by_material,
    summarize_pointwise_loo_by_material,
    summarize_setup_offsets,
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


def test_summarize_setup_offsets_maps_latents_to_physical_setups():
    posterior = xr.Dataset(
        {
            "z_ln_rate_setup": (
                ("chain", "draw", "setup"),
                np.array(
                    [
                        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                    ]
                ),
            ),
            "ln_rate_setup_offset": (
                ("chain", "draw", "setup"),
                np.array(
                    [
                        [[-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1]],
                        [[-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1]],
                    ]
                ),
            ),
        },
        coords={"chain": [0, 1], "draw": [0, 1], "setup": ["A", "B", "C"]},
    )
    observations = pd.DataFrame(
        {
            "material": ["Ag10Pd90"] * 6,
            "electrolyte_concentration_M": [0.25] * 6,
            "replicate": ["A", "A", "B", "B", "C", "C"],
        }
    )
    inputs = SimpleNamespace(
        setup_labels=(
            "material=Ag10Pd90 | electrolyte_concentration_M=0.25 | replicate=A",
            "material=Ag10Pd90 | electrolyte_concentration_M=0.25 | replicate=B",
            "material=Ag10Pd90 | electrolyte_concentration_M=0.25 | replicate=C",
        ),
        observation_setup_index=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        setup_experiment_index=np.array([0, 0, 0], dtype=np.int64),
    )

    result = summarize_setup_offsets(posterior, inputs, observations)

    assert result["setup_index"].tolist() == [0, 1, 2]
    assert result["setup_experiment_index"].tolist() == [0, 0, 0]
    assert result["material"].tolist() == ["Ag10Pd90"] * 3
    assert result["electrolyte_concentration_M"].tolist() == pytest.approx([0.25] * 3)
    assert result["replicate"].tolist() == ["A", "B", "C"]
    assert result["z_median"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert result["offset_median"].tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert result["offset_hdi95_lower"].tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert result["offset_hdi95_upper"].tolist() == pytest.approx([-0.1, 0.0, 0.1])


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
