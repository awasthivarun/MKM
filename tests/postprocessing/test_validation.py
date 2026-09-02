import numpy as np
import pandas as pd
import xarray as xr

from mkm.postprocessing.validation import (
    plot_heldout_pit_conditions,
    plot_validation_parameter_posteriors,
    summarize_validation_posterior_shift,
)


def _inference_data(x_values, sigma_values):
    posterior = xr.Dataset(
        {
            "x": (("chain", "draw"), np.asarray(x_values, dtype=float)),
            "sigma_rate_rel": (
                ("chain", "draw"),
                np.asarray(sigma_values, dtype=float),
            ),
        }
    )
    return xr.DataTree.from_dict({"/posterior": posterior})


def _specs():
    return {
        "x": {"distribution": "normal", "mu": 0.0, "sigma": 1.0},
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": 0.18,
            "log_sd": 0.75,
        },
    }


def test_validation_posterior_shift_reports_direction_and_scale():
    full = _inference_data(
        [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]],
        [[0.15, 0.16, 0.17], [0.14, 0.15, 0.16]],
    )
    validation = _inference_data(
        [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
        [[0.16, 0.17, 0.18], [0.15, 0.16, 0.17]],
    )

    result = summarize_validation_posterior_shift(full, validation, _specs())
    x_row = result.loc[result["parameter"] == "x"].iloc[0]

    assert x_row["median_shift"] > 0
    assert x_row["median_shift_in_full_sd"] > 0
    assert 0.0 <= x_row["hdi95_overlap_fraction_of_full"] <= 1.0


def test_validation_parameter_overlay_writes_figure(tmp_path):
    full = _inference_data(
        [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]],
        [[0.15, 0.16, 0.17], [0.14, 0.15, 0.16]],
    )
    validation = _inference_data(
        [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
        [[0.16, 0.17, 0.18], [0.15, 0.16, 0.17]],
    )
    output_path = tmp_path / "posterior_vs_full.png"

    plot_validation_parameter_posteriors(
        full,
        validation,
        _specs(),
        output_path,
        context_label="LOMO Ag50Pd50",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_heldout_pit_conditions_writes_material_grid(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [0.5, 0.5, 0.5, 0.5],
            "CO_mole_fraction": [0.1, 0.1, 0.1, 0.1],
            "replicate": ["A", "A", "B", "B"],
            "E_V_SHE": [0.0, 0.1, 0.0, 0.1],
            "heldout_pit": [0.2, 0.3, 0.4, 0.5],
        }
    )
    output_path = tmp_path / "heldout_pit.png"

    plot_heldout_pit_conditions(
        frame,
        output_path,
        context_label="LOMO Ag50Pd50",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
