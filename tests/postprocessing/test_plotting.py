import numpy as np
import pandas as pd
import xarray as xr

from mkm.postprocessing.plotting import (
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_variable,
)


def test_plot_observation_grid_saves_model_and_predictive_rate_figures(tmp_path):
    frame = pd.DataFrame(
        {
            "model_point_id": [0, 1],
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "rate": [0.4, 0.7],
            "rate_model_hdi95_lower": [0.3, 0.5],
            "rate_model_median": [0.4, 0.7],
            "rate_model_hdi95_upper": [0.6, 0.9],
            "rate_predictive_hdi95_lower": [0.2, 0.4],
            "rate_predictive_median": [0.4, 0.7],
            "rate_predictive_hdi95_upper": [0.8, 1.2],
            "residual": [0.0, 0.0],
        }
    )

    for distribution in ("model", "predictive"):
        output_path = tmp_path / f"{distribution}.png"
        plot_observation_grid(
            frame,
            output_path,
            distribution=distribution,
            y_scale="linear",
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_observation_grid_saves_single_residual_definition(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "residual": [0.1, -0.1],
        }
    )
    output_path = tmp_path / "residual.png"

    plot_observation_grid(frame, output_path, residual=True, y_scale="linear")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_pointwise_variable_saves_figure(tmp_path):
    summary = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "E_V_SHE": [0.2, 0.3],
            "hdi95_lower": [0.2, 0.3],
            "median": [0.4, 0.5],
            "hdi95_upper": [0.6, 0.7],
        }
    )
    output_path = tmp_path / "theta_CO.png"

    plot_pointwise_variable(summary, "theta_CO", output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_parameter_posteriors_combines_physical_and_named_error_components(tmp_path):
    posterior = xr.Dataset(
        {
            "x": (("chain", "draw"), np.array([[0.8, 0.9, 1.0], [1.0, 1.1, 1.2]])),
            "sigma_rate_rel": (
                ("chain", "draw", "material"),
                np.array(
                    [
                        [[0.15, 0.20], [0.16, 0.21], [0.17, 0.22]],
                        [[0.14, 0.19], [0.15, 0.20], [0.16, 0.21]],
                    ]
                ),
            ),
        },
        coords={"material": ["Ag10Pd90", "Pd100"]},
    )
    specs = {
        "x": {"distribution": "normal", "mu": 0.0, "sigma": 2.0},
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": 0.18,
            "log_sd": 0.75,
        },
    }
    output_path = tmp_path / "parameters.png"

    plot_parameter_posteriors(posterior, specs, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
