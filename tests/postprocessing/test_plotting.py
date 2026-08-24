import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr

from mkm.postprocessing.plotting import (
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_variable,
)


def test_plot_observation_grid_saves_predictive_figure(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "ln_rate": [-1.0, -0.5],
            "ln_rate_predictive_hdi95_lower": [-1.2, -0.7],
            "ln_rate_predictive_median": [-1.0, -0.5],
            "ln_rate_predictive_hdi95_upper": [-0.8, -0.3],
            "residual_conditional": [0.1, -0.1],
        }
    )
    output_path = tmp_path / "predictive.png"

    frame["rate"] = np.exp(frame["ln_rate"])
    frame["rate_predictive_hdi95_lower"] = np.exp(frame["ln_rate_predictive_hdi95_lower"])
    frame["rate_predictive_median"] = np.exp(frame["ln_rate_predictive_median"])
    frame["rate_predictive_hdi95_upper"] = np.exp(frame["ln_rate_predictive_hdi95_upper"])

    plot_observation_grid(
        observations=frame,
        output_path=output_path,
        residual=False,
        y_scale="log",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_observation_grid_saves_rate_figure(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "rate": [0.4, 0.7],
            "rate_predictive_hdi95_lower": [0.2, 0.4],
            "rate_predictive_median": [0.4, 0.7],
            "rate_predictive_hdi95_upper": [0.8, 1.2],
            "residual_conditional": [0.1, -0.1],
        }
    )
    output_path = tmp_path / "predictive_rate.png"

    plot_observation_grid(
        observations=frame,
        output_path=output_path,
        residual=False,
        y_scale="linear",
        distribution="predictive",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_observation_grid_saves_residual_figure(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "residual_conditional": [0.1, -0.1],
        }
    )
    output_path = tmp_path / "residual.png"

    plot_observation_grid(
        observations=frame,
        output_path=output_path,
        residual=True,
    )

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

    plot_pointwise_variable(
        summary=summary,
        variable_name="theta_CO",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_parameter_posteriors_saves_figure(tmp_path):
    posterior = xr.Dataset(
        {
            "x": (("chain", "draw"), np.array([[0.8, 0.9, 1.0], [1.0, 1.1, 1.2]])),
            "q": (("chain", "draw"), np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])),
        }
    )

    specs = {
        "x": {"distribution": "normal", "mu": 0.0, "sigma": 2.0},
        "q": {"distribution": "uniform", "lower": 0.0, "upper": 1.0},
    }
    output_path = tmp_path / "parameters.png"

    plot_parameter_posteriors(posterior, specs, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
