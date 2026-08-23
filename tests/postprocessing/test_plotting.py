import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr

from mkm.postprocessing.plotting import (
    plot_observation_grid,
    plot_pointwise_variable,
    plot_parameter_posteriors,
)

def test_plot_observation_grid_saves_predictive_figure(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [
                1.0,
                1.0,
            ],
            "CO_mole_fraction": [
                0.1,
                0.1,
            ],
            "replicate": [
                "A",
                "A",
            ],
            "E_V_SHE": [
                0.2,
                0.3,
            ],
            "ln_rate": [
                -1.0,
                -0.5,
            ],
            "ln_rate_predictive_q025": [
                -1.2,
                -0.7,
            ],
            "ln_rate_predictive_q50": [
                -1.0,
                -0.5,
            ],
            "ln_rate_predictive_q975": [
                -0.8,
                -0.3,
            ],
            "residual_conditional": [
                0.1,
                -0.1,
            ],
        }
    )

    output_path = tmp_path / "predictive.png"

    plot_observation_grid(
        observations=frame,
        output_path=output_path,
        residual=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_plot_observation_grid_saves_residual_figure(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [
                1.0,
                1.0,
            ],
            "CO_mole_fraction": [
                0.1,
                0.1,
            ],
            "replicate": [
                "A",
                "A",
            ],
            "E_V_SHE": [
                0.2,
                0.3,
            ],
            "residual_conditional": [
                0.1,
                -0.1,
            ],
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

def test_plot_pointwise_variable_saves_figure(
    tmp_path,
):
    summary = pd.DataFrame(
        {
            "electrolyte_concentration_M": [
                1.0,
                1.0,
            ],
            "CO_mole_fraction": [
                0.1,
                0.1,
            ],
            "E_V_SHE": [
                0.2,
                0.3,
            ],
            "q025": [
                0.2,
                0.3,
            ],
            "q50": [
                0.4,
                0.5,
            ],
            "q975": [
                0.6,
                0.7,
            ],
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