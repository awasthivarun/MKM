import numpy as np
import pandas as pd

from mkm.likelihood_diagnostics import (
    calculate_centered_log_residuals,
    calculate_pooled_log_rate_sd,
    summarize_point_dispersion,
)
from mkm.model_data import build_model_data


def _make_diagnostic_data():
    records = []
    log_rates = {0: [0.8, 1.0, 1.2], 1: [1.6, 2.0, 2.4]}

    for grid_index, values in log_rates.items():
        for replicate, ln_rate in zip(["A", "B", "C"], values):
            records.append(
                {
                    "material": "test",
                    "C_KOH_M": 0.5,
                    "CO_mole_fraction": 0.01,
                    "replicate": replicate,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": 0.01 * grid_index,
                    "rate_s_inv": np.exp(ln_rate),
                    "ln_rate": ln_rate,
                }
            )

    data = pd.DataFrame(records)

    return build_model_data(selected_replicates=data, electrolyte_concentration_column="C_KOH_M")


def test_point_dispersion_uses_sample_sd():
    model_data = _make_diagnostic_data()
    summary = summarize_point_dispersion(model_data)

    expected = np.array([np.std([0.8, 1.0, 1.2], ddof=1), np.std([1.6, 2.0, 2.4], ddof=1)])

    np.testing.assert_allclose(summary["ln_rate_sd"].to_numpy(), expected)


def test_centered_residuals_sum_to_zero():
    model_data = _make_diagnostic_data()
    residuals = calculate_centered_log_residuals(model_data)
    sums = residuals.groupby("model_point_id")["ln_rate_centered_residual"].sum()

    np.testing.assert_allclose(sums, 0.0, atol=1e-12)


def test_pooled_log_rate_sd():
    model_data = _make_diagnostic_data()
    result = calculate_pooled_log_rate_sd(model_data)

    residuals_1 = np.array([-0.2, 0.0, 0.2])
    residuals_2 = np.array([-0.4, 0.0, 0.4])

    expected = np.sqrt((np.sum(residuals_1**2) + np.sum(residuals_2**2)) / 4)

    np.testing.assert_allclose(result, expected)