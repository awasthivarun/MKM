import numpy as np
import arviz_stats as azs
import xarray as xr

from mkm.postprocessing.diagnostics import highest_density_interval, summarize_samples


def test_summarize_samples_uses_95_percent_hdi():
    values = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [20.0]])
    summary = summarize_samples(values)

    data = xr.DataArray(values, dims=("sample", "point"))
    expected = azs.hdi(data, prob=0.95, dim="sample")

    np.testing.assert_allclose(
        summary["hdi95_lower"],
        np.asarray(expected.sel(ci_bound="lower")),
    )
    np.testing.assert_allclose(
        summary["hdi95_upper"],
        np.asarray(expected.sel(ci_bound="upper")),
    )
    np.testing.assert_allclose(summary["median"], np.median(values, axis=0))


def test_rate_space_hdi_is_computed_after_exponentiation():
    log_draws = np.linspace(-4.0, 2.0, 101)[:, None]
    rate_draws = np.exp(log_draws)

    log_lower, log_upper = highest_density_interval(log_draws)
    rate_lower, rate_upper = highest_density_interval(rate_draws)

    exponentiated_log_hdi = np.array([np.exp(log_lower[0]), np.exp(log_upper[0])])
    direct_rate_hdi = np.array([rate_lower[0], rate_upper[0]])

    assert not np.allclose(direct_rate_hdi, exponentiated_log_hdi)
