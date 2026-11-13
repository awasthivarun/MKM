
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from mkm.preprocessing.agpd_basic import _build_analysis_grid, _interpolate_log_current
from mkm.preprocessing.interpolation import build_potential_grid, interpolate_log_values
from mkm.preprocessing.validation import DataValidationError


def test_shared_grid_matches_original_index_arithmetic():
    first, grid = build_potential_grid(0.705, 0.735, 0.01, 0.0)

    np.testing.assert_array_equal(first, [71, 72, 73])
    np.testing.assert_allclose(grid, [0.71, 0.72, 0.73], rtol=0, atol=1e-12)


def test_shared_pchip_matches_scipy_reference():
    potential = np.array([0.70, 0.712, 0.724, 0.74])
    ln_rate = np.array([-3.0, -2.5, -2.2, -1.0])
    grid = np.array([0.71, 0.72, 0.73])

    expected = PchipInterpolator(potential, ln_rate, extrapolate=False)(grid)
    actual = interpolate_log_values(potential, ln_rate, grid)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_agpd_wrapper_preserves_decreasing_potential_support():
    potential = np.array([0.74, 0.72, 0.70])
    ln_current = np.array([-1.0, -2.0, -3.0])
    grid = np.array([0.70, 0.71, 0.72, 0.73, 0.74])

    actual = _interpolate_log_current(
        potential, ln_current, grid, condition=("Pd100", 0.5, 0.1), replicate="A",
    )
    expected = PchipInterpolator(
        potential[::-1], ln_current[::-1], extrapolate=False,
    )(grid)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_agpd_wrapper_retains_grid_error_type():
    with pytest.raises(DataValidationError, match="usable overlap"):
        _build_analysis_grid(0.75, 0.70, 0.01, 0.0)


def test_shared_pchip_does_not_extrapolate():
    with pytest.raises(ValueError, match="nonfinite"):
        interpolate_log_values(
            potential=np.array([0.70, 0.72, 0.74]),
            log_values=np.array([-3.0, -2.0, -1.0]),
            grid=np.array([0.69, 0.71, 0.75]),
        )
