"""Shared AgPd/PtRu bounded-slope scaling preserves endpoint constraints."""

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt

from mkm.models.composition_bounds import bounded_linear_slope_scale


@pytest.mark.parametrize("reference,x_max", [(0.5, 1.0), (0.5, 0.5), (0.25, 0.5)])
def test_bounded_slopes_respect_both_endpoints(reference, x_max):
    value, lower, upper = 0.4, 0.0, 1.0
    scale = bounded_linear_slope_scale(value, lower, upper, reference, 0.0, x_max, minimum=min)
    for x in (0.0, x_max):
        assert lower - 1e-12 <= value - scale * (x - reference) <= upper + 1e-12
        assert lower - 1e-12 <= value + scale * (x - reference) <= upper + 1e-12


def test_bounded_slope_numpy_and_symbolic_agree():
    values = np.array([0.2, 0.5, 0.8])
    expected = bounded_linear_slope_scale(values, 0.0, 1.0, 0.5, 0.0, 1.0, minimum=np.minimum)
    variable = pt.dvector("values")
    evaluated = bounded_linear_slope_scale(variable, 0.0, 1.0, 0.5, 0.0, 1.0, minimum=pt.minimum)
    result = pytensor.function([variable], evaluated)(values)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, 2.0 * np.minimum(values, 1.0 - values))


def test_bounded_slope_rejects_invalid_reference():
    with pytest.raises(ValueError, match="reference"):
        bounded_linear_slope_scale(0.5, 0.0, 1.0, 1.1, 0.0, 1.0, minimum=min)
