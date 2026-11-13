
"""Common potential-grid construction and log-domain PCHIP interpolation."""

import numpy as np
from scipy.interpolate import PchipInterpolator


def build_potential_grid(lower, upper, spacing, origin):
    """Return grid indices and potentials inside the supplied interval."""
    spacing = float(spacing)
    origin = float(origin)

    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Analysis-grid spacing must be finite and positive.")

    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Potential-grid bounds must be finite.")

    first = int(np.ceil((lower - origin - 1e-10) / spacing))
    last = int(np.floor((upper - origin + 1e-10) / spacing))

    if first > last:
        raise ValueError("No analysis-grid points inside the potential interval.")

    indices = np.arange(first, last + 1, dtype=int)
    return indices, origin + indices * spacing


def interpolate_log_values(potential, log_values, grid):
    """Interpolate a positive observable in the log domain without extrapolation."""
    potential = np.asarray(potential, dtype=float)
    log_values = np.asarray(log_values, dtype=float)
    grid = np.asarray(grid, dtype=float)

    interpolated = PchipInterpolator(
        potential, log_values, extrapolate=False,
    )(grid)

    if not np.isfinite(interpolated).all():
        raise ValueError("Interpolation produced nonfinite log values.")

    return interpolated
