"""Scale composition-dependent slopes to respect bounds throughout the modeled composition domain."""


def bounded_linear_slope_scale(value, lower, upper, x_reference, x_min, x_max, *, minimum):
    """Largest symmetric slope magnitude that keeps value + slope*(x-x_ref) within [lower, upper].

    The caller supplies `minimum` (pt.minimum, np.minimum, or min) to preserve its
    scalar/symbolic array type. This is a mathematical primitive, not a prior.
    """
    if not x_min <= x_reference <= x_max or x_min == x_max:
        raise ValueError("Composition reference must lie in a nonzero composition interval.")
    maximum_distance = max(abs(x_min - x_reference), abs(x_max - x_reference))
    return minimum(value - lower, upper - value) / maximum_distance
