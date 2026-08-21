import numpy as np
import pymc as pm


def _require_finite(value, name):
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"'{name}' must be finite.")

    return value


def build_prior(name, spec):
    distribution = spec["distribution"]

    if distribution == "normal":
        mu = _require_finite(spec["mu"], f"{name}.mu")
        sigma = _require_finite(spec["sigma"], f"{name}.sigma")

        if sigma <= 0:
            raise ValueError(f"Prior sigma for '{name}' must be positive.")

        return pm.Normal(name, mu=mu, sigma=sigma)

    if distribution == "uniform":
        lower = _require_finite(spec["lower"], f"{name}.lower")
        upper = _require_finite(spec["upper"], f"{name}.upper")

        if lower >= upper:
            raise ValueError(f"Uniform prior for '{name}' requires lower < upper.")

        return pm.Uniform(name, lower=lower, upper=upper)

    if distribution == "truncated_normal":
        mu = _require_finite(spec["mu"], f"{name}.mu")
        sigma = _require_finite(spec["sigma"], f"{name}.sigma")

        if sigma <= 0:
            raise ValueError(f"Prior sigma for '{name}' must be positive.")

        lower = spec.get("lower", -np.inf)
        upper = spec.get("upper", np.inf)

        lower = float(lower)
        upper = float(upper)

        if lower >= upper:
            raise ValueError(f"Truncated-normal prior for '{name}' requires lower < upper.")

        return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=lower, upper=upper)

    raise ValueError(f"Unsupported prior distribution '{distribution}' for '{name}'.")


def build_named_priors(parameter_specs):
    return {name: build_prior(name=name, spec=spec) for name, spec in parameter_specs.items()}