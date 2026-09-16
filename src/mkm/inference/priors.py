"""Prior validation, PyMC construction, and numerical summaries."""

import numpy as np
import pymc as pm
from scipy.stats import lognorm, norm, truncnorm


def _require_finite(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"'{name}' must be finite.")
    return value


def validate_prior_spec(name, spec):
    distribution = spec["distribution"]

    if distribution == "normal":
        mu = _require_finite(spec["mu"], f"{name}.mu")
        sigma = _require_finite(spec["sigma"], f"{name}.sigma")
        if sigma <= 0:
            raise ValueError(f"Prior sigma for '{name}' must be positive.")
        return {"distribution": distribution, "mu": mu, "sigma": sigma}

    if distribution == "uniform":
        lower = _require_finite(spec["lower"], f"{name}.lower")
        upper = _require_finite(spec["upper"], f"{name}.upper")
        if lower >= upper:
            raise ValueError(f"Uniform prior for '{name}' requires lower < upper.")
        return {"distribution": distribution, "lower": lower, "upper": upper}

    if distribution == "truncated_normal":
        mu = _require_finite(spec["mu"], f"{name}.mu")
        sigma = _require_finite(spec["sigma"], f"{name}.sigma")
        if sigma <= 0:
            raise ValueError(f"Prior sigma for '{name}' must be positive.")
        lower = float(spec.get("lower", -np.inf))
        upper = float(spec.get("upper", np.inf))
        if np.isnan(lower) or np.isnan(upper) or lower >= upper:
            raise ValueError(f"Truncated-normal prior for '{name}' requires lower < upper.")
        return {"distribution": distribution, "mu": mu, "sigma": sigma, "lower": lower, "upper": upper}

    if distribution == "lognormal":
        median = _require_finite(spec["median"], f"{name}.median")
        log_sd = _require_finite(spec["log_sd"], f"{name}.log_sd")
        if median <= 0:
            raise ValueError(f"Lognormal prior median for '{name}' must be positive.")
        if log_sd <= 0:
            raise ValueError(f"Lognormal prior log_sd for '{name}' must be positive.")
        return {"distribution": distribution, "median": median, "log_sd": log_sd}

    raise ValueError(f"Unsupported prior distribution '{distribution}' for '{name}'.")


def build_prior(name, spec):
    spec = validate_prior_spec(name, spec)
    distribution = spec["distribution"]
    if distribution == "normal":
        return pm.Normal(name, mu=spec["mu"], sigma=spec["sigma"])
    if distribution == "uniform":
        return pm.Uniform(name, lower=spec["lower"], upper=spec["upper"])
    if distribution == "truncated_normal":
        return pm.TruncatedNormal(
            name,
            mu=spec["mu"],
            sigma=spec["sigma"],
            lower=spec["lower"],
            upper=spec["upper"],
        )
    if distribution == "lognormal":
        return pm.LogNormal(name, mu=np.log(spec["median"]), sigma=spec["log_sd"])
    raise AssertionError("validated prior distribution was not handled")


def build_named_priors(parameter_specs):
    return {name: build_prior(name=name, spec=spec) for name, spec in parameter_specs.items()}


def prior_support(spec):
    spec = validate_prior_spec("prior", spec)
    distribution = spec["distribution"]
    if distribution == "uniform":
        return spec["lower"], spec["upper"]
    if distribution == "truncated_normal":
        return spec["lower"], spec["upper"]
    if distribution == "lognormal":
        return 0.0, np.inf
    return -np.inf, np.inf


def prior_pdf(x, spec):
    spec = validate_prior_spec("prior", spec)
    x = np.asarray(x, dtype=float)
    distribution = spec["distribution"]
    if distribution == "normal":
        return norm.pdf(x, loc=spec["mu"], scale=spec["sigma"])
    if distribution == "uniform":
        return np.where(
            (x >= spec["lower"]) & (x <= spec["upper"]),
            1.0 / (spec["upper"] - spec["lower"]),
            0.0,
        )
    if distribution == "truncated_normal":
        a = (spec["lower"] - spec["mu"]) / spec["sigma"]
        b = (spec["upper"] - spec["mu"]) / spec["sigma"]
        return truncnorm.pdf(x, a=a, b=b, loc=spec["mu"], scale=spec["sigma"])
    if distribution == "lognormal":
        return lognorm.pdf(x, s=spec["log_sd"], scale=spec["median"])
    raise AssertionError("validated prior distribution was not handled")
