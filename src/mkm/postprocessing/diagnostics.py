import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm


COVERAGE_VARIABLES = (
    "theta_CO",
    "theta_OH_Pd",
    "theta_empty_Pd",
    "theta_OH_Ag",
    "theta_empty_Ag",
)

PATHWAY_FRACTION_VARIABLES = (
    "rate_fraction_BF",
    "rate_fraction_ER",
    "rate_fraction_LH",
)

NOISE_VARIABLES = (
    "sigma_ln_rate_material",
    "sigma_ln_rate_setup_material",
)


def flatten_posterior_samples(values):
    values = np.asarray(values)

    if values.ndim < 2:
        raise ValueError("Posterior variable must contain chain and draw dimensions.")

    return values.reshape((-1, *values.shape[2:]))


def summarize_samples(values):
    values = np.asarray(values, dtype=float)

    return {
        "mean": np.mean(values, axis=0),
        "sd": np.std(values, axis=0, ddof=1),
        "q025": np.quantile(values, 0.025, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q975": np.quantile(values, 0.975, axis=0),
    }


def summarize_scalar_samples(values):
    values = np.asarray(values, dtype=float).reshape(-1)

    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.50)),
        "q975": float(np.quantile(values, 0.975)),
    }


def prior_statistics(spec):
    distribution = spec["distribution"]

    if distribution == "normal":
        mu = float(spec["mu"])
        sigma = float(spec["sigma"])

        return {
            "prior_mean": mu,
            "prior_sd": sigma,
            "prior_q025": mu + sigma * norm.ppf(0.025),
            "prior_q50": mu,
            "prior_q975": mu + sigma * norm.ppf(0.975),
            "lower": np.nan,
            "upper": np.nan,
        }

    if distribution == "uniform":
        lower = float(spec["lower"])
        upper = float(spec["upper"])

        return {
            "prior_mean": 0.5 * (lower + upper),
            "prior_sd": (upper - lower) / np.sqrt(12.0),
            "prior_q025": lower + 0.025 * (upper - lower),
            "prior_q50": 0.5 * (lower + upper),
            "prior_q975": lower + 0.975 * (upper - lower),
            "lower": lower,
            "upper": upper,
        }

    if distribution == "truncated_normal":
        mu = float(spec["mu"])
        sigma = float(spec["sigma"])
        lower = float(spec.get("lower", -np.inf))
        upper = float(spec.get("upper", np.inf))

        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        rv = truncnorm(a=a, b=b, loc=mu, scale=sigma)

        return {
            "prior_mean": float(rv.mean()),
            "prior_sd": float(rv.std()),
            "prior_q025": float(rv.ppf(0.025)),
            "prior_q50": float(rv.ppf(0.50)),
            "prior_q975": float(rv.ppf(0.975)),
            "lower": lower,
            "upper": upper,
        }

    raise ValueError(f"Unsupported prior distribution '{distribution}'.")


def build_parameter_contraction(posterior, config, material, model_name):
    parameter_specs = config["prior_profiles"][material][model_name]["parameters"]

    records = []

    for name, spec in parameter_specs.items():
        if name not in posterior:
            raise ValueError(f"Posterior is missing configured parameter '{name}'.")

        prior = prior_statistics(spec)
        post = summarize_scalar_samples(posterior[name])

        prior_width = prior["prior_q975"] - prior["prior_q025"]
        posterior_width = post["q975"] - post["q025"]

        records.append(
            {
                "parameter": name,
                **prior,
                "posterior_mean": post["mean"],
                "posterior_sd": post["sd"],
                "posterior_q025": post["q025"],
                "posterior_q50": post["q50"],
                "posterior_q975": post["q975"],
                "sd_ratio_posterior_over_prior": post["sd"] / prior["prior_sd"],
                "interval_ratio_posterior_over_prior": posterior_width / prior_width,
            }
        )

    return pd.DataFrame(records)


def build_noise_summary(posterior):
    records = []

    for name in NOISE_VARIABLES:
        if name not in posterior:
            continue

        values = flatten_posterior_samples(posterior[name])

        for index in np.ndindex(values.shape[1:]):
            x = values[(slice(None), *index)]
            summary = summarize_scalar_samples(x)

            records.append(
                {
                    "variable": name,
                    "index": str(index),
                    **summary,
                }
            )

    return pd.DataFrame(records)


def build_physical_summary(posterior):
    records = []

    for name in (*COVERAGE_VARIABLES, *PATHWAY_FRACTION_VARIABLES):
        if name not in posterior:
            continue

        values = flatten_posterior_samples(posterior[name])

        records.append(
            {
                "variable": name,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q50": float(np.quantile(values, 0.50)),
                "q975": float(np.quantile(values, 0.975)),
                "fraction_below_zero": float(np.mean(values < -1e-10)),
                "fraction_above_one": float(np.mean(values > 1.0 + 1e-10)),
            }
        )

    return pd.DataFrame(records)


def build_balance_summary(posterior):
    records = []

    if all(
        name in posterior
        for name in ("theta_CO", "theta_OH_Pd", "theta_empty_Pd")
    ):
        balance = (
            flatten_posterior_samples(posterior["theta_CO"])
            + flatten_posterior_samples(posterior["theta_OH_Pd"])
            + flatten_posterior_samples(posterior["theta_empty_Pd"])
        )

        error = balance - 1.0

        records.append(
            {
                "balance": "Pd",
                "max_abs_error": float(np.max(np.abs(error))),
                "q999_abs_error": float(np.quantile(np.abs(error), 0.999)),
            }
        )

    if all(
        name in posterior
        for name in ("theta_OH_Ag", "theta_empty_Ag")
    ):
        balance = (
            flatten_posterior_samples(posterior["theta_OH_Ag"])
            + flatten_posterior_samples(posterior["theta_empty_Ag"])
        )

        error = balance - 1.0

        records.append(
            {
                "balance": "Ag",
                "max_abs_error": float(np.max(np.abs(error))),
                "q999_abs_error": float(np.quantile(np.abs(error), 0.999)),
            }
        )

    fraction_names = [
        name
        for name in PATHWAY_FRACTION_VARIABLES
        if name in posterior
    ]

    if fraction_names:
        total = sum(
            flatten_posterior_samples(posterior[name])
            for name in fraction_names
        )

        error = total - 1.0

        records.append(
            {
                "balance": "pathway_fraction_sum",
                "max_abs_error": float(np.max(np.abs(error))),
                "q999_abs_error": float(np.quantile(np.abs(error), 0.999)),
            }
        )

    return pd.DataFrame(records)