import numpy as np
import pandas as pd
import arviz_stats as azs
import xarray as xr
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

HDI_PROB = 0.95


def flatten_posterior_samples(values):
    values = np.asarray(values)

    if values.ndim < 2:
        raise ValueError("Posterior variable must contain chain and draw dimensions.")

    return values.reshape((-1, *values.shape[2:]))


def highest_density_interval(values, prob=HDI_PROB):
    """Return the HDI over the leading sample axis."""
    values = np.asarray(values, dtype=float)

    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("HDI input must contain at least one sample.")

    dims = ("sample", *(f"dim_{index}" for index in range(values.ndim - 1)))
    data = xr.DataArray(values, dims=dims)
    hdi = azs.hdi(data, prob=float(prob), dim="sample")

    lower = np.asarray(hdi.sel(ci_bound="lower"), dtype=float)
    upper = np.asarray(hdi.sel(ci_bound="upper"), dtype=float)
    return lower, upper


def summarize_samples(values):
    """Summarize draws over the leading sample axis using a 95% HDI."""
    values = np.asarray(values, dtype=float)

    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Posterior summary input must contain at least one sample.")

    hdi_lower, hdi_upper = highest_density_interval(values, prob=HDI_PROB)
    sd = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.full(values.shape[1:], np.nan)

    return {
        "mean": np.mean(values, axis=0),
        "sd": sd,
        "median": np.median(values, axis=0),
        "hdi95_lower": hdi_lower,
        "hdi95_upper": hdi_upper,
    }


def summarize_scalar_samples(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    summary = summarize_samples(values[:, None])

    return {
        "mean": float(summary["mean"][0]),
        "sd": float(summary["sd"][0]),
        "median": float(summary["median"][0]),
        "hdi95_lower": float(summary["hdi95_lower"][0]),
        "hdi95_upper": float(summary["hdi95_upper"][0]),
    }


def build_posterior_parameter_summary(posterior, parameter_specs):
    """Summarize configured scalar physical parameters from posterior draws."""
    records = []

    for name in parameter_specs:
        if name not in posterior:
            raise ValueError(f"Posterior is missing configured parameter '{name}'.")

        values = posterior[name].squeeze(drop=True)
        extra_dims = set(values.dims) - {"chain", "draw"}
        if extra_dims:
            raise ValueError(
                f"Physical parameter '{name}' is not scalar; remaining dimensions: {sorted(extra_dims)}."
            )

        records.append({"parameter": name, **summarize_scalar_samples(values)})

    return pd.DataFrame(records)


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


def build_parameter_contraction(posterior, config, material, model_name, parameter_specs=None):
    if parameter_specs is None:
        parameter_specs = config["prior_profiles"][material][model_name]["parameters"]

    records = []

    for name, spec in parameter_specs.items():
        if name not in posterior:
            raise ValueError(f"Posterior is missing configured parameter '{name}'.")

        prior = prior_statistics(spec)
        post = summarize_scalar_samples(posterior[name])

        prior_width = prior["prior_q975"] - prior["prior_q025"]
        posterior_width = post["hdi95_upper"] - post["hdi95_lower"]

        records.append(
            {
                "parameter": name,
                **prior,
                "posterior_mean": post["mean"],
                "posterior_sd": post["sd"],
                "posterior_median": post["median"],
                "posterior_hdi95_lower": post["hdi95_lower"],
                "posterior_hdi95_upper": post["hdi95_upper"],
                "sd_ratio_posterior_over_prior": post["sd"] / prior["prior_sd"],
                "interval95_width_ratio_posterior_hdi_over_prior_central": posterior_width / prior_width,
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

        values = flatten_posterior_samples(posterior[name]).reshape(-1)
        summary = summarize_scalar_samples(values)

        records.append(
            {
                "variable": name,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "median": summary["median"],
                "hdi95_lower": summary["hdi95_lower"],
                "hdi95_upper": summary["hdi95_upper"],
                "fraction_below_zero": float(np.mean(values < -1e-10)),
                "fraction_above_one": float(np.mean(values > 1.0 + 1e-10)),
            }
        )

    return pd.DataFrame(records)


def build_balance_summary(posterior):
    records = []

    if all(name in posterior for name in ("theta_CO", "theta_OH_Pd", "theta_empty_Pd")):
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

    if all(name in posterior for name in ("theta_OH_Ag", "theta_empty_Ag")):
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

    fraction_names = [name for name in PATHWAY_FRACTION_VARIABLES if name in posterior]

    if fraction_names:
        total = sum(flatten_posterior_samples(posterior[name]) for name in fraction_names)
        error = total - 1.0

        records.append(
            {
                "balance": "pathway_fraction_sum",
                "max_abs_error": float(np.max(np.abs(error))),
                "q999_abs_error": float(np.quantile(np.abs(error), 0.999)),
            }
        )

    return pd.DataFrame(records)
