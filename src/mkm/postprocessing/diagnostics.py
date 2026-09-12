import arviz_stats as azs
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import lognorm, norm, truncnorm


COVERAGE_VARIABLES = (
    "theta_CO",
    "theta_CO_site_occupation",
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

ERROR_VARIABLES = (
    "sigma_rate_abs",
    "sigma_rate_rel",
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
    """Summarize draws over the leading sample axis using 80% and 95% HDIs."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Posterior summary input must contain at least one sample.")

    hdi80_lower, hdi80_upper = highest_density_interval(values, prob=0.80)
    hdi95_lower, hdi95_upper = highest_density_interval(values, prob=0.95)
    if values.shape[0] > 1:
        sd = np.std(values, axis=0, ddof=1)
    else:
        sd = np.full(values.shape[1:], np.nan)
    return {
        "mean": np.mean(values, axis=0),
        "sd": sd,
        "median": np.median(values, axis=0),
        "hdi80_lower": hdi80_lower,
        "hdi80_upper": hdi80_upper,
        "hdi95_lower": hdi95_lower,
        "hdi95_upper": hdi95_upper,
    }


def summarize_scalar_samples(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    summary = summarize_samples(values[:, None])
    return {
        "mean": float(summary["mean"][0]),
        "sd": float(summary["sd"][0]),
        "median": float(summary["median"][0]),
        "hdi80_lower": float(summary["hdi80_lower"][0]),
        "hdi80_upper": float(summary["hdi80_upper"][0]),
        "hdi95_lower": float(summary["hdi95_lower"][0]),
        "hdi95_upper": float(summary["hdi95_upper"][0]),
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
        rv = truncnorm(
            a=(lower - mu) / sigma,
            b=(upper - mu) / sigma,
            loc=mu,
            scale=sigma,
        )
        return {
            "prior_mean": float(rv.mean()),
            "prior_sd": float(rv.std()),
            "prior_q025": float(rv.ppf(0.025)),
            "prior_q50": float(rv.ppf(0.50)),
            "prior_q975": float(rv.ppf(0.975)),
            "lower": lower,
            "upper": upper,
        }
    if distribution == "lognormal":
        median = float(spec["median"])
        log_sd = float(spec["log_sd"])
        rv = lognorm(s=log_sd, scale=median)
        return {
            "prior_mean": float(rv.mean()),
            "prior_sd": float(rv.std()),
            "prior_q025": float(rv.ppf(0.025)),
            "prior_q50": median,
            "prior_q975": float(rv.ppf(0.975)),
            "lower": 0.0,
            "upper": np.inf,
        }
    raise ValueError(f"Unsupported prior distribution '{distribution}'.")


def _coordinate_label(values, dim, index):
    if dim in values.coords and values.coords[dim].ndim == 1:
        return str(values.coords[dim].values[index])
    return str(index)


def _parameter_components(posterior, name):
    values = posterior[name]
    missing_sample_dims = [dim for dim in ("chain", "draw") if dim not in values.dims]
    if missing_sample_dims:
        raise ValueError(
            f"Posterior parameter '{name}' is missing sample dimensions: {missing_sample_dims}."
        )
    extra_dims = tuple(dim for dim in values.dims if dim not in {"chain", "draw"})
    if not extra_dims:
        yield name, values.transpose("chain", "draw")
        return
    shape = tuple(values.sizes[dim] for dim in extra_dims)
    for index in np.ndindex(shape):
        indexers = dict(zip(extra_dims, index, strict=True))
        labels = [
            f"{dim}={_coordinate_label(values, dim, dim_index)}"
            for dim, dim_index in zip(extra_dims, index, strict=True)
        ]
        label = f"{name}[{','.join(labels)}]"
        component = values.isel(indexers, drop=True).transpose("chain", "draw")
        yield label, component


def _scalar_statistic(value):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1:
        raise ValueError("Expected a scalar sampling diagnostic.")
    return float(array[0])


def build_posterior_parameter_summary(inference_data, parameter_specs):
    """Combine posterior estimates, prior contraction, R-hat, and ESS in one table."""
    posterior = inference_data.posterior
    missing = [name for name in parameter_specs if name not in posterior]
    if missing:
        raise ValueError(f"Posterior is missing configured parameters: {missing}.")
    records = []
    for name, spec in parameter_specs.items():
        prior = prior_statistics(spec)
        for label, component in _parameter_components(posterior, name):
            chain_draw = np.asarray(component, dtype=float)
            samples = chain_draw.reshape(-1)
            posterior_summary = summarize_scalar_samples(samples)
            if chain_draw.shape[0] >= 2:
                rhat = _scalar_statistic(azs.rhat(chain_draw, chain_axis=0, draw_axis=1))
            else:
                rhat = np.nan
            record = {
                "parameter": label,
                "variable": name,
                "parameter_type": "error" if name in ERROR_VARIABLES else "physical",
                **posterior_summary,
                "mcse_mean": _scalar_statistic(
                    azs.mcse(chain_draw, method="mean", chain_axis=0, draw_axis=1)
                ),
                "mcse_sd": _scalar_statistic(
                    azs.mcse(chain_draw, method="sd", chain_axis=0, draw_axis=1)
                ),
                "ess_bulk": _scalar_statistic(
                    azs.ess(chain_draw, method="bulk", chain_axis=0, draw_axis=1)
                ),
                "ess_tail": _scalar_statistic(
                    azs.ess(chain_draw, method="tail", prob=(0.05, 0.95), chain_axis=0, draw_axis=1)
                ),
                "rhat": rhat,
                **prior,
            }
            record["sd_ratio_posterior_over_prior"] = record["sd"] / record["prior_sd"]
            prior_width = record["prior_q975"] - record["prior_q025"]
            record["interval95_width_ratio_posterior_hdi_over_prior_central"] = (
                record["hdi95_upper"] - record["hdi95_lower"]
            ) / prior_width
            records.append(record)
    preferred = [
        "parameter",
        "variable",
        "parameter_type",
        "mean",
        "sd",
        "median",
        "hdi80_lower",
        "hdi80_upper",
        "hdi95_lower",
        "hdi95_upper",
        "mcse_mean",
        "mcse_sd",
        "ess_bulk",
        "ess_tail",
        "rhat",
        "prior_mean",
        "prior_sd",
        "prior_q025",
        "prior_q50",
        "prior_q975",
        "lower",
        "upper",
        "sd_ratio_posterior_over_prior",
        "interval95_width_ratio_posterior_hdi_over_prior_central",
    ]
    summary = pd.DataFrame.from_records(records)
    existing = [column for column in preferred if column in summary]
    remainder = [column for column in summary if column not in existing]
    return summary[existing + remainder]


def build_physical_summary(posterior):
    records = []
    for name in (*COVERAGE_VARIABLES, *PATHWAY_FRACTION_VARIABLES):
        if name not in posterior:
            continue
        values = flatten_posterior_samples(posterior[name]).reshape(-1)
        summary = summarize_scalar_samples(values)
        records.append(
            {
                "check_type": "bounded_variable",
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
                "check_type": "balance",
                "variable": "Pd_site_balance",
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
                "check_type": "balance",
                "variable": "Ag_site_balance",
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
                "check_type": "balance",
                "variable": "pathway_fraction_sum",
                "max_abs_error": float(np.max(np.abs(error))),
                "q999_abs_error": float(np.quantile(np.abs(error), 0.999)),
            }
        )
    return pd.DataFrame(records)


def build_physical_checks(posterior):
    frames = [build_physical_summary(posterior), build_balance_summary(posterior)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
