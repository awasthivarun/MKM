"""Shared configuration for the linear-rate Normal likelihood."""

from mkm.inference.likelihoods import RATE_NORMAL, available_error_structures


def rate_normal_likelihood_kwargs(config):
    likelihood = config["likelihood"]

    if likelihood.get("name") != RATE_NORMAL:
        raise ValueError(
            f"Configuration must use likelihood '{RATE_NORMAL}', not '{likelihood.get('name')}'."
        )

    expected_form = "sigma_abs + sigma_rel * model_rate"
    if likelihood.get("sigma_form") != expected_form:
        raise ValueError(
            f"Configured sigma form must be '{expected_form}', not '{likelihood.get('sigma_form')}'."
        )

    keys = (
        "sigma_abs_prior_median_s_inv",
        "sigma_abs_prior_log_sd",
        "sigma_rel_prior_median",
        "sigma_rel_prior_log_sd",
    )
    return {key: likelihood[key] for key in keys}


def error_parameter_specs(config):
    likelihood = config["likelihood"]
    return {
        "sigma_rate_abs": {
            "distribution": "lognormal",
            "median": float(likelihood["sigma_abs_prior_median_s_inv"]),
            "log_sd": float(likelihood["sigma_abs_prior_log_sd"]),
        },
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": float(likelihood["sigma_rel_prior_median"]),
            "log_sd": float(likelihood["sigma_rel_prior_log_sd"]),
        },
    }


def validate_configured_error_structure(config, error_structure):
    supported = tuple(config["likelihood"]["supported_error_structures"])
    unknown = set(supported) - set(available_error_structures())
    if unknown:
        raise ValueError(f"Configuration contains unsupported error structures: {sorted(unknown)}.")
    if error_structure not in supported:
        raise ValueError(f"Error structure '{error_structure}' is not enabled. Configured: {supported}.")
    return error_structure