from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp, ndtr

from mkm.postprocessing.predictions import build_observation_distribution_draws


@dataclass(frozen=True)
class LOOCalibration:
    pointwise: pd.DataFrame
    summary: pd.DataFrame


def _extract_loo_log_weights(loo_result, var_name, expected_shape):
    log_weights = loo_result.log_weights

    if log_weights is None:
        raise ValueError("LOO result does not contain PSIS log weights.")

    if isinstance(log_weights, xr.Dataset):
        if var_name in log_weights:
            log_weights = log_weights[var_name]
        elif len(log_weights.data_vars) == 1:
            log_weights = log_weights[next(iter(log_weights.data_vars))]
        else:
            raise ValueError(
                f"LOO log-weight dataset does not contain '{var_name}' and has multiple variables: "
                f"{list(log_weights.data_vars)}"
            )

    if isinstance(log_weights, xr.DataArray):
        if "chain" in log_weights.dims and "draw" in log_weights.dims:
            other_dims = [dim for dim in log_weights.dims if dim not in {"chain", "draw"}]
            log_weights = log_weights.transpose("chain", "draw", *other_dims)

    values = np.asarray(log_weights, dtype=float)

    if values.shape != expected_shape:
        expected_size = int(np.prod(expected_shape))
        if values.size != expected_size:
            raise ValueError(
                f"LOO log weights have shape {values.shape}, but expected {expected_shape}."
            )
        values = values.reshape(expected_shape)

    return values


def compute_normal_loo_pit(
    inference_data,
    loo_result,
    observations,
    inputs,
    likelihood_name,
    var_name="ln_rate_observed",
):
    observed_ln_rate = None
    if "ln_rate" in observations.columns:
        observed_ln_rate = observations["ln_rate"].to_numpy(dtype=float)

    draws = build_observation_distribution_draws(
        inference_data=inference_data,
        inputs=inputs,
        likelihood_name=likelihood_name,
        observed_ln_rate=observed_ln_rate,
    )

    if likelihood_name == "rate_normal":
        observed = observations["rate_s_inv"].to_numpy(dtype=float)
    else:
        observed = observations["ln_rate"].to_numpy(dtype=float)

    if draws.conditional_mu.shape[-1] != len(observed):
        raise ValueError(
            f"Posterior contains {draws.conditional_mu.shape[-1]} observations, "
            f"but observation table contains {len(observed)}."
        )

    z = (observed[None, None, :] - draws.conditional_mu) / draws.conditional_sigma
    conditional_cdf = ndtr(z)

    log_weights = _extract_loo_log_weights(
        loo_result=loo_result,
        var_name=var_name,
        expected_shape=conditional_cdf.shape,
    )
    log_weights = log_weights - logsumexp(log_weights, axis=(0, 1), keepdims=True)
    weights = np.exp(log_weights)

    loo_pit = np.sum(weights * conditional_cdf, axis=(0, 1))

    pointwise = observations.copy()
    pointwise["loo_pit"] = loo_pit
    pointwise["loo_pit_centered"] = loo_pit - 0.5
    pointwise["loo_pit_tail"] = (loo_pit < 0.05) | (loo_pit > 0.95)

    summary = pd.DataFrame(
        [
            {
                "n_points": len(loo_pit),
                "mean_loo_pit": float(np.mean(loo_pit)),
                "median_loo_pit": float(np.median(loo_pit)),
                "sd_loo_pit": float(np.std(loo_pit, ddof=1)) if len(loo_pit) > 1 else np.nan,
                "uniform_reference_sd": float(1.0 / np.sqrt(12.0)),
                "fraction_below_0p05": float(np.mean(loo_pit < 0.05)),
                "fraction_above_0p95": float(np.mean(loo_pit > 0.95)),
                "fraction_outside_0p05_0p95": float(np.mean((loo_pit < 0.05) | (loo_pit > 0.95))),
            }
        ]
    )

    return LOOCalibration(pointwise=pointwise, summary=summary)


def build_loo_pit_datatree(loo_pit, variable_name="ln_rate_observed"):
    values = np.asarray(loo_pit, dtype=float).reshape(-1)
    dataset = xr.Dataset({variable_name: ("observation", values)})
    return xr.DataTree.from_dict({"/loo_pit": dataset})