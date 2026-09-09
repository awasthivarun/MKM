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

    if isinstance(log_weights, xr.DataArray) and {"chain", "draw"}.issubset(log_weights.dims):
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
    var_name="rate_observed",
):
    draws = build_observation_distribution_draws(
        inference_data=inference_data,
        inputs=inputs,
        sample_predictive=False,
    )
    observed = observations["rate_s_inv"].to_numpy(dtype=float)

    if draws.model_rate.shape[-1] != len(observed):
        raise ValueError(
            f"Posterior contains {draws.model_rate.shape[-1]} observations, "
            f"but observation table contains {len(observed)}."
        )

    model_rate = np.asarray(draws.model_rate, dtype=float)
    sigma_rate = np.asarray(draws.sigma_rate, dtype=float)
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed rates contain non-finite values.")
    if not np.all(np.isfinite(model_rate)):
        raise ValueError("Posterior model-rate draws contain non-finite values.")
    if not np.all(np.isfinite(sigma_rate)) or np.any(sigma_rate <= 0.0):
        raise ValueError("Posterior rate-scale draws must be finite and strictly positive.")

    z = (observed[None, None, :] - model_rate) / sigma_rate
    normal_cdf = ndtr(z)
    if not np.all(np.isfinite(normal_cdf)):
        raise ValueError("Normal CDF values for LOO-PIT contain non-finite values.")

    log_weights = _extract_loo_log_weights(
        loo_result=loo_result,
        var_name=var_name,
        expected_shape=normal_cdf.shape,
    )
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("PSIS log weights contain NaN or +inf values.")
    has_finite_weight = np.any(np.isfinite(log_weights), axis=(0, 1))
    if not np.all(has_finite_weight):
        bad = np.flatnonzero(~has_finite_weight)
        raise ValueError(
            "PSIS log weights contain no finite draw for observation index/indices "
            f"{bad[:10].tolist()}{'...' if len(bad) > 10 else ''}."
        )

    log_norm = logsumexp(log_weights, axis=(0, 1), keepdims=True)
    if not np.all(np.isfinite(log_norm)):
        raise ValueError("PSIS log-weight normalization produced non-finite values.")

    normalized_log_weights = log_weights - log_norm
    weights = np.exp(normalized_log_weights)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Normalized PSIS weights contain non-finite values.")

    weight_sums = np.sum(weights, axis=(0, 1))
    if not np.all(np.isfinite(weight_sums)) or np.any(weight_sums <= 0.0):
        raise ValueError("Normalized PSIS weights have invalid observation-wise sums.")
    if not np.allclose(weight_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Normalized PSIS weights do not sum to one within numerical tolerance.")

    loo_pit = np.sum(weights * normal_cdf, axis=(0, 1))
    if not np.all(np.isfinite(loo_pit)):
        raise ValueError("LOO-PIT calculation produced non-finite values.")
    if np.any((loo_pit < -1e-12) | (loo_pit > 1.0 + 1e-12)):
        raise ValueError("LOO-PIT calculation produced values outside [0, 1].")
    loo_pit = np.clip(loo_pit, 0.0, 1.0)

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
                "sd_loo_pit": (
                    float(np.std(loo_pit, ddof=1)) if len(loo_pit) > 1 else np.nan
                ),
                "uniform_reference_sd": float(1.0 / np.sqrt(12.0)),
                "fraction_below_0p05": float(np.mean(loo_pit < 0.05)),
                "fraction_above_0p95": float(np.mean(loo_pit > 0.95)),
                "fraction_outside_0p05_0p95": float(
                    np.mean((loo_pit < 0.05) | (loo_pit > 0.95))
                ),
            }
        ]
    )
    return LOOCalibration(pointwise=pointwise, summary=summary)


def build_loo_pit_datatree(loo_pit, variable_name="rate_observed"):
    values = np.asarray(loo_pit, dtype=float).reshape(-1)
    dataset = xr.Dataset({variable_name: ("observation", values)})
    return xr.DataTree.from_dict({"/loo_pit": dataset})
