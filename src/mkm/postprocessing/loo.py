from dataclasses import dataclass

import arviz_stats as azs
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOOModelDiagnostics:
    loo_result: object
    summary: pd.DataFrame
    pointwise: pd.DataFrame


_LOO_REFF_EXCLUDED_DIMS = frozenset({"model_point", "observation", "setup"})


def _finite_reff_variable_names(posterior):
    names = []

    for name, variable in posterior.data_vars.items():
        dims = set(variable.dims)

        if not {"chain", "draw"}.issubset(dims):
            continue

        # Exclude large pointwise/conditional deterministics. In composition fits these can
        # legitimately contain -inf for an impossible pathway, e.g. BF on Pd100.
        if dims & _LOO_REFF_EXCLUDED_DIMS:
            continue

        values = np.asarray(variable, dtype=float)
        if not np.all(np.isfinite(values)):
            continue

        names.append(name)

    return tuple(names)


def compute_loo_reff(inference_data):
    """Estimate relative MCMC efficiency from finite sampling-level posterior variables.

    arviz-stats otherwise computes ESS over every posterior variable. That is inappropriate
    for this repository because valid pathway log-rate deterministics may equal -inf when a
    pathway is structurally absent at a composition (for example BF on Pd100).
    """
    if not hasattr(inference_data, "posterior"):
        return None

    posterior = inference_data.posterior

    if "chain" not in posterior.sizes or "draw" not in posterior.sizes:
        return None

    n_chains = int(posterior.sizes["chain"])
    n_draws = int(posterior.sizes["draw"])

    if n_chains <= 1:
        return 1.0

    names = _finite_reff_variable_names(posterior)
    if not names:
        raise ValueError("Could not identify finite posterior variables for PSIS-LOO relative efficiency.")

    ess_values = []

    for name in names:
        variable = posterior[name]
        ess = variable.azstats.ess(method="mean")
        ess_values.append(np.asarray(ess, dtype=float).reshape(-1))

    ess_values = np.hstack(ess_values)

    finite_ess = ess_values[np.isfinite(ess_values)]
    if finite_ess.size == 0:
        raise ValueError("PSIS-LOO relative-efficiency calculation produced no finite ESS values.")

    n_samples = n_chains * n_draws
    reff = float(np.mean(finite_ess) / n_samples)

    if not np.isfinite(reff) or reff <= 0:
        raise ValueError(f"Invalid PSIS-LOO relative efficiency: {reff}.")

    return reff


def compute_loo_diagnostics(
    inference_data,
    observations,
    model_name=None,
    var_name="ln_rate_observed",
):
    reff = compute_loo_reff(inference_data)

    loo_kwargs = {
        "var_name": var_name,
        "pointwise": True,
    }
    if reff is not None:
        loo_kwargs["reff"] = reff

    loo_result = azs.loo(inference_data, **loo_kwargs)
    elpd_i = np.asarray(loo_result.elpd_i, dtype=float).reshape(-1)
    pareto_k = np.asarray(loo_result.pareto_k, dtype=float).reshape(-1)

    if len(elpd_i) != len(observations):
        raise ValueError(
            f"Pointwise ELPD has {len(elpd_i)} values, but there are {len(observations)} observations."
        )
    if len(pareto_k) != len(observations):
        raise ValueError(
            f"Pareto-k has {len(pareto_k)} values, but there are {len(observations)} observations."
        )

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "elpd": float(loo_result.elpd),
                "se": float(loo_result.se),
                "p": float(loo_result.p),
                "n_samples": int(loo_result.n_samples),
                "n_data_points": int(loo_result.n_data_points),
                "reff": reff,
                "good_k": float(loo_result.good_k),
                "max_pareto_k": float(np.max(pareto_k)),
                "n_pareto_k_above_good_k": int(np.sum(pareto_k > float(loo_result.good_k))),
                "warning": bool(loo_result.warning),
            }
        ]
    )

    pointwise = observations.copy()
    pointwise["elpd_loo"] = elpd_i
    pointwise["pareto_k"] = pareto_k
    pointwise["good_k"] = float(loo_result.good_k)
    pointwise["pareto_k_above_good_k"] = pointwise["pareto_k"] > pointwise["good_k"]

    return LOOModelDiagnostics(
        loo_result=loo_result,
        summary=summary,
        pointwise=pointwise,
    )
