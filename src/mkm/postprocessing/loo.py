from dataclasses import dataclass

import arviz_stats as azs
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOOModelDiagnostics:
    loo_result: object
    summary: pd.DataFrame
    pointwise: pd.DataFrame


def compute_loo_diagnostics(
    inference_data,
    observations,
    model_name=None,
    var_name="ln_rate_observed",
):
    loo_result = azs.loo(inference_data, var_name=var_name, pointwise=True)

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