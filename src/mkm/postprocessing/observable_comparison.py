from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentalObservableComparison:
    pooled: pd.DataFrame
    by_chain: pd.DataFrame
    chain_spread: pd.DataFrame


def build_experimental_observable_comparison(
    pooled,
    by_chain,
    experimental,
    key_columns,
    observed_column,
    observed_sd_column,
):
    experimental_columns = key_columns + [observed_column, observed_sd_column]

    pooled_result = pooled.merge(
        experimental[experimental_columns],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    pooled_result["residual_q50"] = pooled_result["q50"] - pooled_result[observed_column]
    pooled_result["abs_residual_q50"] = np.abs(pooled_result["residual_q50"])
    pooled_result["standardized_residual_q50"] = pooled_result["residual_q50"] / pooled_result[observed_sd_column]
    pooled_result["experimental_value_inside_posterior_95"] = (
        (pooled_result[observed_column] >= pooled_result["q025"])
        & (pooled_result[observed_column] <= pooled_result["q975"])
    )

    chain_result = by_chain.merge(
        experimental[experimental_columns],
        on=key_columns,
        how="inner",
        validate="many_to_one",
    )

    chain_spread = (
        chain_result.groupby(key_columns, dropna=False)["q50"]
        .agg(chain_q50_min="min", chain_q50_max="max")
        .reset_index()
    )
    chain_spread["chain_q50_range"] = chain_spread["chain_q50_max"] - chain_spread["chain_q50_min"]
    chain_spread = chain_spread.merge(
        experimental[key_columns + [observed_sd_column]],
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    chain_spread["chain_q50_range_over_exp_sd"] = (
        chain_spread["chain_q50_range"] / chain_spread[observed_sd_column]
    )

    return ExperimentalObservableComparison(
        pooled=pooled_result,
        by_chain=chain_result,
        chain_spread=chain_spread,
    )


def summarize_experimental_observable(name, comparison):
    pooled = comparison.pooled
    spread = comparison.chain_spread

    standardized = pooled["standardized_residual_q50"].replace([np.inf, -np.inf], np.nan).dropna()
    chain_scaled = spread["chain_q50_range_over_exp_sd"].replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "observable": name,
        "n_points": len(pooled),
        "median_abs_residual_q50": float(pooled["abs_residual_q50"].median()),
        "median_abs_standardized_residual_q50": float(standardized.abs().median()),
        "posterior_95_contains_experimental_mean": float(
            pooled["experimental_value_inside_posterior_95"].mean()
        ),
        "median_chain_q50_range_over_exp_sd": float(chain_scaled.median()),
        "p95_chain_q50_range_over_exp_sd": float(chain_scaled.quantile(0.95)),
        "max_chain_q50_range_over_exp_sd": float(chain_scaled.max()),
    }