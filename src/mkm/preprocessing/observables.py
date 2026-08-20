import numpy as np
import pandas as pd 


from mkm.constants import (
    F_C_mol,
    R_J_mol_K,
)


def calculate_transfer_coefficient(
    potential_V,
    ln_rate,
    temperature_K,
):
    potential_V = np.asarray(
        potential_V,
        dtype=float,
    )
    ln_rate = np.asarray(
        ln_rate,
        dtype=float,
    )

    if potential_V.ndim != 1:
        raise ValueError(
            "Potential must be one-dimensional."
        )

    if ln_rate.ndim != 1:
        raise ValueError(
            "Log rate must be one-dimensional."
        )

    if len(potential_V) != len(ln_rate):
        raise ValueError(
            "Potential and log-rate arrays must have the same length."
        )

    if len(potential_V) < 3:
        raise ValueError(
            "At least three potential points are required "
            "to calculate a transfer coefficient."
        )

    if not np.all(np.isfinite(potential_V)):
        raise ValueError(
            "Potential contains non-finite values."
        )

    if not np.all(np.isfinite(ln_rate)):
        raise ValueError(
            "Log rate contains non-finite values."
        )

    if temperature_K <= 0:
        raise ValueError(
            "Temperature must be greater than zero kelvin."
        )

    if not np.all(np.diff(potential_V) > 0):
        raise ValueError(
            "Potential must be strictly increasing."
        )

    dln_rate_dE = np.gradient(
        ln_rate,
        potential_V,
        edge_order=2,
    )

    alpha = (
        R_J_mol_K
        * temperature_K
        / F_C_mol
        * dln_rate_dE
    )

    return alpha


def add_transfer_coefficients(
    data,
    group_columns,
    temperature_K,
    potential_column="E_V_SHE",
    log_rate_column="ln_rate",
    output_column="alpha",
):
    result = data.copy()

    result[output_column] = np.nan

    grouped = result.groupby(
        group_columns,
        sort=False,
    )

    for _, curve in grouped:
        curve = curve.sort_values(
            potential_column
        )

        alpha = calculate_transfer_coefficient(
            potential_V=curve[
                potential_column
            ].to_numpy(),
            ln_rate=curve[
                log_rate_column
            ].to_numpy(),
            temperature_K=temperature_K,
        )

        result.loc[
            curve.index,
            output_column,
        ] = alpha

    if not np.all(
        np.isfinite(
            result[output_column].to_numpy()
        )
    ):
        raise ValueError(
            "Transfer-coefficient calculation produced "
            "non-finite values."
        )

    return result


def summarize_transfer_coefficients(
    data,
    group_columns,
    replicate_column="replicate",
    alpha_column="alpha",
):
    summary = (
        data
        .groupby(
            group_columns,
            sort=False,
        )
        .agg(
            n_replicates=(
                replicate_column,
                "nunique",
            ),
            alpha_mean=(
                alpha_column,
                "mean",
            ),
            alpha_sd=(
                alpha_column,
                "std",
            ),
        )
        .reset_index()
    )

    if not np.all(
        np.isfinite(
            summary["alpha_mean"].to_numpy()
        )
    ):
        raise ValueError(
            "Mean transfer coefficient contains "
            "non-finite values."
        )

    return summary


def _get_common_grid_indices(
    condition_frames,
    grid_column,
):
    common_indices = None

    for frame in condition_frames:
        indices = set(
            frame[grid_column].to_numpy()
        )

        if common_indices is None:
            common_indices = indices
        else:
            common_indices &= indices

    if not common_indices:
        raise ValueError(
            "Conditions do not share any common analysis-grid points."
        )

    return np.array(
        sorted(common_indices),
        dtype=int,
    )


def _get_value_at_grid_index(
    frame,
    grid_index,
    grid_column,
    value_column,
):
    rows = frame[
        frame[grid_column] == grid_index
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row at "
            f"{grid_column}={grid_index}; found {len(rows)}."
        )

    return rows.iloc[0][value_column]


def calculate_log_slope_order(
    data,
    varying_column,
    varying_values,
    group_columns,
    output_column,
    output_sd_column,
    mean_log_rate_column="ln_rate_mean",
    sd_log_rate_column="ln_rate_sd",
    grid_column="analysis_grid_index",
    potential_column="E_V_SHE",
):
    varying_values = np.asarray(
        varying_values,
        dtype=float,
    )

    if varying_values.ndim != 1:
        raise ValueError(
            "Varying-condition values must be one-dimensional."
        )

    if len(varying_values) < 2:
        raise ValueError(
            "At least two condition values are required "
            "to calculate a reaction order."
        )

    if not np.all(
        np.isfinite(varying_values)
    ):
        raise ValueError(
            "Varying-condition values contain non-finite values."
        )

    if not np.all(
        varying_values > 0
    ):
        raise ValueError(
            "Reaction-order condition values must be positive."
        )

    log_values = np.log(
        varying_values
    )

    centered = (
        log_values
        - np.mean(log_values)
    )

    denominator = np.sum(
        centered**2
    )

    if denominator <= 0:
        raise ValueError(
            "Reaction-order condition values must not all be identical."
        )

    slope_weights = (
        centered
        / denominator
    )

    records = []

    grouped = data.groupby(
        group_columns,
        sort=False,
    )

    for group_values, group_data in grouped:
        if not isinstance(
            group_values,
            tuple,
        ):
            group_values = (
                group_values,
            )

        condition_frames = []

        for value in varying_values:
            condition = group_data[
                np.isclose(
                    group_data[
                        varying_column
                    ].to_numpy(),
                    value,
                    rtol=0,
                    atol=1e-12,
                )
            ].sort_values(
                grid_column
            )

            if condition.empty:
                raise ValueError(
                    f"Missing {varying_column}={value} "
                    f"for group {group_values}."
                )

            condition_frames.append(
                condition
            )

        common_indices = (
            _get_common_grid_indices(
                condition_frames,
                grid_column,
            )
        )

        for grid_index in common_indices:
            log_rate_means = np.array(
                [
                    _get_value_at_grid_index(
                        frame,
                        grid_index,
                        grid_column,
                        mean_log_rate_column,
                    )
                    for frame in condition_frames
                ],
                dtype=float,
            )

            log_rate_sds = np.array(
                [
                    _get_value_at_grid_index(
                        frame,
                        grid_index,
                        grid_column,
                        sd_log_rate_column,
                    )
                    for frame in condition_frames
                ],
                dtype=float,
            )

            if not np.all(
                np.isfinite(
                    log_rate_means
                )
            ):
                raise ValueError(
                    "Mean log rates contain non-finite values."
                )

            reaction_order = np.sum(
                slope_weights
                * log_rate_means
            )

            if np.all(
                np.isfinite(
                    log_rate_sds
                )
            ):
                reaction_order_sd = np.sqrt(
                    np.sum(
                        (
                            slope_weights
                            * log_rate_sds
                        )**2
                    )
                )
            else:
                reaction_order_sd = np.nan

            potential_values = np.array(
                [
                    _get_value_at_grid_index(
                        frame,
                        grid_index,
                        grid_column,
                        potential_column,
                    )
                    for frame in condition_frames
                ],
                dtype=float,
            )

            if not np.allclose(
                potential_values,
                potential_values[0],
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Common grid index {grid_index} "
                    "maps to inconsistent potentials."
                )

            record = {
                column: value
                for column, value in zip(
                    group_columns,
                    group_values,
                )
            }

            record.update(
                {
                    grid_column: grid_index,
                    potential_column: (
                        potential_values[0]
                    ),
                    output_column: (
                        reaction_order
                    ),
                    output_sd_column: (
                        reaction_order_sd
                    ),
                    "n_order_conditions": (
                        len(varying_values)
                    ),
                    "order_condition_min": (
                        float(
                            np.min(
                                varying_values
                            )
                        )
                    ),
                    "order_condition_max": (
                        float(
                            np.max(
                                varying_values
                            )
                        )
                    ),
                }
            )

            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


def calculate_adjacent_log_orders(
    data,
    varying_column,
    varying_values,
    group_columns,
    output_column,
    output_sd_column,
    lower_value_column,
    upper_value_column,
    mean_log_rate_column="ln_rate_mean",
    sd_log_rate_column="ln_rate_sd",
    grid_column="analysis_grid_index",
    potential_column="E_V_SHE",
):
    varying_values = np.asarray(
        varying_values,
        dtype=float,
    )

    if varying_values.ndim != 1:
        raise ValueError(
            "Varying-condition values must be one-dimensional."
        )

    if len(varying_values) < 2:
        raise ValueError(
            "At least two condition values are required."
        )

    if not np.all(
        varying_values > 0
    ):
        raise ValueError(
            "Reaction-order condition values must be positive."
        )

    records = []

    grouped = data.groupby(
        group_columns,
        sort=False,
    )

    for group_values, group_data in grouped:
        if not isinstance(
            group_values,
            tuple,
        ):
            group_values = (
                group_values,
            )

        for i in range(
            len(varying_values) - 1
        ):
            lower_value = (
                varying_values[i]
            )
            upper_value = (
                varying_values[i + 1]
            )

            lower_data = group_data[
                np.isclose(
                    group_data[
                        varying_column
                    ].to_numpy(),
                    lower_value,
                    rtol=0,
                    atol=1e-12,
                )
            ].sort_values(
                grid_column
            )

            upper_data = group_data[
                np.isclose(
                    group_data[
                        varying_column
                    ].to_numpy(),
                    upper_value,
                    rtol=0,
                    atol=1e-12,
                )
            ].sort_values(
                grid_column
            )

            if lower_data.empty:
                raise ValueError(
                    f"Missing {varying_column}={lower_value} "
                    f"for group {group_values}."
                )

            if upper_data.empty:
                raise ValueError(
                    f"Missing {varying_column}={upper_value} "
                    f"for group {group_values}."
                )

            common_indices = (
                _get_common_grid_indices(
                    [
                        lower_data,
                        upper_data,
                    ],
                    grid_column,
                )
            )

            log_ratio = np.log(
                upper_value
                / lower_value
            )

            for grid_index in common_indices:
                lower_mean = float(
                    _get_value_at_grid_index(
                        lower_data,
                        grid_index,
                        grid_column,
                        mean_log_rate_column,
                    )
                )

                upper_mean = float(
                    _get_value_at_grid_index(
                        upper_data,
                        grid_index,
                        grid_column,
                        mean_log_rate_column,
                    )
                )

                lower_sd = float(
                    _get_value_at_grid_index(
                        lower_data,
                        grid_index,
                        grid_column,
                        sd_log_rate_column,
                    )
                )

                upper_sd = float(
                    _get_value_at_grid_index(
                        upper_data,
                        grid_index,
                        grid_column,
                        sd_log_rate_column,
                    )
                )

                reaction_order = (
                    upper_mean
                    - lower_mean
                ) / log_ratio

                if (
                    np.isfinite(
                        lower_sd
                    )
                    and np.isfinite(
                        upper_sd
                    )
                ):
                    reaction_order_sd = (
                        np.sqrt(
                            lower_sd**2
                            + upper_sd**2
                        )
                        / abs(
                            log_ratio
                        )
                    )
                else:
                    reaction_order_sd = (
                        np.nan
                    )

                lower_E = float(
                    _get_value_at_grid_index(
                        lower_data,
                        grid_index,
                        grid_column,
                        potential_column,
                    )
                )

                upper_E = float(
                    _get_value_at_grid_index(
                        upper_data,
                        grid_index,
                        grid_column,
                        potential_column,
                    )
                )

                if not np.isclose(
                    lower_E,
                    upper_E,
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Common grid index {grid_index} "
                        "maps to inconsistent potentials."
                    )

                record = {
                    column: value
                    for column, value in zip(
                        group_columns,
                        group_values,
                    )
                }

                record.update(
                    {
                        lower_value_column: (
                            float(
                                lower_value
                            )
                        ),
                        upper_value_column: (
                            float(
                                upper_value
                            )
                        ),
                        grid_column: (
                            grid_index
                        ),
                        potential_column: (
                            lower_E
                        ),
                        output_column: (
                            reaction_order
                        ),
                        output_sd_column: (
                            reaction_order_sd
                        ),
                    }
                )

                records.append(
                    record
                )

    return pd.DataFrame(
        records
    )