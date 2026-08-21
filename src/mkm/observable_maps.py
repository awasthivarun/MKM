from dataclasses import dataclass

import numpy as np
import pandas as pd

from mkm.constants import F_C_mol, R_J_mol_K


@dataclass(frozen=True)
class LinearObservableMap:
    outputs: pd.DataFrame
    terms: pd.DataFrame


def _finite_difference_weights(source_potentials, target_potential):
    source_potentials = np.asarray(source_potentials, dtype=float)
    target_potential = float(target_potential)

    if len(source_potentials) != 3:
        raise ValueError("Second-order derivative stencil requires exactly three source points.")

    offsets = source_potentials - target_potential
    matrix = np.vstack([np.ones(3), offsets, offsets**2])
    rhs = np.array([0.0, 1.0, 0.0])

    return np.linalg.solve(matrix, rhs)


def build_alpha_map(model_points, temperature_K):
    required_columns = ["model_point_id", "condition_id", "analysis_grid_index", "E_V_SHE"]

    missing = [column for column in required_columns if column not in model_points.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    factor = R_J_mol_K * float(temperature_K) / F_C_mol

    output_records = []
    term_records = []

    observable_id = 0

    for condition_id, condition in model_points.groupby("condition_id", sort=False):
        condition = condition.sort_values("E_V_SHE").reset_index(drop=True)

        if len(condition) < 3:
            raise ValueError(f"Condition {condition_id} has fewer than three model points.")

        potential = condition["E_V_SHE"].to_numpy()

        if not np.all(np.diff(potential) > 0):
            raise ValueError(f"Condition {condition_id} potentials must be strictly increasing.")

        point_ids = condition["model_point_id"].to_numpy(dtype=np.int64)

        for i in range(len(condition)):
            if i == 0:
                source_positions = np.array([0, 1, 2])
            elif i == len(condition) - 1:
                source_positions = np.array([len(condition) - 3, len(condition) - 2, len(condition) - 1])
            else:
                source_positions = np.array([i - 1, i, i + 1])

            derivative_weights = _finite_difference_weights(
                source_potentials=potential[source_positions],
                target_potential=potential[i],
            )

            output_records.append(
                {
                    "observable_id": observable_id,
                    "condition_id": int(condition_id),
                    "analysis_grid_index": int(condition.loc[i, "analysis_grid_index"]),
                    "E_V_SHE": float(potential[i]),
                }
            )

            for source_position, weight in zip(source_positions, derivative_weights):
                term_records.append(
                    {
                        "observable_id": observable_id,
                        "model_point_id": int(point_ids[source_position]),
                        "coefficient": float(factor * weight),
                    }
                )

            observable_id += 1

    return LinearObservableMap(outputs=pd.DataFrame(output_records), terms=pd.DataFrame(term_records))


def build_log_slope_order_map(model_points, varying_column, varying_values, group_columns):
    varying_values = np.asarray(varying_values, dtype=float)

    log_values = np.log(varying_values)
    centered = log_values - np.mean(log_values)
    denominator = np.sum(centered**2)

    if denominator <= 0:
        raise ValueError("Reaction-order values must not all be identical.")

    weights = centered / denominator

    output_records = []
    term_records = []

    observable_id = 0

    for group_values, group_data in model_points.groupby(group_columns, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        condition_frames = []

        for value in varying_values:
            condition = group_data[np.isclose(group_data[varying_column].to_numpy(), value, rtol=0, atol=1e-12)]

            if condition.empty:
                raise ValueError(f"Missing {varying_column}={value} for group {group_values}.")

            condition_frames.append(condition)

        common_indices = set(condition_frames[0]["analysis_grid_index"])

        for condition in condition_frames[1:]:
            common_indices &= set(condition["analysis_grid_index"])

        for grid_index in sorted(common_indices):
            source_rows = []

            for condition in condition_frames:
                rows = condition[condition["analysis_grid_index"] == grid_index]

                if len(rows) != 1:
                    raise ValueError("Expected exactly one model point per condition/grid index.")

                source_rows.append(rows.iloc[0])

            potentials = np.array([row["E_V_SHE"] for row in source_rows], dtype=float)

            if not np.allclose(potentials, potentials[0], rtol=0, atol=1e-12):
                raise ValueError("Common analysis-grid index maps to inconsistent potentials.")

            output_record = {
                "observable_id": observable_id,
                "analysis_grid_index": int(grid_index),
                "E_V_SHE": float(potentials[0]),
            }

            output_record.update({column: value for column, value in zip(group_columns, group_values)})
            output_records.append(output_record)

            for row, weight in zip(source_rows, weights):
                term_records.append(
                    {
                        "observable_id": observable_id,
                        "model_point_id": int(row["model_point_id"]),
                        "coefficient": float(weight),
                    }
                )

            observable_id += 1

    return LinearObservableMap(outputs=pd.DataFrame(output_records), terms=pd.DataFrame(term_records))


def build_adjacent_log_order_map(
    model_points, varying_column, varying_values, group_columns, lower_value_column, upper_value_column
):
    varying_values = np.asarray(varying_values, dtype=float)

    output_records = []
    term_records = []

    observable_id = 0

    for group_values, group_data in model_points.groupby(group_columns, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        for i in range(len(varying_values) - 1):
            lower_value = varying_values[i]
            upper_value = varying_values[i + 1]

            lower = group_data[np.isclose(group_data[varying_column].to_numpy(), lower_value, rtol=0, atol=1e-12)]
            upper = group_data[np.isclose(group_data[varying_column].to_numpy(), upper_value, rtol=0, atol=1e-12)]

            if lower.empty or upper.empty:
                raise ValueError(f"Missing adjacent values {lower_value}, {upper_value} for group {group_values}.")

            common_indices = set(lower["analysis_grid_index"]) & set(upper["analysis_grid_index"])
            log_ratio = np.log(upper_value / lower_value)

            for grid_index in sorted(common_indices):
                lower_row = lower[lower["analysis_grid_index"] == grid_index]
                upper_row = upper[upper["analysis_grid_index"] == grid_index]

                if len(lower_row) != 1 or len(upper_row) != 1:
                    raise ValueError("Expected exactly one model point per condition/grid index.")

                lower_row = lower_row.iloc[0]
                upper_row = upper_row.iloc[0]

                if not np.isclose(lower_row["E_V_SHE"], upper_row["E_V_SHE"], rtol=0, atol=1e-12):
                    raise ValueError("Common analysis-grid index maps to inconsistent potentials.")

                output_record = {
                    "observable_id": observable_id,
                    lower_value_column: float(lower_value),
                    upper_value_column: float(upper_value),
                    "analysis_grid_index": int(grid_index),
                    "E_V_SHE": float(lower_row["E_V_SHE"]),
                }

                output_record.update({column: value for column, value in zip(group_columns, group_values)})
                output_records.append(output_record)

                term_records.extend(
                    [
                        {
                            "observable_id": observable_id,
                            "model_point_id": int(lower_row["model_point_id"]),
                            "coefficient": float(-1.0 / log_ratio),
                        },
                        {
                            "observable_id": observable_id,
                            "model_point_id": int(upper_row["model_point_id"]),
                            "coefficient": float(1.0 / log_ratio),
                        },
                    ]
                )

                observable_id += 1

    return LinearObservableMap(outputs=pd.DataFrame(output_records), terms=pd.DataFrame(term_records))


def evaluate_linear_observable_map(log_rate, observable_map):
    log_rate = np.asarray(log_rate, dtype=float)

    n_outputs = len(observable_map.outputs)
    result = np.zeros(n_outputs, dtype=float)

    for row in observable_map.terms.itertuples(index=False):
        result[row.observable_id] += row.coefficient * log_rate[row.model_point_id]

    return result

def evaluate_linear_observable_map_draws(log_rate, observable_map):

    log_rate = np.asarray(log_rate, dtype=float)

    if log_rate.ndim < 1:
        raise ValueError("Log-rate draws must have at least one dimension.")

    required_output_columns = {"observable_id"}
    required_term_columns = {"observable_id", "model_point_id", "coefficient"}

    if not required_output_columns.issubset(observable_map.outputs.columns):
        raise ValueError("Observable-map outputs are missing 'observable_id'.")

    if not required_term_columns.issubset(observable_map.terms.columns):
        raise ValueError("Observable-map terms are missing required columns.")

    n_outputs = len(observable_map.outputs)

    if n_outputs == 0:
        return np.empty((*log_rate.shape[:-1], 0), dtype=float)

    output_ids = observable_map.outputs["observable_id"].to_numpy(dtype=np.int64)

    if not np.array_equal(output_ids, np.arange(n_outputs)):
        raise ValueError("Observable IDs must be contiguous and ordered from zero.")

    term_output_ids = observable_map.terms["observable_id"].to_numpy(dtype=np.int64)
    model_point_ids = observable_map.terms["model_point_id"].to_numpy(dtype=np.int64)
    coefficients = observable_map.terms["coefficient"].to_numpy(dtype=float)

    if np.any(term_output_ids < 0) or np.any(term_output_ids >= n_outputs):
        raise ValueError("Observable-map term contains an invalid observable ID.")

    if np.any(model_point_ids < 0) or np.any(model_point_ids >= log_rate.shape[-1]):
        raise ValueError("Observable-map term contains an invalid model-point ID.")

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Observable-map coefficients must be finite.")

    term_counts = np.bincount(term_output_ids, minlength=n_outputs)

    if np.any(term_counts == 0):
        raise ValueError("Every observable must contain at least one map term.")

    max_terms = int(term_counts.max())
    source_indices = np.zeros((n_outputs, max_terms), dtype=np.int64)
    coefficient_matrix = np.zeros((n_outputs, max_terms), dtype=float)
    positions = np.zeros(n_outputs, dtype=np.int64)

    for observable_id, model_point_id, coefficient in zip(term_output_ids, model_point_ids, coefficients):
        position = positions[observable_id]
        source_indices[observable_id, position] = model_point_id
        coefficient_matrix[observable_id, position] = coefficient
        positions[observable_id] += 1

    return np.sum(log_rate[..., source_indices] * coefficient_matrix, axis=-1)