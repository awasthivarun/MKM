from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelDataTables:
    conditions: pd.DataFrame
    model_points: pd.DataFrame
    observations: pd.DataFrame


def _require_columns(data, required_columns):
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_finite(data, columns):
    for column in columns:
        values = data[column].to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column '{column}' contains non-finite values.")


def build_model_data(selected_replicates, electrolyte_concentration_column):
    required_columns = [
        "material",
        electrolyte_concentration_column,
        "CO_mole_fraction",
        "replicate",
        "analysis_grid_index",
        "E_V_SHE",
        "rate_s_inv",
        "ln_rate",
    ]

    _require_columns(selected_replicates, required_columns)

    if selected_replicates.empty:
        raise ValueError("Selected replicate data are empty.")

    data = (
        selected_replicates[required_columns]
        .copy()
        .rename(columns={electrolyte_concentration_column: "electrolyte_concentration_M"})
    )

    _validate_finite(
        data,
        ["electrolyte_concentration_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE", "rate_s_inv", "ln_rate"],
    )

    if np.any(data["electrolyte_concentration_M"].to_numpy() <= 0):
        raise ValueError("Electrolyte concentrations must be positive.")

    if np.any(data["CO_mole_fraction"].to_numpy() <= 0):
        raise ValueError("CO mole fractions must be positive.")

    if np.any(data["rate_s_inv"].to_numpy() <= 0):
        raise ValueError("Rates must be positive.")

    np.testing.assert_allclose(
        data["ln_rate"].to_numpy(),
        np.log(data["rate_s_inv"].to_numpy()),
        rtol=1e-10,
        atol=1e-12,
        err_msg="ln_rate is inconsistent with rate_s_inv.",
    )

    duplicate_observations = data.duplicated(
        subset=["material", "electrolyte_concentration_M", "CO_mole_fraction", "replicate", "analysis_grid_index"]
    )

    if duplicate_observations.any():
        raise ValueError("Duplicate replicate observations exist at the same experimental state.")

    condition_columns = ["material", "electrolyte_concentration_M", "CO_mole_fraction"]

    conditions = data[condition_columns].drop_duplicates().sort_values(condition_columns).reset_index(drop=True)
    conditions.insert(0, "condition_id", np.arange(len(conditions), dtype=np.int64))

    data = data.merge(conditions, on=condition_columns, how="left", validate="many_to_one")

    potential_counts = data.groupby(["condition_id", "analysis_grid_index"], sort=False)["E_V_SHE"].nunique()

    if np.any(potential_counts.to_numpy() != 1):
        raise ValueError("A condition/grid index maps to multiple potentials.")

    model_points = (
        data[["condition_id", "analysis_grid_index", "E_V_SHE"]]
        .drop_duplicates()
        .sort_values(["condition_id", "E_V_SHE", "analysis_grid_index"])
        .reset_index(drop=True)
    )

    model_points.insert(0, "model_point_id", np.arange(len(model_points), dtype=np.int64))
    model_points = model_points.merge(conditions, on="condition_id", how="left", validate="many_to_one")

    observations = data.merge(
        model_points[["model_point_id", "condition_id", "analysis_grid_index"]],
        on=["condition_id", "analysis_grid_index"],
        how="left",
        validate="many_to_one",
    )

    observations = (
        observations[
            [
                "model_point_id",
                "condition_id",
                "material",
                "electrolyte_concentration_M",
                "CO_mole_fraction",
                "replicate",
                "analysis_grid_index",
                "E_V_SHE",
                "rate_s_inv",
                "ln_rate",
            ]
        ]
        .sort_values(["model_point_id", "replicate"])
        .reset_index(drop=True)
    )

    observations.insert(0, "observation_id", np.arange(len(observations), dtype=np.int64))

    if observations["model_point_id"].isna().any():
        raise ValueError("Some observations could not be mapped to model points.")

    point_counts = observations["model_point_id"].value_counts()

    if np.any(point_counts.to_numpy() < 1):
        raise ValueError("A model point has no observations.")

    return ModelDataTables(conditions=conditions, model_points=model_points, observations=observations)