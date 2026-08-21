from dataclasses import dataclass

import numpy as np

from mkm.model_data import ModelDataTables


@dataclass(frozen=True)
class ModelInputArrays:
    materials: tuple[str, ...]

    condition_material_index: np.ndarray
    condition_ln_electrolyte_concentration: np.ndarray
    condition_ln_CO_mole_fraction: np.ndarray

    model_point_condition_index: np.ndarray
    model_point_E_V_SHE: np.ndarray

    observation_model_point_index: np.ndarray
    observation_ln_rate: np.ndarray
    observation_replicate: np.ndarray


@dataclass(frozen=True)
class ModelPointInputs:
    materials: tuple[str, ...]

    material_index: np.ndarray
    E_V_SHE: np.ndarray

    ln_electrolyte_concentration: np.ndarray
    ln_CO_mole_fraction: np.ndarray


def _validate_contiguous_ids(data, column):
    values = data[column].to_numpy(dtype=np.int64)
    expected = np.arange(len(data), dtype=np.int64)

    if not np.array_equal(values, expected):
        raise ValueError(f"'{column}' must be contiguous and aligned with row position.")


def build_model_input_arrays(model_data: ModelDataTables):
    conditions = model_data.conditions.sort_values("condition_id").reset_index(drop=True)
    model_points = model_data.model_points.sort_values("model_point_id").reset_index(drop=True)
    observations = model_data.observations.sort_values("observation_id").reset_index(drop=True)

    _validate_contiguous_ids(conditions, "condition_id")
    _validate_contiguous_ids(model_points, "model_point_id")
    _validate_contiguous_ids(observations, "observation_id")

    materials = tuple(conditions["material"].drop_duplicates().tolist())
    material_to_index = {material: index for index, material in enumerate(materials)}

    condition_material_index = conditions["material"].map(material_to_index).to_numpy(dtype=np.int64)
    electrolyte_concentration = conditions["electrolyte_concentration_M"].to_numpy(dtype=float)
    CO_mole_fraction = conditions["CO_mole_fraction"].to_numpy(dtype=float)

    if not np.all(electrolyte_concentration > 0):
        raise ValueError("Electrolyte concentrations must be positive.")

    if not np.all(CO_mole_fraction > 0):
        raise ValueError("CO mole fractions must be positive.")

    condition_ln_electrolyte_concentration = np.log(electrolyte_concentration)
    condition_ln_CO_mole_fraction = np.log(CO_mole_fraction)

    model_point_condition_index = model_points["condition_id"].to_numpy(dtype=np.int64)
    model_point_E_V_SHE = model_points["E_V_SHE"].to_numpy(dtype=float)

    observation_model_point_index = observations["model_point_id"].to_numpy(dtype=np.int64)
    observation_ln_rate = observations["ln_rate"].to_numpy(dtype=float)
    observation_replicate = observations["replicate"].astype(str).to_numpy()

    if not np.all(np.isfinite(condition_ln_electrolyte_concentration)):
        raise ValueError("Log electrolyte concentrations contain non-finite values.")

    if not np.all(np.isfinite(condition_ln_CO_mole_fraction)):
        raise ValueError("Log CO mole fractions contain non-finite values.")

    if not np.all(np.isfinite(model_point_E_V_SHE)):
        raise ValueError("Model-point potentials contain non-finite values.")

    if not np.all(np.isfinite(observation_ln_rate)):
        raise ValueError("Observed log rates contain non-finite values.")

    if np.any(model_point_condition_index < 0) or np.any(model_point_condition_index >= len(conditions)):
        raise ValueError("Model points contain invalid condition indices.")

    if np.any(observation_model_point_index < 0) or np.any(observation_model_point_index >= len(model_points)):
        raise ValueError("Observations contain invalid model-point indices.")

    return ModelInputArrays(
        materials=materials,
        condition_material_index=condition_material_index,
        condition_ln_electrolyte_concentration=condition_ln_electrolyte_concentration,
        condition_ln_CO_mole_fraction=condition_ln_CO_mole_fraction,
        model_point_condition_index=model_point_condition_index,
        model_point_E_V_SHE=model_point_E_V_SHE,
        observation_model_point_index=observation_model_point_index,
        observation_ln_rate=observation_ln_rate,
        observation_replicate=observation_replicate,
    )


def build_model_point_inputs(inputs: ModelInputArrays):
    condition_index = np.asarray(inputs.model_point_condition_index, dtype=np.int64)

    material_index = inputs.condition_material_index[condition_index]
    ln_electrolyte_concentration = inputs.condition_ln_electrolyte_concentration[condition_index]
    ln_CO_mole_fraction = inputs.condition_ln_CO_mole_fraction[condition_index]
    E_V_SHE = np.asarray(inputs.model_point_E_V_SHE, dtype=float)

    n_points = len(E_V_SHE)

    arrays = {
        "material_index": material_index,
        "ln_electrolyte_concentration": ln_electrolyte_concentration,
        "ln_CO_mole_fraction": ln_CO_mole_fraction,
    }

    for name, values in arrays.items():
        if len(values) != n_points:
            raise ValueError(f"'{name}' does not have one value per model point.")

    if not np.all(np.isfinite(E_V_SHE)):
        raise ValueError("Model-point potentials contain non-finite values.")

    if not np.all(np.isfinite(ln_electrolyte_concentration)):
        raise ValueError("Model-point log electrolyte concentrations contain non-finite values.")

    if not np.all(np.isfinite(ln_CO_mole_fraction)):
        raise ValueError("Model-point log CO mole fractions contain non-finite values.")

    return ModelPointInputs(
        materials=inputs.materials,
        material_index=np.asarray(material_index, dtype=np.int64),
        E_V_SHE=E_V_SHE,
        ln_electrolyte_concentration=np.asarray(ln_electrolyte_concentration, dtype=float),
        ln_CO_mole_fraction=np.asarray(ln_CO_mole_fraction, dtype=float),
    )


def build_model_coords(inputs: ModelInputArrays):
    return {
        "material": list(inputs.materials),
        "condition": np.arange(len(inputs.condition_material_index)),
        "model_point": np.arange(len(inputs.model_point_E_V_SHE)),
        "observation": np.arange(len(inputs.observation_ln_rate)),
    }