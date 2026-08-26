"""Shared loading and assembly for AgPd basic-media workflows."""

from pathlib import Path

import pandas as pd
import yaml

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.project_paths import ProjectPaths


def load_yaml(path: str | Path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_agpd_model_config(paths: ProjectPaths):
    return load_yaml(paths.agpd_model_config_path)


def load_agpd_preprocessing_config(paths: ProjectPaths):
    return load_yaml(paths.agpd_preprocessing_config_path)


def available_agpd_materials(config):
    return tuple(config.get("surface_composition", {}))


def validate_agpd_material(config, material: str):
    available = available_agpd_materials(config)
    if material not in available:
        raise ValueError(
            f"No AgPd surface-composition configuration is defined for material '{material}'. "
            f"Configured materials: {available}."
        )


def load_agpd_selected_material(paths: ProjectPaths, material: str):
    selected = pd.read_parquet(paths.agpd_selected_path)
    selected = selected.loc[selected["material"] == material].copy()

    if selected.empty:
        raise ValueError(f"No selected AgPd observations found for material '{material}'.")

    return selected


def build_agpd_model_data(paths: ProjectPaths, material: str):
    selected = load_agpd_selected_material(paths, material)
    return build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )


def build_agpd_inputs(model_data, config, likelihood_name: str):
    if likelihood_name == "setup_intercept":
        setup_config = config["likelihood"]["setup_intercept"]
        return build_model_input_arrays(
            model_data,
            setup_group_columns=setup_config["group_columns"],
            setup_zero_sum_columns=setup_config["zero_sum_within"],
        )

    if likelihood_name in {"iid", "mvn", "rate_normal"}:
        return build_model_input_arrays(model_data)

    raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")

def load_agpd_selected_materials(paths: ProjectPaths, materials):
    materials = tuple(materials)
    if not materials:
        raise ValueError("At least one material is required.")

    selected = pd.read_parquet(paths.agpd_selected_path)
    available = set(selected["material"].unique())
    missing = [material for material in materials if material not in available]
    if missing:
        raise ValueError(f"No selected AgPd observations found for materials: {missing}.")

    return selected.loc[selected["material"].isin(materials)].copy()


def build_agpd_composition_model_data(paths: ProjectPaths, materials):
    selected = load_agpd_selected_materials(paths, materials)
    return build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )
