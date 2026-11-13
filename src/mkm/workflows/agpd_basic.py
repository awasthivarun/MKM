"""Shared loading and assembly for AgPd basic-media workflows."""

import pandas as pd

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.project_paths import ProjectPaths
from mkm.workflows.configuration import load_yaml, merge_config_sections


def _merge_config_sections(*configs):
    return merge_config_sections(*configs)


def load_agpd_model_config(paths: ProjectPaths):
    return _merge_config_sections(*(load_yaml(path) for path in paths.agpd_model_config_paths))


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


def load_agpd_selected_material(paths: ProjectPaths, material: str):
    return load_agpd_selected_materials(paths, (material,))


def build_agpd_model_data(paths: ProjectPaths, materials):
    if isinstance(materials, str):
        materials = (materials,)

    selected = load_agpd_selected_materials(paths, materials)
    return build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )


def build_agpd_inputs(model_data):
    return build_model_input_arrays(model_data)
