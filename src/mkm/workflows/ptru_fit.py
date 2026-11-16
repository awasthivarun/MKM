"""Unified individual and pooled PtRu fit construction, independently in acid and base."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.model import build_pymc_model
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.ptru import (
    available_ptru_models, build_ptru_mechanism, ptru_all_parameter_specs,
    ptru_parameterization_metadata,
)
from mkm.workflows.configuration import load_yaml
from mkm.workflows.likelihood_configuration import (
    error_parameter_specs, rate_normal_likelihood_kwargs, validate_configured_error_structure,
)


@dataclass(frozen=True)
class PtRuFitSpecification:
    medium: str
    model_name: str
    materials: tuple[str, ...]
    fit_scope: str
    material: str | None
    parameterization: str | None
    error_structure: str
    excluded_materials: tuple[str, ...] = ()

    @property
    def is_all_materials(self):
        return self.fit_scope == "all_materials"


@dataclass(frozen=True)
class PtRuBuiltFit:
    specification: PtRuFitSpecification
    medium: str
    model_name: str
    inputs: object
    built_model: object
    parameter_specs: dict
    config: dict
    data_path: Path
    model_config_path: Path
    prior_config_path: Path
    output_dir: Path
    model_data: object


def load_ptru_config(paths):
    config = load_yaml(paths.ptru_model_config_path)
    prior_path = paths.root / config["priors"]["path"]
    return config, load_yaml(prior_path)


def resolve_ptru_fit_specification(
    config, medium, model_name, *, material="Pt100", all_materials=False,
    parameterization="shared", error_structure="shared", excluded_materials=(),
):
    if medium not in config["media"]:
        raise ValueError(f"Unknown PtRu medium: {medium}")
    if model_name not in available_ptru_models() or model_name not in config["mechanisms"]:
        raise ValueError(f"Unknown PtRu model: {model_name}")
    validate_configured_error_structure(config, error_structure)
    all_material_names = tuple(config["materials"])
    exclusions = tuple(dict.fromkeys(excluded_materials))
    if set(exclusions) - set(all_material_names):
        raise ValueError(f"Unknown excluded PtRu materials: {sorted(set(exclusions) - set(all_material_names))}")
    if all_materials:
        if parameterization not in config["parameterizations"]:
            raise ValueError(f"Unknown PtRu parameterization: {parameterization}")
        materials = tuple(name for name in all_material_names if name not in exclusions)
        if len(materials) < 2:
            raise ValueError("All-material fitting requires at least two materials.")
        if model_name == "CO_BF" and "Pt100" in materials:
            raise ValueError("BF-only pooled fits require --exclude-materials Pt100.")
        if parameterization == "independent" and error_structure != "shared":
            raise ValueError("Independent parameterization requires shared likelihood errors.")
        return PtRuFitSpecification(
            medium, model_name, materials, "all_materials", None, parameterization,
            error_structure, tuple(name for name in all_material_names if name in exclusions),
        )
    if exclusions:
        raise ValueError("--exclude-materials requires --all-materials.")
    if material not in all_material_names:
        raise ValueError(f"Unknown PtRu material: {material}")
    if material == "Pt100" and "BF" in model_name.split("_"):
        raise ValueError("Pt100 supports only CO_LH, CO_ER, and CO_ER_LH.")
    if error_structure != "shared":
        raise ValueError("Individual PtRu fits use --error-structure shared.")
    return PtRuFitSpecification(medium, model_name, (material,), "individual", material, None, error_structure)


def build_ptru_fit(paths, specification, *, selected=None, output_dir=None):
    config, priors = load_ptru_config(paths)
    # Revalidate the specification against the configuration and its allowed material set.
    expected = resolve_ptru_fit_specification(
        config, specification.medium, specification.model_name,
        material=specification.material if specification.material is not None else "Pt100",
        all_materials=specification.is_all_materials, parameterization=specification.parameterization or "shared",
        error_structure=specification.error_structure, excluded_materials=specification.excluded_materials,
    )
    if expected != specification:
        raise ValueError("PtRu fit specification differs from the current configuration.")
    if config["electrolyte"]["activity_model"] != "ideal_molarity":
        raise ValueError("PtRu model requires ideal-molarity electrolyte activities.")
    if not np.isclose(float(config["electrolyte"]["standard_state_concentration_M"]), 1.0):
        raise ValueError("PtRu model expects a 1 M electrolyte standard state.")
    gas = config["gas"]
    if not np.isclose(float(gas["standard_state_pressure_bar"]), float(gas["total_pressure_bar"])):
        raise ValueError("PtRu model expects equal standard and total gas pressures.")
    data_path = paths.root / config["media"][specification.medium]["selected_data"]
    if data_path != paths.ptru_selected_path(specification.medium):
        raise ValueError("PtRu data path does not match canonical selected-data location.")
    if selected is None:
        selected = pd.read_parquet(data_path)
    selected = selected.loc[selected["material"].isin(specification.materials)].copy()
    if selected.empty or set(selected["material"]) != set(specification.materials):
        raise ValueError("PtRu selected data do not contain every requested material.")
    if "medium" in selected and not selected["medium"].eq(specification.medium).all():
        raise ValueError("PtRu selected data contain the wrong electrolyte medium.")
    model_data = build_model_data(selected, electrolyte_concentration_column="C_electrolyte_M")
    inputs = build_model_input_arrays(model_data)
    if set(inputs.materials) != set(specification.materials):
        raise ValueError("PtRu model inputs do not match the requested materials.")
    # The model-point material order is defined by observed data and must match the specification.
    if tuple(inputs.materials) != specification.materials:
        raise ValueError(f"PtRu material order differs: {inputs.materials} != {specification.materials}.")
    physical_specs = ptru_all_parameter_specs(priors, config, specification)
    mechanism = build_ptru_mechanism(specification, config, priors)
    built_model = build_pymc_model(
        inputs, mechanism, likelihood_name=RATE_NORMAL,
        error_structure=specification.error_structure,
        likelihood_kwargs=rate_normal_likelihood_kwargs(config),
    )
    return PtRuBuiltFit(
        specification=specification, medium=specification.medium, model_name=specification.model_name,
        inputs=inputs, built_model=built_model,
        parameter_specs={**physical_specs, **error_parameter_specs(config)}, config=config,
        data_path=data_path, model_config_path=paths.ptru_model_config_path,
        prior_config_path=paths.ptru_prior_config_path,
        output_dir=output_dir or paths.ptru_posterior_output_dir(
            specification.medium, specification.model_name, specification.material,
            fit_scope=specification.fit_scope, parameterization=specification.parameterization,
            error_structure=specification.error_structure, excluded_materials=specification.excluded_materials,
        ),
        model_data=model_data,
    )
