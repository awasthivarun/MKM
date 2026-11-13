"""Build acidic and basic Pt100 fits using shared MKM infrastructure."""

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.model import build_pymc_model
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.ptru_registry import available_ptru_models, get_ptru_model_definition
from mkm.models.pt100 import build_pt100_mechanism
from mkm.workflows.configuration import load_yaml
from mkm.workflows.likelihood_configuration import (
    error_parameter_specs,
    rate_normal_likelihood_kwargs,
    validate_configured_error_structure,
)


@dataclass(frozen=True)
class Pt100FitSpecification:
    medium: str
    model_name: str
    fit_scope: str = "individual"
    material: str = "Pt100"
    parameterization: str | None = None
    error_structure: str = "shared"

    @property
    def is_all_materials(self):
        return self.fit_scope == "all_materials"


@dataclass(frozen=True)
class Pt100BuiltFit:
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
    specification: Pt100FitSpecification
    model_data: object


def load_pt100_config(paths):
    """Load Pt100 model and prior configurations through the common YAML loader."""
    config = load_yaml(paths.ptru_model_config_path)
    prior_path = paths.root / config["priors"]["path"]
    prior_config = load_yaml(prior_path)
    return config, prior_config


def resolve_pt100_fit_specification(
    config,
    medium,
    model_name,
    *,
    fit_scope="individual",
    material="Pt100",
    error_structure="shared",
):
    if medium not in config["media"]:
        raise ValueError(f"Unsupported Pt100 medium: {medium}")

    if config["materials"] != ["Pt100"]:
        raise ValueError("Pt100 fitting configuration must contain only Pt100.")

    if material != "Pt100":
        raise ValueError(f"Pt100 fitting does not support material '{material}'.")

    if fit_scope != "individual":
        raise ValueError("Pt100 currently supports only individual-material fitting.")

    available = available_ptru_models(medium=medium, fit_scope=fit_scope)
    if model_name not in config["mechanisms"] or model_name not in available:
        raise ValueError(f"Unsupported Pt100 mechanism: {model_name}")

    validate_configured_error_structure(config, error_structure)
    if error_structure != "shared":
        raise ValueError("Pt100 currently requires error_structure='shared'.")

    return Pt100FitSpecification(
        medium=medium,
        model_name=model_name,
        fit_scope=fit_scope,
        material=material,
        parameterization=None,
        error_structure=error_structure,
    )


def resolve_pt100_parameter_specs(prior_config, medium, model_name):
    """Select the physical parameters required by the registered mechanism."""
    definition = get_ptru_model_definition(model_name)
    required_names = tuple(field.name for field in fields(definition.parameter_class))

    profile = prior_config["prior_profiles"][medium]["parameters"]
    missing = [name for name in required_names if name not in profile]
    if missing:
        raise ValueError(f"Missing Pt100 priors for {medium}/{model_name}: {missing}")

    return {name: dict(profile[name]) for name in required_names}


def pt100_likelihood_kwargs(config):
    validate_configured_error_structure(config, "shared")
    return rate_normal_likelihood_kwargs(config)


def pt100_parameter_specs(prior_config, specification, config):
    return {
        **resolve_pt100_parameter_specs(
            prior_config, specification.medium, specification.model_name,
        ),
        **error_parameter_specs(config),
    }


def build_pt100_fit(paths, medium, model_name):
    """Assemble a Pt100 fit without changing the existing public function signature."""
    config, prior_config = load_pt100_config(paths)
    specification = resolve_pt100_fit_specification(config, medium, model_name)

    medium_config = config["media"][specification.medium]
    data_path = paths.root / medium_config["selected_data"]

    if data_path != paths.ptru_selected_path(specification.medium):
        raise ValueError("Configured Pt100 selected-data path differs from the canonical PtRu path.")

    selected = pd.read_parquet(data_path)
    selected = selected.loc[selected["material"] == specification.material].copy()

    if selected.empty:
        raise ValueError(f"No selected Pt100 observations found for {medium}.")
    if "medium" in selected and not selected["medium"].eq(medium).all():
        raise ValueError(f"Selected Pt100 dataset contains observations outside {medium}.")

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_electrolyte_M",
    )
    inputs = build_model_input_arrays(model_data)

    physical_specs = resolve_pt100_parameter_specs(
        prior_config,
        specification.medium,
        specification.model_name,
    )
    likelihood_kwargs = pt100_likelihood_kwargs(config)

    mechanism = build_pt100_mechanism(
        model_name=specification.model_name,
        medium=specification.medium,
        temperature_K=config["temperature_K"],
        parameter_specs=physical_specs,
    )

    built_model = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        likelihood_name=RATE_NORMAL,
        error_structure=specification.error_structure,
        likelihood_kwargs=likelihood_kwargs,
    )

    return Pt100BuiltFit(
        medium=specification.medium,
        model_name=specification.model_name,
        inputs=inputs,
        built_model=built_model,
        parameter_specs=pt100_parameter_specs(prior_config, specification, config),
        config=config,
        data_path=data_path,
        model_config_path=paths.ptru_model_config_path,
        prior_config_path=paths.root / config["priors"]["path"],
        output_dir=paths.ptru_posterior_output_dir(
            specification.medium,
            specification.model_name,
            specification.material,
        ),
        specification=specification,
        model_data=model_data,
    )