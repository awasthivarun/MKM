
"""Build acidic and basic Pt100 Bayesian fits using the shared MKM inference infrastructure."""

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import yaml

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.model import build_pymc_model
from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.pt100 import build_pt100_mechanism


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


def load_pt100_config(paths):
    """Load Pt100 model and prior configurations without modifying AgPd settings."""
    model_path = paths.config_dir / "models" / "pt100.yaml"

    with open(model_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    prior_path = paths.root / config["priors"]["path"]
    with open(prior_path, encoding="utf-8") as file:
        prior_config = yaml.safe_load(file)

    return config, prior_config


def resolve_pt100_parameter_specs(prior_config, medium, model_name):
    """Select only parameters required by the requested Pt100 mechanism."""
    parameter_class = PT100_PARAMETER_CLASSES[model_name]
    required_names = [field.name for field in fields(parameter_class)]

    profile = prior_config["prior_profiles"][medium]["parameters"]
    missing = [name for name in required_names if name not in profile]
    if missing:
        raise ValueError(f"Missing Pt100 priors for {medium}/{model_name}: {missing}")

    return {name: dict(profile[name]) for name in required_names}


def pt100_likelihood_kwargs(config):
    """Resolve the existing linear-rate Normal likelihood configuration."""
    likelihood = config["likelihood"]
    expected_form = "sigma_abs + sigma_rel * model_rate"

    if likelihood["name"] != RATE_NORMAL or likelihood["sigma_form"] != expected_form:
        raise ValueError("Pt100 must use the shared linear-rate Normal likelihood.")

    if "shared" not in likelihood["supported_error_structures"]:
        raise ValueError("Pt100 fitting requires the shared error structure.")

    keys = (
        "sigma_abs_prior_median_s_inv",
        "sigma_abs_prior_log_sd",
        "sigma_rel_prior_median",
        "sigma_rel_prior_log_sd",
    )
    return {name: likelihood[name] for name in keys}


def build_pt100_fit(paths, medium, model_name):
    """Load selected Pt100 data and assemble its PyMC model and parameter specifications."""
    config, prior_config = load_pt100_config(paths)

    if medium not in config["media"]:
        raise ValueError(f"Unsupported Pt100 medium: {medium}")
    if model_name not in config["mechanisms"] or model_name not in PT100_PARAMETER_CLASSES:
        raise ValueError(f"Unsupported Pt100 mechanism: {model_name}")
    if config["materials"] != ["Pt100"]:
        raise ValueError("Pt100 fitting configuration must contain only Pt100.")

    medium_config = config["media"][medium]
    data_path = paths.root / medium_config["selected_data"]

    selected = pd.read_parquet(data_path)
    selected = selected.loc[selected["material"] == "Pt100"].copy()

    if selected.empty:
        raise ValueError(f"No selected Pt100 observations found for {medium}.")
    if "medium" in selected and not selected["medium"].eq(medium).all():
        raise ValueError(f"Selected Pt100 dataset contains observations outside {medium}.")

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_electrolyte_M",
    )
    inputs = build_model_input_arrays(model_data)

    physical_specs = resolve_pt100_parameter_specs(prior_config, medium, model_name)
    likelihood_kwargs = pt100_likelihood_kwargs(config)

    mechanism = build_pt100_mechanism(
        model_name=model_name,
        medium=medium,
        temperature_K=config["temperature_K"],
        parameter_specs=physical_specs,
    )

    built_model = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        likelihood_name=RATE_NORMAL,
        error_structure="shared",
        likelihood_kwargs=likelihood_kwargs,
    )

    # Include the likelihood parameters in posterior/prior summary specifications.
    parameter_specs = {
        **physical_specs,
        "sigma_rate_abs": {
            "distribution": "lognormal",
            "median": likelihood_kwargs["sigma_abs_prior_median_s_inv"],
            "log_sd": likelihood_kwargs["sigma_abs_prior_log_sd"],
        },
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": likelihood_kwargs["sigma_rel_prior_median"],
            "log_sd": likelihood_kwargs["sigma_rel_prior_log_sd"],
        },
    }

    output_dir = (
        paths.results_dir / medium_config["dataset_name"]
        / "posterior" / "individual" / "Pt100" / model_name
    )

    return Pt100BuiltFit(
        medium=medium,
        model_name=model_name,
        inputs=inputs,
        built_model=built_model,
        parameter_specs=parameter_specs,
        config=config,
        data_path=data_path,
        model_config_path=paths.config_dir / "models" / "pt100.yaml",
        prior_config_path=paths.root / config["priors"]["path"],
        output_dir=output_dir,
    )
