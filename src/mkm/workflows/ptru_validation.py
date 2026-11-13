"""Pt100-only leave-one-condition-out (LOCO) validation helpers."""


import numpy as np

from mkm.inference.likelihoods import RATE_NORMAL
from mkm.inference.model import build_pymc_model
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.pt100 import build_pt100_mechanism
from mkm.provenance import sha256_file
from mkm.workflows.pt100_fit import (
    Pt100BuiltFit,
    load_pt100_config,
    pt100_likelihood_kwargs,
    pt100_parameter_specs,
    resolve_pt100_fit_specification,
    resolve_pt100_parameter_specs,
)
from mkm.workflows.ptru_posterior import build_pt100_run_metadata, validate_pt100_run_metadata


def split_ptru_loco(selected, electrolyte_M, co_mole_fraction):
    """Hold out every Pt100 replicate/potential in one electrolyte/CO condition."""
    if selected.empty or set(selected["material"]) != {"Pt100"}:
        raise ValueError("Pt100 LOCO requires selected data containing only Pt100.")

    electrolyte_M = float(electrolyte_M)
    co_mole_fraction = float(co_mole_fraction)
    if not np.isfinite(electrolyte_M) or electrolyte_M <= 0:
        raise ValueError("Holdout electrolyte concentration must be finite and positive.")
    if not np.isfinite(co_mole_fraction) or not 0 < co_mole_fraction <= 1:
        raise ValueError("Holdout CO mole fraction must lie in (0, 1].")

    holdout = (
        np.isclose(selected["C_electrolyte_M"].to_numpy(float), electrolyte_M, rtol=0, atol=1e-12)
        & np.isclose(selected["CO_mole_fraction"].to_numpy(float), co_mole_fraction, rtol=0, atol=1e-12)
    )
    if not holdout.any():
        raise ValueError(f"No Pt100 observations at electrolyte={electrolyte_M}, CO={co_mole_fraction}.")
    if holdout.all():
        raise ValueError("LOCO requires at least one training condition.")
    return selected.loc[~holdout].copy(), selected.loc[holdout].copy()


def ptru_validation_output_dir(paths, medium, model_name, electrolyte_M, co_mole_fraction):
    """Keep Pt100 validation results apart from fitted posteriors."""
    # Validate the medium/mechanism using the already-established P1 path contract.
    paths.ptru_posterior_output_dir(medium, model_name)
    concentration = float(electrolyte_M)
    co_fraction = float(co_mole_fraction)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Electrolyte concentration must be finite and positive.")
    if not np.isfinite(co_fraction) or not 0 < co_fraction <= 1:
        raise ValueError("CO mole fraction must lie in (0, 1].")
    condition = f"electrolyte_{concentration:.12g}M__CO_{co_fraction:.12g}"
    return paths.ptru_results_root(medium) / "validation" / "loco" / model_name / condition


def build_ptru_loco_fit(paths, medium, model_name, selected_training, output_dir):
    """Build an actual training-data fit; never reuse the full-data likelihood."""
    config, prior_config = load_pt100_config(paths)
    specification = resolve_pt100_fit_specification(config, medium, model_name)
    if selected_training.empty or set(selected_training["material"]) != {"Pt100"}:
        raise ValueError("Pt100 LOCO training data must contain Pt100 observations only.")
    if "medium" in selected_training and not selected_training["medium"].eq(medium).all():
        raise ValueError("Training data contain observations from another medium.")

    model_data = build_model_data(selected_training, electrolyte_concentration_column="C_electrolyte_M")
    inputs = build_model_input_arrays(model_data)
    physical = resolve_pt100_parameter_specs(prior_config, medium, model_name)
    mechanism = build_pt100_mechanism(model_name, medium, config["temperature_K"], physical)
    built_model = build_pymc_model(
        inputs=inputs,
        mechanism=mechanism,
        likelihood_name=RATE_NORMAL,
        error_structure="shared",
        likelihood_kwargs=pt100_likelihood_kwargs(config),
    )
    return Pt100BuiltFit(
        medium=medium,
        model_name=model_name,
        inputs=inputs,
        built_model=built_model,
        parameter_specs=pt100_parameter_specs(prior_config, specification, config),
        config=config,
        data_path=paths.ptru_selected_path(medium),
        model_config_path=paths.ptru_model_config_path,
        prior_config_path=paths.root / config["priors"]["path"],
        output_dir=output_dir,
        specification=specification,
        model_data=model_data,
    )


def ptru_loco_metadata(paths, fit, sampler, electrolyte_M, co_mole_fraction, n_heldout):
    """Extend P1 provenance with the exact LOCO split and validation code identity."""
    metadata = build_pt100_run_metadata(paths, fit, sampler)
    metadata.update({
        "validation_scheme": "loco",
        "holdout_electrolyte_M": float(electrolyte_M),
        "holdout_co_mole_fraction": float(co_mole_fraction),
        "n_training_observations": len(fit.model_data.observations),
        "n_heldout_observations": int(n_heldout),
        "validation_source_sha256": sha256_file(paths.root / "src/mkm/workflows/ptru_validation.py"),
    })
    return metadata


def validate_ptru_loco_metadata(
    metadata, paths, fit, electrolyte_M, co_mole_fraction, n_heldout, *, allowed_statuses,
):
    validate_pt100_run_metadata(metadata, paths, fit, allowed_statuses=allowed_statuses)
    expected = {
        "validation_scheme": "loco",
        "holdout_electrolyte_M": float(electrolyte_M),
        "holdout_co_mole_fraction": float(co_mole_fraction),
        "n_training_observations": len(fit.model_data.observations),
        "n_heldout_observations": int(n_heldout),
        "validation_source_sha256": sha256_file(paths.root / "src/mkm/workflows/ptru_validation.py"),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"Pt100 LOCO checkpoint split/provenance mismatch: {', '.join(mismatches)}.")
