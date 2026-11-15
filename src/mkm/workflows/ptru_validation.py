"""Pt100-only leave-one-condition-out (LOCO) validation helpers."""


import numpy as np

from mkm.provenance import sha256_file, sha256_mapping
from mkm.workflows.pt100_fit import (
    _build_pt100_fit_from_selected,
    load_pt100_config,
    resolve_pt100_fit_specification,
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

    return _build_pt100_fit_from_selected(
        paths, config, prior_config, specification, selected_training,
        paths.ptru_selected_path(medium), output_dir,
    )


def _model_data_observation_sha256(model_data):
    """Fingerprint the canonical observation ordering and model-point assignments."""
    observations = model_data.observations
    return sha256_mapping({
        "columns": list(observations.columns),
        "rows": observations.to_dict(orient="records"),
    })


def _ptru_loco_identity(paths, fit, electrolyte_M, co_mole_fraction, heldout_data):
    """Identify both subsets, independently of the full selected-data file hash."""
    return {
        "validation_scheme": "loco",
        "holdout_electrolyte_M": float(electrolyte_M),
        "holdout_co_mole_fraction": float(co_mole_fraction),
        "n_training_observations": len(fit.model_data.observations),
        "n_heldout_observations": len(heldout_data.observations),
        "training_observations_sha256": _model_data_observation_sha256(fit.model_data),
        "heldout_observations_sha256": _model_data_observation_sha256(heldout_data),
        "validation_source_sha256": sha256_file(paths.root / "src/mkm/workflows/ptru_validation.py"),
    }


def ptru_loco_metadata(paths, fit, sampler, electrolyte_M, co_mole_fraction, heldout_data):
    """Record source-file provenance and exact canonical LOCO training/holdout subsets."""
    metadata = build_pt100_run_metadata(paths, fit, sampler)
    metadata.update(_ptru_loco_identity(paths, fit, electrolyte_M, co_mole_fraction, heldout_data))
    return metadata


def validate_ptru_loco_metadata(
    metadata, paths, fit, electrolyte_M, co_mole_fraction, heldout_data, *, allowed_statuses,
):
    validate_pt100_run_metadata(metadata, paths, fit, allowed_statuses=allowed_statuses)
    expected = _ptru_loco_identity(paths, fit, electrolyte_M, co_mole_fraction, heldout_data)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"Pt100 LOCO checkpoint split/provenance mismatch: {', '.join(mismatches)}.")
