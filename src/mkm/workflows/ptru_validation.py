"""PtRu individual-material leave-one-condition-out validation helpers."""

import numpy as np

from mkm.provenance import model_data_observation_sha256, sha256_file
from mkm.workflows.ptru_fit import build_ptru_fit, load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import build_ptru_run_metadata, validate_ptru_run_metadata


def split_ptru_loco(selected, electrolyte_M, co_mole_fraction):
    """Hold out all replicates and potentials for one electrolyte/CO condition."""
    if selected.empty or selected["material"].nunique() != 1:
        raise ValueError("PtRu individual LOCO requires observations from exactly one material.")
    concentration, co_fraction = float(electrolyte_M), float(co_mole_fraction)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Holdout electrolyte concentration must be finite and positive.")
    if not np.isfinite(co_fraction) or not 0 < co_fraction <= 1:
        raise ValueError("Holdout CO mole fraction must be in (0, 1].")
    holdout = (
        np.isclose(selected["C_electrolyte_M"].to_numpy(float), concentration, rtol=0, atol=1e-12)
        & np.isclose(selected["CO_mole_fraction"].to_numpy(float), co_fraction, rtol=0, atol=1e-12)
    )
    if not holdout.any() or holdout.all():
        raise ValueError("LOCO requires both training and held-out observations.")
    return selected.loc[~holdout].copy(), selected.loc[holdout].copy()


def ptru_validation_output_dir(paths, medium, model_name, electrolyte_M, co_mole_fraction, material="Pt100"):
    paths.ptru_posterior_output_dir(medium, model_name, material)
    concentration, co_fraction = float(electrolyte_M), float(co_mole_fraction)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Electrolyte concentration must be positive and finite.")
    if not np.isfinite(co_fraction) or not 0 < co_fraction <= 1:
        raise ValueError("CO mole fraction must lie in (0, 1].")
    label = f"electrolyte_{concentration:.12g}M__CO_{co_fraction:.12g}"
    return paths.ptru_results_root(medium) / "validation" / "loco" / material / model_name / label


def build_ptru_loco_fit(paths, medium, model_name, selected_training, output_dir, material="Pt100"):
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(config, medium, model_name, material=material)
    if selected_training.empty or set(selected_training["material"]) != {material}:
        raise ValueError("LOCO training observations must contain only the requested PtRu material.")
    return build_ptru_fit(paths, spec, selected=selected_training, output_dir=output_dir)


_model_data_observation_sha256 = model_data_observation_sha256


def _ptru_loco_identity(paths, fit, electrolyte_M, co_mole_fraction, heldout_data):
    return {
        "validation_scheme": "loco", "holdout_electrolyte_M": float(electrolyte_M),
        "holdout_co_mole_fraction": float(co_mole_fraction),
        "n_training_observations": len(fit.model_data.observations),
        "n_heldout_observations": len(heldout_data.observations),
        "training_observations_sha256": _model_data_observation_sha256(fit.model_data),
        "heldout_observations_sha256": _model_data_observation_sha256(heldout_data),
        "validation_source_sha256": sha256_file(paths.root / "src/mkm/workflows/ptru_validation.py"),
    }


def ptru_loco_metadata(paths, fit, sampler, electrolyte_M, co_mole_fraction, heldout_data):
    metadata = build_ptru_run_metadata(paths, fit, sampler)
    metadata.update(_ptru_loco_identity(paths, fit, electrolyte_M, co_mole_fraction, heldout_data))
    return metadata


def validate_ptru_loco_metadata(
    metadata, paths, fit, electrolyte_M, co_mole_fraction, heldout_data, *, allowed_statuses,
):
    validate_ptru_run_metadata(metadata, paths, fit, allowed_statuses=allowed_statuses)
    expected = _ptru_loco_identity(paths, fit, electrolyte_M, co_mole_fraction, heldout_data)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError("PtRu LOCO checkpoint split/provenance mismatch: " + ", ".join(mismatches))
