
"""Shared PtRu measurement construction and normalization."""

import numpy as np
import pandas as pd

from mkm.preprocessing.potential import convert_rhe_to_she


def build_ptru_measurement_frame(
    *,
    medium,
    material,
    concentration_M,
    co_fraction,
    replicate,
    source_workbook,
    source_sheet,
    source_column,
    source_rows,
    potential,
    current,
    sample_id=None,
):
    """Construct the standardized raw-measurement schema used by Todd and Em."""
    values = {
        "medium": medium,
        "material": material,
    }
    if sample_id is not None:
        values["sample_id"] = sample_id

    values.update({
        "C_electrolyte_M": concentration_M,
        "CO_mole_fraction": co_fraction,
        "replicate": replicate,
        "point_index": np.arange(len(potential)),
        "source_workbook": source_workbook,
        "source_sheet": source_sheet,
        "source_column": source_column,
        "source_row": source_rows,
        "E_V_RHE": potential,
        "j_mA_cm2": current,
    })
    return pd.DataFrame(values)


def standardize_ptru_measurements(data, medium, config):
    """Convert RHE potentials and ECSA-normalized currents without merging observations."""
    if medium not in ("acidic", "basic"):
        raise ValueError(f"Unsupported PtRu medium: {medium}")
    if config["activity_model"] != "ideal_molarity":
        raise ValueError("Only ideal_molarity is implemented for PtRu preprocessing.")

    temperature = float(config["temperature_K"])
    pkw = float(config["pKw"])
    norm = config["rate_normalization"]
    electrons = float(norm["electrons_per_CO_oxidation"])
    site_charge = float(norm["site_charge_density_uC_cm2_Pt"])

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if not np.isfinite(pkw):
        raise ValueError("pKw must be finite.")
    if not np.isfinite(electrons * site_charge) or electrons <= 0 or site_charge <= 0:
        raise ValueError("Rate-normalization constants must be finite and positive.")

    result = data.copy()
    if not result["medium"].eq(medium).all():
        raise ValueError("PtRu measurements contain observations from another medium.")

    concentrations = result["C_electrolyte_M"].to_numpy(float)
    if not np.isfinite(concentrations).all() or (concentrations <= 0).any():
        raise ValueError("Electrolyte concentrations must be finite and positive.")

    ln_major = np.log(concentrations)
    ln_kw = -pkw * np.log(10.0)
    ln_a_h = ln_major if medium == "acidic" else ln_kw - ln_major

    result["ln_a_H"] = ln_a_h
    result["ln_a_OH"] = ln_kw - ln_a_h
    result["E_V_SHE"] = convert_rhe_to_she(
        result["E_V_RHE"].to_numpy(float),
        pH=-ln_a_h / np.log(10.0),
        temperature_K=temperature,
    )

    result["rate_s_inv"] = result["j_mA_cm2"].to_numpy(float) * 1000.0 / (electrons * site_charge)
    result["ln_rate"] = np.log(result["rate_s_inv"])

    return result
