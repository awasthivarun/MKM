
"""Read Em's Pt/C CO oxidation data, preserving preparation and replicate identities."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ptru_todd import F, R

EM_CO_FRACTIONS = (0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.00)
EM_WORKBOOKS = {
    "acidic": "Pt100_PtATO_100mM_acidic_current_densities_Em.xlsx",
    "basic": "Pt100_100mM_basic_current_densities_Em.xlsx",
}


def load_em_ptc(path, medium, config):
    path = Path(path)
    header = "[HClO4] (mM)" if medium == "acidic" else "[KOH] (mM)"
    frames = []

    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            if "Pt_ATO" in sheet_name:
                continue

            match = re.fullmatch(r"20% Pt_C( 1mg)? Rep (\d+)", sheet_name)
            if not match:
                raise ValueError(f"Unexpected Em worksheet: {sheet_name}")

            sample_id = "PtC_20pct_1mg" if match[1] else "PtC_20pct"
            replicate = match[2]
            sheet = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

            if sheet.shape[1] != 2 * len(EM_CO_FRACTIONS):
                raise ValueError(f"Unexpected column count in {path.name}/{sheet_name}")

            observed_co = []

            for col in range(0, sheet.shape[1], 2):
                if (sheet.iat[0, col], sheet.iat[0, col + 1]) != (header, "PCO (%)"):
                    raise ValueError(f"Invalid Em condition headers: {sheet_name}, column {col + 1}")
                if str(sheet.iat[1, col]) != "100mM":
                    raise ValueError(f"Unexpected electrolyte concentration in {sheet_name}")

                co_fraction = float(sheet.iat[1, col + 1]) / 100
                if not any(np.isclose(co_fraction, x) for x in EM_CO_FRACTIONS):
                    raise ValueError(f"Unexpected CO fraction in {sheet_name}: {co_fraction}")

                if sheet.iat[2, col] != "E_RHE (V)" or sheet.iat[2, col + 1] != "j (mA/cm²)":
                    raise ValueError(f"Invalid potential/current headers in {sheet_name}")

                e_raw = sheet.iloc[3:, col]
                j_raw = sheet.iloc[3:, col + 1]
                e = pd.to_numeric(e_raw, errors="coerce")
                j = pd.to_numeric(j_raw, errors="coerce")

                if (e_raw.notna() & e.isna()).any() or (j_raw.notna() & j.isna()).any():
                    raise ValueError(f"Nonnumeric measurement in {sheet_name}")
                if (e.isna() & j.notna()).any():
                    raise ValueError(f"Current without potential in {sheet_name}")

                # Blank currents are expected at endpoints and occasionally near the upper endpoint.
                valid = e.notna() & j.notna()
                potential = e[valid].to_numpy(float)
                current = j[valid].to_numpy(float)

                if len(potential) < 2 or not np.all(np.diff(potential) > 0):
                    raise ValueError(f"Invalid potential ordering in {sheet_name}")
                if not np.all(np.isfinite(current) & (current > 0)):
                    raise ValueError(f"Invalid current density in {sheet_name}")

                frames.append(pd.DataFrame({
                    "medium": medium,
                    "material": "Pt100",
                    "sample_id": sample_id,
                    "C_electrolyte_M": 0.10,
                    "CO_mole_fraction": co_fraction,
                    "replicate": replicate,
                    "point_index": np.arange(len(potential)),
                    "source_workbook": path.name,
                    "source_sheet": sheet_name,
                    "source_column": col + 1,
                    "source_row": e.index[valid].to_numpy() + 1,
                    "E_V_RHE": potential,
                    "j_mA_cm2": current,
                }))
                observed_co.append(co_fraction)

            if len(observed_co) != len(EM_CO_FRACTIONS) or not np.allclose(
                sorted(observed_co), EM_CO_FRACTIONS
            ):
                raise ValueError(f"Incomplete CO condition series in {sheet_name}")

    if not frames:
        raise ValueError(f"No Pt/C measurements found in {path.name}")

    data = pd.concat(frames, ignore_index=True)

    if config["activity_model"] != "ideal_molarity":
        raise ValueError("Only ideal_molarity is currently implemented")

    ln_major = np.log(0.10)
    ln_kw = -float(config["pKw"]) * np.log(10)
    ln_a_h = ln_major if medium == "acidic" else ln_kw - ln_major

    data["ln_a_H"] = ln_a_h
    data["ln_a_OH"] = ln_kw - ln_a_h
    data["E_V_SHE"] = data["E_V_RHE"] + R * config["temperature_K"] / F * ln_a_h

    norm = config["rate_normalization"]
    denominator = norm["electrons_per_CO_oxidation"] * norm["site_charge_density_uC_cm2_Pt"]

    data["rate_s_inv"] = data["j_mA_cm2"] * 1000 / denominator
    data["ln_rate"] = np.log(data["rate_s_inv"])

    return data
