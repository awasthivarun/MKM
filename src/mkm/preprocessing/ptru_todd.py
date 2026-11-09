
"""Preprocess Todd's Pt/PtRu CO oxidation workbooks without collapsing trials."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

R, F = 8.314462618, 96485.33212

TODD_FILES = {
    "acidic": ("PtRu_acidic_current_densities_Todd.xlsx",),
    "basic": ("Pt100_basic_current_densities_Todd.xlsx", "PtRu_basic_current_densities_Todd.xlsx"),
}


def load_todd_workbook(path, medium, config):
    path = Path(path)
    is_pt_trials = path.name == "Pt100_basic_current_densities_Todd.xlsx"
    frames = []

    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            if is_pt_trials and sheet_name in {"TC", "CO order", "OH- Order"}:
                continue

            sheet = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

            if is_pt_trials:
                match = re.fullmatch(r"(\d+) mM KOH - Trial (\d+)", sheet_name)
                if not match:
                    raise ValueError(f"Unrecognized trial worksheet: {sheet_name}")

                C_M, replicate = int(match[1]) / 1000, match[2]
                material, header_row, start_row = "Pt100", 1, 2
            else:
                material, replicate, header_row, start_row = sheet_name, "1", 2, 3
                if material not in config["materials"]:
                    raise ValueError(f"Unrecognized material worksheet: {sheet_name}")

            if sheet.shape[1] % 2:
                raise ValueError(f"Expected potential/current column pairs in {path.name}/{sheet_name}")

            for col in range(0, sheet.shape[1], 2):
                if is_pt_trials:
                    match = re.fullmatch(r"([\d.]+)% CO", str(sheet.iat[0, col]))
                    if not match:
                        raise ValueError(f"Unrecognized CO header in {path.name}/{sheet_name}")

                    x_co = float(match[1]) / 100
                else:
                    C_M = float(sheet.iat[1, col])
                    x_co = float(sheet.iat[1, col + 1]) / 100
                    expected = "[HClO4] (M)" if medium == "acidic" else "[KOH] (M)"

                    if sheet.iat[0, col] != expected or sheet.iat[0, col + 1] != "P_CO (%)":
                        raise ValueError(f"Unexpected condition headers in {path.name}/{sheet_name}")

                if not any(np.isclose(C_M, x) for x in config["electrolyte_concentrations_M"]):
                    raise ValueError(f"Unexpected electrolyte concentration: {C_M}")
                if not any(np.isclose(x_co, x) for x in config["CO_mole_fractions"]):
                    raise ValueError(f"Unexpected CO fraction: {x_co}")

                if sheet.iat[header_row, col] not in {"E_RHE (V)", "E (V_RHE)"}:
                    raise ValueError(f"Unexpected potential header in {path.name}/{sheet_name}")
                if sheet.iat[header_row, col + 1] != "j (mA/cm2)":
                    raise ValueError(f"Unexpected current header in {path.name}/{sheet_name}")

                e_raw = sheet.iloc[start_row:, col]
                j_raw = sheet.iloc[start_row:, col + 1]
                e = pd.to_numeric(e_raw, errors="coerce")
                j = pd.to_numeric(j_raw, errors="coerce")

                if (e_raw.notna() & e.isna()).any() or (j_raw.notna() & j.isna()).any():
                    raise ValueError(f"Nonnumeric measurement in {path.name}/{sheet_name}")
                if (e.isna() != j.isna()).any():
                    raise ValueError(f"Unpaired potential/current values in {path.name}/{sheet_name}")

                valid = e.notna() & j.notna()
                potential = e[valid].to_numpy(float)
                current = j[valid].to_numpy(float)

                if len(potential) < 2 or not np.all(np.diff(potential) > 0):
                    raise ValueError(f"Invalid potential trace in {path.name}/{sheet_name}, column {col + 1}")
                if not np.all(np.isfinite(current) & (current > 0)):
                    raise ValueError(f"Nonpositive or nonfinite currents in {path.name}/{sheet_name}")

                frames.append(pd.DataFrame({
                    "medium": medium,
                    "material": material,
                    "C_electrolyte_M": C_M,
                    "CO_mole_fraction": x_co,
                    "replicate": replicate,
                    "point_index": np.arange(len(potential)),
                    "source_workbook": path.name,
                    "source_sheet": sheet_name,
                    "source_column": col + 1,
                    "source_row": e.index[valid].to_numpy() + 1,
                    "E_V_RHE": potential,
                    "j_mA_cm2": current,
                }))

    return pd.concat(frames, ignore_index=True)


def load_todd_dataset(raw_root, medium, config):
    folder = Path(raw_root) / f"PtRu_COOx_{medium}"

    if config["activity_model"] != "ideal_molarity":
        raise ValueError("Only ideal_molarity is implemented for this first preprocessing pass")

    data = pd.concat(
        [load_todd_workbook(folder / name, medium, config) for name in TODD_FILES[medium]],
        ignore_index=True,
    )

    keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "replicate", "point_index"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate PtRu observations in the raw workbook structure")

    expected = {
        (material, conc, co, trial)
        for material in config["materials"]
        for conc in config["electrolyte_concentrations_M"]
        for co in config["CO_mole_fractions"]
        for trial in (("1", "2", "3", "4") if medium == "basic" and material == "Pt100" else ("1",))
    }

    observed = set(
        data[["material", "C_electrolyte_M", "CO_mole_fraction", "replicate"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            f"Missing or unexpected Todd conditions: missing={expected - observed}, "
            f"unexpected={observed - expected}"
        )

    if (
        config["temperature_K"] <= 0
        or config["rate_normalization"]["electrons_per_CO_oxidation"] <= 0
        or config["rate_normalization"]["site_charge_density_uC_cm2_Pt"] <= 0
    ):
        raise ValueError("Temperature and normalization constants must be positive")

    # Ideal major-ion activities; revise when literature activity coefficients are introduced.
    ln_major = np.log(data["C_electrolyte_M"].to_numpy(float))
    ln_kw = -float(config["pKw"]) * np.log(10)
    ln_a_h = ln_major if medium == "acidic" else ln_kw - ln_major
    ln_a_oh = ln_kw - ln_a_h

    data["ln_a_H"] = ln_a_h
    data["ln_a_OH"] = ln_a_oh
    data["E_V_SHE"] = data["E_V_RHE"] + R * config["temperature_K"] / F * ln_a_h

    norm = config["rate_normalization"]
    denominator = norm["electrons_per_CO_oxidation"] * norm["site_charge_density_uC_cm2_Pt"]

    data["rate_s_inv"] = data["j_mA_cm2"] * 1000 / denominator
    data["ln_rate"] = np.log(data["rate_s_inv"])

    return data


def build_todd_analysis_grid(standardized, config):
    spacing = float(config["analysis_grid"]["spacing_V"])
    origin = float(config["analysis_grid"]["origin_V"])
    if spacing <= 0:
        raise ValueError("Analysis-grid spacing must be positive")

    keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "replicate"]
    frames = []

    for _, curve in standardized.groupby(keys, sort=False):
        curve = curve.sort_values("E_V_SHE")
        e = curve["E_V_SHE"].to_numpy()

        lo = int(np.ceil((e[0] - origin - 1e-10) / spacing))
        hi = int(np.floor((e[-1] - origin + 1e-10) / spacing))
        if lo > hi:
            raise ValueError(f"No analysis-grid points inside trace {curve.iloc[0][keys].to_dict()}")

        index = np.arange(lo, hi + 1)
        grid = origin + index * spacing
        log_rate = PchipInterpolator(e, curve["ln_rate"].to_numpy(), extrapolate=False)(grid)

        if not np.isfinite(log_rate).all():
            raise ValueError("Interpolation produced nonfinite log rates")

        first = curve.iloc[0]
        frame = pd.DataFrame({k: first[k] for k in keys}, index=np.arange(len(grid)))

        for k in ("source_workbook", "source_sheet", "source_column", "ln_a_H", "ln_a_OH"):
            frame[k] = first[k]

        frame["analysis_grid_index"] = index
        frame["E_V_SHE"] = grid
        frame["ln_rate"] = log_rate
        frame["rate_s_inv"] = np.exp(log_rate)

        norm = config["rate_normalization"]
        frame["j_mA_cm2"] = (
            frame["rate_s_inv"]
            * norm["electrons_per_CO_oxidation"]
            * norm["site_charge_density_uC_cm2_Pt"] / 1000
        )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
