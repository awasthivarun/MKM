
"""Preprocess Todd's Pt/PtRu CO oxidation workbooks without collapsing trials."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mkm.preprocessing.ptru_common import build_ptru_measurement_frame, standardize_ptru_measurements
from mkm.preprocessing.interpolation import build_potential_grid, interpolate_log_values

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

                frames.append(build_ptru_measurement_frame(
                    medium=medium,
                    material=material,
                    concentration_M=C_M,
                    co_fraction=x_co,
                    replicate=replicate,
                    source_workbook=path.name,
                    source_sheet=sheet_name,
                    source_column=col + 1,
                    source_rows=e.index[valid].to_numpy() + 1,
                    potential=potential,
                    current=current,
                ))


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

    return standardize_ptru_measurements(data, medium, config)


def build_todd_analysis_grid(standardized, config):
    spacing = float(config["analysis_grid"]["spacing_V"])
    origin = float(config["analysis_grid"]["origin_V"])
    if spacing <= 0:
        raise ValueError("Analysis-grid spacing must be positive")

    keys = ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "replicate"]
    frames = []

    for _, curve in standardized.groupby(keys, sort=False):
        curve = curve.sort_values("E_V_SHE")
        potential = curve["E_V_SHE"].to_numpy(float)

        try:
            index, grid = build_potential_grid(potential[0], potential[-1], spacing, origin)
        except ValueError as exc:
            raise ValueError(
                f"No analysis-grid points inside trace {curve.iloc[0][keys].to_dict()}"
            ) from exc

        log_rate = interpolate_log_values(
            potential, curve["ln_rate"].to_numpy(float), grid,
        )

        first = curve.iloc[0]
        frame = pd.DataFrame({key: first[key] for key in keys}, index=np.arange(len(grid)))

        for key in ("source_workbook", "source_sheet", "source_column", "ln_a_H", "ln_a_OH"):
            frame[key] = first[key]

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

