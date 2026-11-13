
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from mkm.preprocessing.ptru_todd import (
    build_todd_analysis_grid,
    load_todd_dataset,
    load_todd_workbook,
)


@pytest.fixture
def config():
    return {
        "materials": ["Pt100", "Pt66Ru33", "Pt50Ru50"],
        "electrolyte_concentrations_M": [0.10],
        "CO_mole_fractions": [0.001, 0.01],
        "activity_model": "ideal_molarity",
        "temperature_K": 293.15,
        "pKw": 14.167,
        "rate_normalization": {
            "electrons_per_CO_oxidation": 2,
            "site_charge_density_uC_cm2_Pt": 210.0,
        },
        "analysis_grid": {"spacing_V": 0.01, "origin_V": 0.0},
    }


def write_alloy_workbook(path, medium, co_fractions=(0.01,), material="Pt66Ru33"):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = material
    electrolyte = "[HClO4] (M)" if medium == "acidic" else "[KOH] (M)"

    for i, co_fraction in enumerate(co_fractions):
        col = 2 * i + 1
        sheet.cell(1, col, electrolyte)
        sheet.cell(1, col + 1, "P_CO (%)")
        sheet.cell(2, col, 0.10)
        sheet.cell(2, col + 1, 100 * co_fraction)
        sheet.cell(3, col, "E_RHE (V)")
        sheet.cell(3, col + 1, "j (mA/cm2)")

        for row, potential in enumerate([0.70, 0.71, 0.72, 0.73], start=4):
            sheet.cell(row, col, potential)
            sheet.cell(row, col + 1, 2.0 * np.exp(10 * (potential - 0.70)))

    workbook.save(path)


def write_trial_workbook(path):
    workbook = Workbook()
    for trial in (1, 2):
        sheet = workbook.active if trial == 1 else workbook.create_sheet()
        sheet.title = f"100 mM KOH - Trial {trial}"
        sheet.append(["1% CO", None])
        sheet.append(["E (V_RHE)", "j (mA/cm2)"])

        start = 0.70 if trial == 1 else 0.71
        for potential in start + 0.01 * np.arange(4):
            sheet.append([float(potential), float(2.0 * np.exp(10 * (potential - 0.70)))])

    workbook.create_sheet("TC")
    workbook.save(path)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_alloy_workbook_preserves_condition_and_source_columns(tmp_path, config, medium):
    name = f"PtRu_{medium}_current_densities_Todd.xlsx"
    path = tmp_path / name
    write_alloy_workbook(path, medium, co_fractions=(0.001, 0.01))

    data = load_todd_workbook(path, medium, config)

    assert len(data) == 8
    assert set(data["medium"]) == {medium}
    assert set(data["material"]) == {"Pt66Ru33"}
    assert set(data["C_electrolyte_M"]) == {0.10}
    assert set(data["CO_mole_fraction"]) == {0.001, 0.01}
    assert set(data["replicate"]) == {"1"}
    assert set(data["source_workbook"]) == {name}
    assert set(data["source_sheet"]) == {"Pt66Ru33"}
    assert set(data["source_column"]) == {1, 3}
    assert set(data["source_row"]) == {4, 5, 6, 7}
    assert not data.duplicated(
        ["medium", "material", "C_electrolyte_M", "CO_mole_fraction", "replicate", "point_index"]
    ).any()


def test_trial_workbook_preserves_replicates_and_separate_grids(tmp_path, config):
    path = tmp_path / "Pt100_basic_current_densities_Todd.xlsx"
    write_trial_workbook(path)

    data = load_todd_workbook(path, "basic", config)

    # Standardize the synthetic measurements before analysis-grid construction.
    ln_kw = -config["pKw"] * np.log(10)
    ln_h = ln_kw - np.log(data["C_electrolyte_M"].to_numpy(float))
    shift = 8.314462618 * config["temperature_K"] / 96485.33212 * ln_h

    norm = config["rate_normalization"]
    denominator = norm["electrons_per_CO_oxidation"] * norm["site_charge_density_uC_cm2_Pt"]

    data["ln_a_H"] = ln_h
    data["ln_a_OH"] = ln_kw - ln_h
    data["E_V_SHE"] = data["E_V_RHE"] + shift
    data["rate_s_inv"] = data["j_mA_cm2"] * 1000 / denominator
    data["ln_rate"] = np.log(data["rate_s_inv"])

    analysis = build_todd_analysis_grid(data, config)

    assert len(data) == 8
    assert set(data["replicate"]) == {"1", "2"}
    assert set(data["source_sheet"]) == {
        "100 mM KOH - Trial 1",
        "100 mM KOH - Trial 2",
    }

    grids = {
        replicate: group.sort_values("analysis_grid_index")
        for replicate, group in analysis.groupby("replicate")
    }
    assert all(len(group) == 3 for group in grids.values())
    assert grids["1"]["analysis_grid_index"].min() != grids["2"]["analysis_grid_index"].min()

    # PCHIP on a log-linear rate curve must reproduce the log-linear curve.
    for group in grids.values():
        E = group["E_V_SHE"].to_numpy()
        expected = np.log(2.0 * 1000 / 420) + 10.0 * (E - shift[0] - 0.70)
        np.testing.assert_allclose(group["ln_rate"], expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(group["rate_s_inv"], np.exp(expected), rtol=1e-10)


def test_todd_dataset_converts_potential_and_current(tmp_path, config):
    folder = tmp_path / "PtRu_COOx_acidic"
    folder.mkdir()
    path = folder / "PtRu_acidic_current_densities_Todd.xlsx"
    write_alloy_workbook(path, "acidic", material="Pt100")

    config["materials"] = ["Pt100"]
    config["CO_mole_fractions"] = [0.01]
    data = load_todd_dataset(tmp_path, "acidic", config)

    ln_h = np.log(0.10)
    shift = 8.314462618 * 293.15 / 96485.33212 * ln_h

    assert len(data) == 4
    np.testing.assert_allclose(data["ln_a_H"], ln_h)
    np.testing.assert_allclose(data["ln_a_OH"], -14.167 * np.log(10) - ln_h)
    np.testing.assert_allclose(data["E_V_SHE"], data["E_V_RHE"] + shift)
    np.testing.assert_allclose(data["rate_s_inv"], data["j_mA_cm2"] * 1000 / 420)
    np.testing.assert_allclose(data["ln_rate"], np.log(data["rate_s_inv"]))


def test_todd_rejects_invalid_measurements(tmp_path, config):
    path = tmp_path / "PtRu_acidic_current_densities_Todd.xlsx"
    write_alloy_workbook(path, "acidic")

    from openpyxl import load_workbook

    workbook = load_workbook(path)
    workbook.active.cell(5, 2, -1.0)
    workbook.save(path)

    with pytest.raises(ValueError, match="Nonpositive or nonfinite currents"):
        load_todd_workbook(path, "acidic", config)
