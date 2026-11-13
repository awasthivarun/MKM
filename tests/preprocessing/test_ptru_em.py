
import numpy as np
import pytest
from openpyxl import Workbook

from mkm.preprocessing.ptru_em import EM_CO_FRACTIONS, load_em_ptc
from mkm.preprocessing.ptru_todd import build_todd_analysis_grid


@pytest.fixture
def config():
    return {
        "activity_model": "ideal_molarity",
        "temperature_K": 293.15,
        "pKw": 14.167,
        "rate_normalization": {
            "electrons_per_CO_oxidation": 2,
            "site_charge_density_uC_cm2_Pt": 210.0,
        },
        "analysis_grid": {"spacing_V": 0.01, "origin_V": 0.0},
    }


def write_em_workbook(path, medium):
    workbook = Workbook()
    header = "[HClO4] (mM)" if medium == "acidic" else "[KOH] (mM)"

    for i, name in enumerate(("20% Pt_C Rep 1", "20% Pt_C 1mg Rep 2")):
        sheet = workbook.active if i == 0 else workbook.create_sheet()
        sheet.title = name

        for j, co_fraction in enumerate(EM_CO_FRACTIONS):
            col = 2 * j + 1
            sheet.cell(1, col, header)
            sheet.cell(1, col + 1, "PCO (%)")
            sheet.cell(2, col, "100mM")
            sheet.cell(2, col + 1, 100 * co_fraction)
            sheet.cell(3, col, "E_RHE (V)")
            sheet.cell(3, col + 1, "j (mA/cm²)")

            for row, E in enumerate([0.70, 0.71, 0.72, 0.73], start=4):
                sheet.cell(row, col, E)
                current = 1.0 + i + j / 10 + (row - 4) / 10
                sheet.cell(row, col + 1, current)

        # Em accepts a missing endpoint current, but retains valid rows.
        sheet.cell(7, 2, None)
        sheet.cell(7, 2).value = None

    workbook.create_sheet("Pt_ATO control")
    workbook.save(path)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_em_preserves_preparation_replicate_and_source(tmp_path, config, medium):
    path = tmp_path / f"em_{medium}.xlsx"
    write_em_workbook(path, medium)

    data = load_em_ptc(path, medium, config)

    assert len(data) == 54
    assert set(data["medium"]) == {medium}
    assert set(data["material"]) == {"Pt100"}
    assert set(data["sample_id"]) == {"PtC_20pct", "PtC_20pct_1mg"}
    assert set(data["replicate"]) == {"1", "2"}
    assert set(data["CO_mole_fraction"]) == set(EM_CO_FRACTIONS)
    assert set(data["C_electrolyte_M"]) == {0.10}
    assert set(data["source_workbook"]) == {path.name}
    assert not data["source_sheet"].str.contains("Pt_ATO").any()

    assert not data.duplicated(
        ["sample_id", "replicate", "CO_mole_fraction", "point_index"]
    ).any()

    first = data.loc[
        (data["sample_id"] == "PtC_20pct") & np.isclose(data["CO_mole_fraction"], 0.001)
    ]
    assert len(first) == 3
    assert set(first["source_row"]) == {4, 5, 6}

    np.testing.assert_allclose(
        data["rate_s_inv"], data["j_mA_cm2"] * 1000 / 420, rtol=1e-12,
    )
    np.testing.assert_allclose(data["ln_rate"], np.log(data["rate_s_inv"]))


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_em_potential_conversion(tmp_path, config, medium):
    path = tmp_path / f"em_{medium}.xlsx"
    write_em_workbook(path, medium)
    data = load_em_ptc(path, medium, config)

    ln_major = np.log(0.10)
    ln_kw = -14.167 * np.log(10)
    ln_h = ln_major if medium == "acidic" else ln_kw - ln_major
    shift = 8.314462618 * 293.15 / 96485.33212 * ln_h

    np.testing.assert_allclose(data["ln_a_H"], ln_h)
    np.testing.assert_allclose(data["ln_a_OH"], ln_kw - ln_h)
    np.testing.assert_allclose(data["E_V_SHE"], data["E_V_RHE"] + shift)


def test_em_grids_can_be_generated_per_preparation(tmp_path, config):
    path = tmp_path / "em_basic.xlsx"
    write_em_workbook(path, "basic")
    data = load_em_ptc(path, "basic", config)

    for sample_id, sample in data.groupby("sample_id"):
        analysis = build_todd_analysis_grid(sample, config)

        assert not analysis.empty
        assert set(analysis["material"]) == {"Pt100"}
        assert set(analysis["CO_mole_fraction"]) == set(EM_CO_FRACTIONS)
        assert np.all(analysis["rate_s_inv"] > 0)
        assert np.isfinite(analysis["ln_rate"]).all()

        for _, curve in analysis.groupby("CO_mole_fraction"):
            E = curve.sort_values("E_V_SHE")["E_V_SHE"].to_numpy()
            np.testing.assert_allclose(np.diff(E), 0.01, rtol=0, atol=1e-12)


def test_em_rejects_current_without_potential(tmp_path, config):
    path = tmp_path / "em_acidic.xlsx"
    write_em_workbook(path, "acidic")

    from openpyxl import load_workbook

    workbook = load_workbook(path)
    workbook.active.cell(4, 1).value = None
    workbook.save(path)

    with pytest.raises(ValueError, match="Current without potential"):
        load_em_ptc(path, "acidic", config)
