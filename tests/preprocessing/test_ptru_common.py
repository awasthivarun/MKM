
import numpy as np
import pytest

from mkm.constants import F_C_mol, R_J_mol_K
from mkm.preprocessing.ptru_common import (
    build_ptru_measurement_frame,
    standardize_ptru_measurements,
)


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
    }


def make_measurements(medium, sample_id=None):
    return build_ptru_measurement_frame(
        medium=medium,
        material="Pt100",
        sample_id=sample_id,
        concentration_M=0.50,
        co_fraction=0.01,
        replicate="1",
        source_workbook="synthetic.xlsx",
        source_sheet="Pt100",
        source_column=1,
        source_rows=np.array([4, 5, 6]),
        potential=np.array([0.70, 0.71, 0.72]),
        current=np.array([1.0, 2.0, 4.0]),
    )


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("sample_id", [None, "PtC_20pct"])
def test_common_standardization_preserves_measurements(config, medium, sample_id):
    raw = make_measurements(medium, sample_id)
    result = standardize_ptru_measurements(raw, medium, config)

    assert len(result) == len(raw)
    assert (result["medium"] == medium).all()
    assert (result["material"] == "Pt100").all()
    assert result["source_row"].tolist() == [4, 5, 6]

    if sample_id is None:
        assert "sample_id" not in result.columns
    else:
        assert set(result["sample_id"]) == {sample_id}

    ln_major = np.log(0.50)
    ln_kw = -config["pKw"] * np.log(10.0)
    ln_h = ln_major if medium == "acidic" else ln_kw - ln_major

    expected_E = raw["E_V_RHE"] + R_J_mol_K * 293.15 / F_C_mol * ln_h
    expected_rate = raw["j_mA_cm2"] * 1000.0 / 420.0

    np.testing.assert_allclose(result["ln_a_H"], ln_h, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["ln_a_OH"], ln_kw - ln_h, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["E_V_SHE"], expected_E, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["rate_s_inv"], expected_rate, rtol=1e-12)
    np.testing.assert_allclose(result["ln_rate"], np.log(expected_rate), rtol=1e-12)


def test_common_standardization_rejects_mixed_media(config):
    raw = make_measurements("acidic")
    raw.loc[0, "medium"] = "basic"

    with pytest.raises(ValueError, match="another medium"):
        standardize_ptru_measurements(raw, "acidic", config)


def test_common_standardization_rejects_nonideal_activity_model(config):
    config["activity_model"] = "nonideal"

    with pytest.raises(ValueError, match="ideal_molarity"):
        standardize_ptru_measurements(make_measurements("basic"), "basic", config)


def test_common_standardization_rejects_invalid_normalization(config):
    config["rate_normalization"]["site_charge_density_uC_cm2_Pt"] = 0.0

    with pytest.raises(ValueError, match="normalization"):
        standardize_ptru_measurements(make_measurements("acidic"), "acidic", config)
