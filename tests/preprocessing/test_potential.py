import numpy as np

from mkm.constants import F_C_mol, R_J_mol_K
from mkm.preprocessing.potential import convert_rhe_to_she, kohl_molarity_to_pH, nernst_slope_V_per_pH


def test_nernst_slope():
    temperature_K = 293.15
    expected = np.log(10.0) * R_J_mol_K * temperature_K / F_C_mol

    np.testing.assert_allclose(nernst_slope_V_per_pH(temperature_K), expected)


def test_kohl_molarity_to_pH():
    concentrations = np.array([0.01, 0.10, 1.00])
    result = kohl_molarity_to_pH(concentrations, pKw=14.0)

    np.testing.assert_allclose(result, np.array([12.0, 13.0, 14.0]))


def test_rhe_to_she_conversion():
    temperature_K = 293.15
    pH = 13.0
    E_RHE = np.array([0.0, 0.1, 0.2])

    slope = np.log(10.0) * R_J_mol_K * temperature_K / F_C_mol
    expected = E_RHE - slope * pH
    result = convert_rhe_to_she(E_RHE, pH, temperature_K)

    np.testing.assert_allclose(result, expected)


def test_rhe_to_she_preserves_spacing():
    E_RHE = np.array([-0.10, 0.00, 0.10])
    result = convert_rhe_to_she(E_RHE, pH=13.0, temperature_K=293.15)

    np.testing.assert_allclose(np.diff(result), np.diff(E_RHE))