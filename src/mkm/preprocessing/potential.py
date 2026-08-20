import numpy as np

from mkm.constants import (
    F_C_mol,
    R_J_mol_K,
)


def nernst_slope_V_per_pH(
    temperature_K,
):
    temperature_K = float(
        temperature_K
    )

    if (
        not np.isfinite(temperature_K)
        or temperature_K <= 0
    ):
        raise ValueError(
            "Temperature must be finite and positive."
        )

    return (
        np.log(10.0)
        * R_J_mol_K
        * temperature_K
        / F_C_mol
    )


def kohl_molarity_to_pH(
    C_KOH_M,
    pKw,
):
    C_KOH_M = np.asarray(
        C_KOH_M,
        dtype=float,
    )

    pKw = float(
        pKw
    )

    if not np.all(
        np.isfinite(C_KOH_M)
    ):
        raise ValueError(
            "KOH concentration contains non-finite values."
        )

    if not np.all(
        C_KOH_M > 0
    ):
        raise ValueError(
            "KOH concentration must be positive."
        )

    if not np.isfinite(pKw):
        raise ValueError(
            "pKw must be finite."
        )

    return (
        pKw
        + np.log10(C_KOH_M)
    )


def convert_rhe_to_she(
    E_V_RHE,
    pH,
    temperature_K,
):
    E_V_RHE = np.asarray(
        E_V_RHE,
        dtype=float,
    )

    pH = np.asarray(
        pH,
        dtype=float,
    )

    if not np.all(
        np.isfinite(E_V_RHE)
    ):
        raise ValueError(
            "RHE potential contains non-finite values."
        )

    if not np.all(
        np.isfinite(pH)
    ):
        raise ValueError(
            "pH contains non-finite values."
        )

    slope = nernst_slope_V_per_pH(
        temperature_K
    )

    return (
        E_V_RHE
        - slope * pH
    )