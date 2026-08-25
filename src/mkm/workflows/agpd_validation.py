"""Cross-validation splits for AgPd composition models."""

import numpy as np


def split_agpd_lomo(selected, material):
    """Hold out every observation from one material."""
    heldout_mask = selected["material"].eq(material)
    if not heldout_mask.any():
        raise ValueError(f"No observations found for held-out material '{material}'.")

    train = selected.loc[~heldout_mask].copy()
    heldout = selected.loc[heldout_mask].copy()
    if train.empty:
        raise ValueError("LOMO training data are empty.")
    return train, heldout


def split_agpd_loco(selected, material, koh_M, co_mole_fraction):
    """Hold out one material/KOH/CO condition, including all replicate curves."""
    heldout_mask = (
        selected["material"].eq(material)
        & np.isclose(selected["C_KOH_M"].to_numpy(dtype=float), float(koh_M), rtol=0.0, atol=1e-12)
        & np.isclose(
            selected["CO_mole_fraction"].to_numpy(dtype=float),
            float(co_mole_fraction),
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not heldout_mask.any():
        raise ValueError(
            f"No observations found for held-out condition: material={material}, "
            f"KOH={koh_M}, CO={co_mole_fraction}."
        )

    train = selected.loc[~heldout_mask].copy()
    heldout = selected.loc[heldout_mask].copy()
    if train.empty:
        raise ValueError("LOCO training data are empty.")

    condition_count = heldout[["material", "C_KOH_M", "CO_mole_fraction"]].drop_duplicates().shape[0]
    if condition_count != 1:
        raise RuntimeError("LOCO split did not isolate exactly one experimental condition.")
    return train, heldout
