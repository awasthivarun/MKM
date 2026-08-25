import pandas as pd

from mkm.workflows.agpd_validation import split_agpd_loco, split_agpd_lomo


def _selected_frame():
    rows = []
    for material in ("Ag10Pd90", "Ag50Pd50"):
        for koh in (0.25, 0.50):
            for co in (0.01, 0.10):
                for replicate in ("A", "B", "C"):
                    rows.append(
                        {
                            "material": material,
                            "C_KOH_M": koh,
                            "CO_mole_fraction": co,
                            "replicate": replicate,
                        }
                    )
    return pd.DataFrame(rows)


def test_loco_removes_all_replicates_for_one_condition():
    train, heldout = split_agpd_loco(_selected_frame(), "Ag10Pd90", 0.25, 0.01)

    assert set(heldout["replicate"]) == {"A", "B", "C"}
    assert len(heldout) == 3
    assert not (
        train["material"].eq("Ag10Pd90")
        & train["C_KOH_M"].eq(0.25)
        & train["CO_mole_fraction"].eq(0.01)
    ).any()


def test_lomo_removes_entire_material():
    train, heldout = split_agpd_lomo(_selected_frame(), "Ag10Pd90")

    assert set(heldout["material"]) == {"Ag10Pd90"}
    assert "Ag10Pd90" not in set(train["material"])
    assert set(train["material"]) == {"Ag50Pd50"}
