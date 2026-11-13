
import numpy as np
import pandas as pd
import pytest

from mkm.preprocessing.ptru_selection import select_ptru_analysis


def make_analysis(medium="acidic"):
    rows = []

    for co_fraction in (0.001, 0.01, 0.10):
        for replicate, multiplier in (("1", 1.0), ("2", 1.2)):
            for index, rate in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
                rows.append({
                    "medium": medium,
                    "material": "Pt100",
                    "C_electrolyte_M": 0.50,
                    "CO_mole_fraction": co_fraction,
                    "replicate": replicate,
                    "analysis_grid_index": index,
                    "E_V_SHE": index * 0.01,
                    "rate_s_inv": multiplier * rate,
                })

    return pd.DataFrame(rows)


def make_config():
    return {
        "truncation": {
            "todd": {
                "acidic": {
                    "Pt100": {
                        "0.001": {
                            "method": "fixed_potential",
                            "E_cut_V_SHE_by_electrolyte_M": {"0.5": 0.02},
                        },
                        "0.01": {
                            "method": "fraction_of_max_rate",
                            "fraction": 0.50,
                        },
                    }
                }
            }
        }
    }


def test_ptru_selection_preserves_replicates_and_cutoff_metadata():
    selected, cutoffs = select_ptru_analysis(make_analysis(), make_config())

    assert len(selected) == 22
    assert len(cutoffs) == 3

    expected = {
        0.001: ("fixed_potential", 2, 3),
        0.01: ("fraction_of_max_rate", 2, 3),
        0.10: ("none", 0, 5),
    }

    for co_fraction, (method, cutoff_index, n_points) in expected.items():
        rows = selected.loc[np.isclose(selected["CO_mole_fraction"], co_fraction)]
        cutoff = cutoffs.loc[np.isclose(cutoffs["CO_mole_fraction"], co_fraction)].iloc[0]

        assert cutoff["low_potential_method"] == method
        assert cutoff["low_cut_analysis_grid_index"] == cutoff_index
        assert cutoff["n_points_original"] == 5
        assert cutoff["n_points_retained"] == n_points
        assert cutoff["n_observations_original"] == 10
        assert cutoff["n_observations_retained"] == 2 * n_points

        assert set(rows["replicate"]) == {"1", "2"}
        assert rows["analysis_grid_index"].min() == cutoff_index
        assert len(rows) == 2 * n_points


def test_ptru_basic_has_no_truncation_under_acidic_only_rules():
    analysis = make_analysis(medium="basic")
    selected, cutoffs = select_ptru_analysis(analysis, make_config())

    assert len(selected) == len(analysis)
    assert set(cutoffs["low_potential_method"]) == {"none"}
    assert set(cutoffs["low_cut_analysis_grid_index"]) == {0}


def test_ptru_fixed_cutoff_beyond_data_is_rejected():
    config = make_config()
    config["truncation"]["todd"]["acidic"]["Pt100"]["0.001"][
        "E_cut_V_SHE_by_electrolyte_M"
    ]["0.5"] = 1.0

    with pytest.raises(ValueError, match="Fixed cutoff removes all observations"):
        select_ptru_analysis(make_analysis(), config)


def test_ptru_invalid_fraction_is_rejected():
    config = make_config()
    config["truncation"]["todd"]["acidic"]["Pt100"]["0.01"]["fraction"] = 1.5

    with pytest.raises(ValueError, match="Invalid relative TOF threshold"):
        select_ptru_analysis(make_analysis(), config)


def test_ptru_rejects_fewer_than_three_retained_points():
    config = make_config()
    config["truncation"]["todd"]["acidic"]["Pt100"]["0.001"][
        "E_cut_V_SHE_by_electrolyte_M"
    ]["0.5"] = 0.04

    with pytest.raises(ValueError, match="Fewer than three retained potential points"):
        select_ptru_analysis(make_analysis(), config)
