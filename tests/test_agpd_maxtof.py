import numpy as np
import pandas as pd

from mkm.preprocessing.agpd_basic_maxtof import truncate_agpd_analysis_maxtof
from mkm.project_paths import ProjectPaths


def _make_test_data():
    potential = np.array([-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04])
    grid_index = np.arange(-3, 5)
    rates = {
        "A": [5.0e-4, 9.0e-4, 1.3e-3, 1.0e-3, 1.5e-3, 2.2e-3, 8.2e-3, 8.0e-4],
        "B": [4.0e-4, 8.0e-4, 1.2e-3, 9.0e-4, 1.4e-3, 2.0e-3, 8.0e-3, 7.0e-4],
        "C": [6.0e-4, 1.0e-3, 1.4e-3, 1.1e-3, 1.6e-3, 2.4e-3, 8.4e-3, 9.0e-4],
    }
    frames = []
    for replicate, replicate_rates in rates.items():
        frames.append(
            pd.DataFrame(
                {
                    "material": "test",
                    "C_KOH_M": 0.25,
                    "CO_mole_fraction": 0.001,
                    "replicate": replicate,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": potential,
                    "rate_s_inv": replicate_rates,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _config():
    return {
        "replicates": ["A", "B", "C"],
        "truncation": {
            "low_potential": {
                "method": "rate_threshold",
                "threshold_s_inv": 1.0e-3,
                "require_all_replicates": True,
            },
            "high_potential": {"method": "max_tof"},
        },
    }


def test_maxtof_truncation_retains_low_cut_through_peak_only():
    selected, cutoffs = truncate_agpd_analysis_maxtof(_make_test_data(), _config())

    np.testing.assert_array_equal(
        np.sort(selected["E_V_SHE"].unique()),
        np.array([0.01, 0.02, 0.03]),
    )
    assert cutoffs.loc[0, "high_potential_method"] == "max_tof"
    np.testing.assert_allclose(cutoffs.loc[0, "retained_min_E_V_SHE"], 0.01)
    np.testing.assert_allclose(cutoffs.loc[0, "retained_max_E_V_SHE"], 0.03)
    np.testing.assert_allclose(cutoffs.loc[0, "peak_E_V_SHE"], 0.03)
    assert cutoffs.loc[0, "n_points_retained"] == 3


def test_maxtof_truncation_preserves_common_replicate_grid():
    selected, _ = truncate_agpd_analysis_maxtof(_make_test_data(), _config())
    expected = np.array([0.01, 0.02, 0.03])

    for replicate in ("A", "B", "C"):
        curve = selected[selected["replicate"] == replicate].sort_values("E_V_SHE")
        np.testing.assert_array_equal(curve["E_V_SHE"].to_numpy(), expected)


def test_maxtof_project_paths_are_separate_from_base(tmp_path):
    base = ProjectPaths(root=tmp_path)
    maxtof = base.with_agpd_data_variant("maxtof")

    assert base.agpd_dataset_name == "AgPd_COOx_basic"
    assert maxtof.agpd_dataset_name == "AgPd_COOx_basic_maxtof"
    assert base.agpd_raw_dir == maxtof.agpd_raw_dir
    assert base.agpd_processed_dir != maxtof.agpd_processed_dir
    assert base.agpd_results_root != maxtof.agpd_results_root
    assert maxtof.agpd_selected_path.name == "AgPd_COOx_basic_maxtof_selected.parquet"
