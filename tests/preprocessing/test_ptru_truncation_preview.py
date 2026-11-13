
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from mkm.preprocessing.ptru_selection import select_ptru_analysis


ROOT = Path(__file__).resolve().parents[2]


def load_preview_script():
    path = ROOT / "scripts" / "plot_ptru_acidic_pt100_truncation.py"
    spec = importlib.util.spec_from_file_location("test_ptru_truncation_preview_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preview_uses_configured_truncation():
    rows = []
    for replicate in ("1", "2"):
        for index, rate in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
            rows.append({
                "medium": "acidic",
                "material": "Pt100",
                "C_electrolyte_M": 0.50,
                "CO_mole_fraction": 0.01,
                "replicate": replicate,
                "analysis_grid_index": index,
                "E_V_SHE": index * 0.01,
                "rate_s_inv": rate,
            })

    analysis = pd.DataFrame(rows)
    config = {
        "truncation": {
            "todd": {
                "acidic": {
                    "Pt100": {
                        "0.01": {
                            "method": "fraction_of_max_rate",
                            "fraction": 0.5,
                        }
                    }
                }
            }
        }
    }

    _, expected = select_ptru_analysis(analysis, config)
    script = load_preview_script()
    cutoffs, preview = script.find_onset_cutoffs(analysis, config)

    assert len(preview) == 1
    assert cutoffs[(0.5, 0.01)] == expected.iloc[0]["E_cut_V_SHE"]
    assert preview.iloc[0]["criterion"] == "relative_max_rate"
    assert preview.iloc[0]["threshold_fraction"] == 0.5
    assert preview.iloc[0]["n_retained"] == expected.iloc[0]["n_points_retained"]
    np.testing.assert_allclose(preview.iloc[0]["E_cut_V_SHE"], 0.02, rtol=0, atol=1e-12)
