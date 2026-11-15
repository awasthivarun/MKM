"""Shared AgPd/PtRu condition-overlay and LOO figure orchestration regression tests."""

from types import SimpleNamespace

import pandas as pd
import pytest

from mkm.postprocessing.plotting import observables as overlay
from mkm.workflows import ptru_postprocess as ptru


def _overlay_frames():
    alpha = []
    co = []
    ion = []
    rates = []
    summary = {
        "median": 0.3, "hdi80_lower": 0.25, "hdi80_upper": 0.35,
        "hdi95_lower": 0.2, "hdi95_upper": 0.4,
    }
    for electrolyte in (0.1, 0.5):
        for fraction in (0.01, 0.10):
            for potential in (0.2, 0.3):
                point = {
                    "material": "Pt100", "C_electrolyte_M": electrolyte,
                    "CO_mole_fraction": fraction, "E_V_SHE": potential,
                }
                alpha.append({**point, **summary, "alpha_mean": 0.31, "alpha_sd": 0.02})
                for replicate, offset in (("A", 0.0), ("B", 0.02)):
                    rates.append({
                        **point, "electrolyte_concentration_M": electrolyte, "replicate": replicate,
                        "rate": 0.5 + offset, "rate_model_median": 0.5,
                        "rate_model_hdi80_lower": 0.4, "rate_model_hdi80_upper": 0.6,
                        "rate_model_hdi95_lower": 0.3, "rate_model_hdi95_upper": 0.7,
                    })
        for potential in (0.2, 0.3):
            co.append({
                "material": "Pt100", "C_electrolyte_M": electrolyte,
                "CO_lower_mole_fraction": 0.01, "CO_upper_mole_fraction": 0.1,
                "E_V_SHE": potential, **summary, "delta_CO": 0.1, "delta_CO_sd": 0.03,
            })
    for fraction in (0.01, 0.10):
        for potential in (0.2, 0.3):
            ion.append({
                "material": "Pt100", "CO_mole_fraction": fraction, "E_V_SHE": potential,
                **summary, "delta_Hplus": -0.1, "delta_Hplus_sd": 0.03,
            })
    return pd.DataFrame(alpha), pd.DataFrame(co), pd.DataFrame(ion), pd.DataFrame(rates)


def test_shared_overlays_accept_acidic_electrolyte_and_hplus_columns(tmp_path, monkeypatch):
    alpha, co, ion, rates = _overlay_frames()
    labels = []
    original = overlay._finish_overlay_grid

    def capture_grid(*args, **kwargs):
        labels.append((kwargs["title"], kwargs["ylabel"], kwargs["condition_labels"], kwargs.get("row_labels")))
        return original(*args, **kwargs)

    monkeypatch.setattr(overlay, "_finish_overlay_grid", capture_grid)
    overlay.plot_alpha_overlay_pco(alpha, tmp_path / "alpha.png",
                                   electrolyte_column="C_electrolyte_M", electrolyte_label="HClO4")
    overlay.plot_delta_co_overlay_koh(co, tmp_path / "co.png",
                                      electrolyte_column="C_electrolyte_M", electrolyte_label="HClO4")
    overlay.plot_delta_oh_overlay_pco(ion, tmp_path / "hplus.png",
                                      ion_name="delta_Hplus", ion_label="H+")
    overlay.plot_rate_overlay_koh(rates, tmp_path / "rate.png", electrolyte_label="HClO4")
    assert all((tmp_path / name).is_file() for name in ("alpha.png", "co.png", "hplus.png", "rate.png"))
    assert any(rows is not None and "0.1 M HClO4" in rows for _, _, _, rows in labels)
    assert any("0.5 M HClO4" in conditions for _, _, conditions, _ in labels)
    assert any(title == "H+ reaction order" and ylabel == "H+ order" for title, ylabel, _, _ in labels)


@pytest.mark.parametrize("medium,ion_name,electrolyte", [
    ("acidic", "delta_Hplus", "HClO4"), ("basic", "delta_OH", "KOH"),
])
def test_ptru_full_reuses_shared_overlay_and_loo_plots(tmp_path, monkeypatch, medium, ion_name, electrolyte):
    calls = []

    def record_plot(status_rows, name, function, *args, **kwargs):
        calls.append((name, function.__name__, args, kwargs))
        return True

    monkeypatch.setattr(ptru, "safe_plot", record_plot)
    run = SimpleNamespace(
        specification=SimpleNamespace(medium=medium, model_name="CO_LH"),
        inference_data=SimpleNamespace(posterior=None), parameter_specs={}, free_parameter_names=(),
    )
    observations = pd.DataFrame({"material": ["Pt100"], "rate": [0.5]})
    points = pd.DataFrame({"rate_fraction_LH_median": [1.0]})
    observable_points = pd.DataFrame({"observable": ["alpha", "delta_CO", ion_name]})
    loo = SimpleNamespace(pointwise=pd.DataFrame({"material": ["Pt100"]}), loo_result=object())
    calibration = SimpleNamespace(pointwise=pd.DataFrame({"material": ["Pt100"], "loo_pit": [0.5]}))

    status = []
    ptru.plot_ptru_products(
        run, observations, points, tmp_path / medium, "full", status,
        observable_points=observable_points, loo=loo, calibration=calibration,
    )
    calls_by_name = {name: (function_name, args, kwargs) for name, function_name, args, kwargs in calls}
    for name in (
        "alpha_overlay_pco", "alpha_overlay_electrolyte", "delta_CO_overlay_pco",
        "delta_CO_overlay_electrolyte", f"{ion_name}_overlay_pco",
        "rate_overlay_pco", "rate_overlay_electrolyte", "Pt100/loo_pit",
        "Pt100/loo_pointwise_pit", "loo_diagnostics",
    ):
        assert name in calls_by_name
    assert calls_by_name["alpha_overlay_pco"][2]["electrolyte_label"] == electrolyte
    assert calls_by_name[f"{ion_name}_overlay_pco"][2]["ion_name"] == ion_name
    assert calls_by_name["Pt100/loo_pointwise_pit"][2]["electrolyte_label"] == electrolyte
    assert status[-1]["status"] == "complete"


def test_ptru_loo_without_pit_keeps_pointwise_plot_and_diagnostics(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ptru, "safe_plot", lambda _, name, *args, **kwargs: calls.append(name) or True)
    run = SimpleNamespace(
        specification=SimpleNamespace(medium="acidic", model_name="CO_LH"),
        inference_data=SimpleNamespace(posterior=None), parameter_specs={}, free_parameter_names=(),
    )
    loo = SimpleNamespace(pointwise=pd.DataFrame({"material": ["Pt100"]}), loo_result=object())
    ptru.plot_ptru_products(
        run, pd.DataFrame(), pd.DataFrame(), tmp_path / "figures", "full", [], loo=loo,
    )
    assert "Pt100/loo_pointwise" in calls
    assert "loo_diagnostics" in calls
    assert "Pt100/loo_pit" not in calls


@pytest.mark.parametrize("group_columns,key,sort_potential", [
    (("material", "C_KOH_M", "CO_mole_fraction"), ("Pd100", 0.5, 0.1), True),
    (("material", "C_KOH_M", "CO_lower_mole_fraction", "CO_upper_mole_fraction"),
     ("Pd100", 0.5, 0.01, 0.1), True),
    (("material", "C_KOH_M", "CO_mole_fraction"), ("Pd100", 0.5, 0.1), False),
])
def test_indexed_overlay_condition_rows_match_boolean_filter(group_columns, key, sort_potential):
    frame = pd.DataFrame({
        "material": ["Pd100"] * 5 + ["Ag10Pd90"],
        "C_KOH_M": [0.5, 0.25, 0.5, 0.5, float("nan"), 0.5],
        "CO_mole_fraction": [0.1, 0.1, 0.1, 0.01, 0.1, 0.1],
        "CO_lower_mole_fraction": [0.01, 0.01, 0.01, 0.001, 0.01, 0.01],
        "CO_upper_mole_fraction": [0.1, 0.1, 0.1, 0.01, 0.1, 0.1],
        "E_V_SHE": [0.3, 0.2, 0.1, 0.4, 0.5, 0.6],
        "replicate": ["A", "B", "C", "A", "A", "A"],
    })
    groups = frame.groupby(list(group_columns), sort=False).indices
    expected_mask = pd.Series(True, index=frame.index)
    for column, value in zip(group_columns, key, strict=True):
        expected_mask &= frame[column].eq(value)
    expected = frame.loc[expected_mask]
    if sort_potential:
        expected = expected.sort_values("E_V_SHE")
    result = overlay._overlay_condition_rows(frame, groups, key, sort_potential=sort_potential)
    pd.testing.assert_frame_equal(result, expected)

    missing_key = ("Ag10Pd90", *key[1:-1], 0.99)
    missing = overlay._overlay_condition_rows(frame, groups, missing_key, sort_potential=sort_potential)
    assert missing.empty
    assert missing.columns.equals(frame.columns)


def test_shared_condition_overlay_variants_save_figures(tmp_path):
    alpha, co, _, rates = _overlay_frames()
    overlay.plot_alpha_overlay_koh(alpha, tmp_path / "alpha_electrolyte.png", electrolyte_column="C_electrolyte_M")
    overlay.plot_delta_co_overlay_pco(co, tmp_path / "co_pressure.png", electrolyte_column="C_electrolyte_M")
    overlay.plot_rate_overlay_pco(rates, tmp_path / "rate_pressure.png")
    for name in ("alpha_electrolyte.png", "co_pressure.png", "rate_pressure.png"):
        assert (tmp_path / name).is_file()
