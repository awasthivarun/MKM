"""DRC skip provenance must never silently claim that retained results were recomputed."""

import json

from mkm.postprocessing.drc_status import record_drc_status


def test_skipped_drc_without_previous_artifacts(tmp_path):
    status = record_drc_status(tmp_path / "drc", recomputed=False)
    assert status["status"] == "skipped_no_existing_products"
    assert status["last_computed_at_utc"] is None


def test_skipped_drc_preserves_previous_products_and_timestamp(tmp_path):
    drc_dir = tmp_path / "drc"
    drc_dir.mkdir()
    output = drc_dir / "drc_transition_state_checks.csv"
    output.write_text("check,value\nsum,1\n", encoding="utf-8")

    first = record_drc_status(drc_dir, recomputed=True, options={"step_eV": 1e-4})
    skipped = record_drc_status(drc_dir, recomputed=False)
    assert skipped["status"] == "retained_from_previous_invocation"
    assert skipped["last_computed_at_utc"] == first["last_computed_at_utc"]
    assert skipped["last_computation_options"] == {"step_eV": 1e-4}
    assert output.read_text(encoding="utf-8") == "check,value\nsum,1\n"
    assert json.loads((drc_dir / "drc_status.json").read_text(encoding="utf-8")) == skipped


def test_existing_pre_status_drc_has_unknown_generation_time(tmp_path):
    drc_dir = tmp_path / "drc"
    drc_dir.mkdir()
    (drc_dir / "drc_transition_state.parquet").write_bytes(b"previous")
    status = record_drc_status(drc_dir, recomputed=False)
    assert status["status"] == "retained_from_previous_invocation"
    assert status["last_computed_at_utc"] is None
    assert status["products_present"] == ["drc_transition_state.parquet"]


def test_plot_cleanup_retains_only_existing_drc_figures(tmp_path):
    from mkm.postprocessing.drc_status import reset_figures_preserving_drc

    figures = tmp_path / "figures"
    material = figures / "Pt66Ru33"
    material.mkdir(parents=True)
    drc_figure = material / "drc_transition_states.png"
    drc_figure.write_bytes(b"old-drc-figure")
    (material / "old_rate_plot.png").write_bytes(b"old-rate-figure")
    reset_figures_preserving_drc(figures)
    assert drc_figure.read_bytes() == b"old-drc-figure"
    assert not (material / "old_rate_plot.png").exists()
    status = record_drc_status(tmp_path / "drc", recomputed=False, figures_dir=figures)
    assert status["status"] == "retained_from_previous_invocation"
    assert status["figures_present"] == ["Pt66Ru33/drc_transition_states.png"]
