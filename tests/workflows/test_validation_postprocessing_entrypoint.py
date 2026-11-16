"""Exercise the formerly broken validation postprocessing script import/call path."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agpd_validation_postprocessing_uses_shared_workflow(tmp_path, monkeypatch):
    script = _load_script("postprocess_agpd_validation.py")
    args = SimpleNamespace(
        model="CO_BF_ER_LH", scheme="loco", holdout_material="Ag10Pd90", koh_M=0.5,
        co_mole_fraction=0.1, parameterization="shared", error_structure="shared",
        prior_material="Ag10Pd90", random_seed=42, second_order_step_v=0.03,
        drc_step_eV=1e-4, plot_level="none",
    )
    paths = SimpleNamespace(fit_tables_dir=lambda out: out / "tables")
    spec = SimpleNamespace(is_all_materials=True, parameterization="shared")
    original_run = SimpleNamespace(output_dir=tmp_path, specification=spec)
    # Use the original dataclass with minimal fields because main must call replace().
    from mkm.workflows.agpd_posterior import AgPdPosteriorRun
    run = AgPdPosteriorRun(spec, {}, tmp_path, None, None, None, None, {}, ())
    called = {}
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(script.ProjectPaths, "discover", lambda *_: paths)
    monkeypatch.setattr(script, "load_agpd_model_config", lambda *_: {})
    monkeypatch.setattr(script, "resolve_agpd_fit_specification", lambda *_args, **_kw: spec)
    monkeypatch.setattr(script, "_load_validation_run", lambda *_: run)
    monkeypatch.setattr(script, "postprocess_agpd_run", lambda current, *_a, **kw: (
        called.update(output=current.output_dir, options=kw) or "complete"
    ))
    monkeypatch.setattr(script, "postprocess_agpd_drc", lambda *_a, **_kw: SimpleNamespace(
        checks=pd.DataFrame([{"max_abs_sum_error": 0.0}]),
    ))
    monkeypatch.setattr(pd, "read_csv", lambda *_: pd.DataFrame([{"status": "complete"}]))
    script.main()
    assert called["output"] == tmp_path / "postprocessing"
    assert called["options"]["random_seed"] == 42
    assert called["options"]["plot_level"] == "none"
    assert called["options"]["skip_observables"] is True


def test_agpd_validation_postprocessing_rejects_wrong_training_hash(tmp_path, monkeypatch):
    script = _load_script("postprocess_agpd_validation.py")
    args = SimpleNamespace(
        scheme="lomo", model="CO_BF_ER_LH", holdout_material="Ag10Pd90",
        koh_M=None, co_mole_fraction=None, parameterization="shared", error_structure="shared",
    )
    frame = pd.DataFrame({"material": ["Ag10Pd90", "Ag25Pd75"]})
    model_data = SimpleNamespace(observations=pd.DataFrame({"observation_id": [0]}))
    fit = SimpleNamespace(inputs=SimpleNamespace(materials=("Ag25Pd75",)))
    monkeypatch.setattr(script, "available_agpd_materials", lambda *_: ("Ag10Pd90", "Ag25Pd75"))
    monkeypatch.setattr(script, "load_agpd_selected_materials", lambda *_: frame)
    monkeypatch.setattr(script, "split_agpd_lomo", lambda *_: (frame, frame))
    monkeypatch.setattr(script, "build_model_data", lambda *_a, **_kw: model_data)
    monkeypatch.setattr(script, "build_agpd_fit_model", lambda *_: fit)
    monkeypatch.setattr(script, "read_run_metadata", lambda *_: {"validation_scheme": "lomo"})
    monkeypatch.setattr(script, "validate_agpd_run_metadata", lambda *_a, **_kw: None)
    monkeypatch.setattr(script, "sha256_file", lambda *_: "source-hash")
    paths = SimpleNamespace(
        root=tmp_path, agpd_validation_output_dir=lambda **_: tmp_path,
    )
    with pytest.raises(ValueError, match="training_observations_sha256"):
        script._load_validation_run(paths, {}, SimpleNamespace(error_structure="shared"), args)


def test_pt100_heldout_figures_cover_all_existing_diagnostics(tmp_path, monkeypatch):
    script = _load_script("validate_pt100_posterior.py")
    calls = []

    def record(name):
        def plot(_frame, output_path, **kwargs):
            calls.append((name, output_path.name, kwargs))
        return plot

    monkeypatch.setattr(script, "plot_observation_grid", record("observation"))
    monkeypatch.setattr(script, "plot_heldout_pit_conditions", record("pit"))
    args = SimpleNamespace(medium="acidic", electrolyte_M=0.5, co_mole_fraction=0.1)
    predicted = SimpleNamespace(pointwise=pd.DataFrame({"model_residual": [0.1]}))
    script.plot_pt100_heldout_products(predicted, tmp_path, args)
    assert {name for _, name, _ in calls} == {
        "heldout_rates_model.png", "heldout_rates_predictive.png",
        "heldout_residuals.png", "heldout_pit_conditions.png",
    }
    assert all(options["electrolyte_label"] == "HClO4" for _, _, options in calls)
