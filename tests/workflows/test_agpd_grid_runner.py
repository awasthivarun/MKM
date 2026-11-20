"""Lightweight subprocess and retry-contract checks for the AgPd fit-grid runner."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_agpd_individual_grid.py"
    spec = importlib.util.spec_from_file_location("agpd_grid_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_commands_preserve_fit_scope_and_retry_arguments(runner, tmp_path):
    individual = runner._individual_job("Ag50Pd50", "CO_ER")
    first = runner._fit_command(tmp_path, individual, 0.90, overwrite=False)
    retry = runner._fit_command(tmp_path, individual, 0.95, overwrite=True)
    assert first[-2:] == ["--target-accept", "0.90"]
    assert retry[-3:] == ["--target-accept", "0.95", "--overwrite"]
    assert first[first.index("--material") + 1] == "Ag50Pd50"
    assert "--all-materials" not in first

    pooled = runner._all_material_job("Full", "CO_BF_ER_LH")
    command = runner._fit_command(tmp_path, pooled, 0.99, overwrite=False)
    assert "--all-materials" in command
    assert command[command.index("--parameterization") + 1] == "linear_xAg"
    assert command[command.index("--prior-material") + 1] == "Ag10Pd90"


def test_divergence_retry_and_terminal_diagnostics_are_recorded(runner, monkeypatch, tmp_path):
    commands, snapshots = [], []

    def execute(command):
        commands.append(list(command))
        return 0, 0.01

    def diagnostics(record, _run_dir):
        record["n_divergent"] = 1 if record["attempt"] == 1 else 0
        record["diagnostics_error"] = ""
        record["sampling_status"] = ""

    monkeypatch.setattr(runner, "_run", execute)
    monkeypatch.setattr(runner, "_populate_sampling_diagnostics", diagnostics)
    monkeypatch.setattr(runner, "_sampling_status", lambda record: ("PASS", ""))
    monkeypatch.setattr(runner, "_populate_postprocessing_status", lambda *_: None)
    monkeypatch.setattr(runner, "_write_diagnostics", lambda path, records: snapshots.append([dict(r) for r in records]))
    records = []
    job = runner._individual_job("Ag50Pd50", "CO_ER")
    result = runner._run_one_fit(tmp_path, SimpleNamespace(overwrite=False), "now", tmp_path, records, job)

    assert [record["sampling_status"] for record in records] == ["RETRY_DIVERGENCES", "PASS"]
    assert records[0]["terminal_attempt"] is False
    assert result["terminal_attempt"] is True
    assert len(commands) == 4  # Two fits, postprocessing, then DRC.
    assert [commands[i][commands[i].index("--target-accept") + 1] for i in (0, 1)] == ["0.90", "0.95"]
    assert "--overwrite" not in commands[0] and "--overwrite" in commands[1]
    assert len(snapshots) >= 2


def test_failed_fit_does_not_run_postprocessing(runner, monkeypatch, tmp_path):
    commands = []

    def execute(command):
        commands.append(command)
        return 7, 0.01

    monkeypatch.setattr(runner, "_run", execute)
    monkeypatch.setattr(runner, "_write_diagnostics", lambda *_: None)
    records = []
    job = runner._individual_job("Ag50Pd50", "CO_ER")
    result = runner._run_one_fit(tmp_path, SimpleNamespace(overwrite=False), "now", tmp_path, records, job)
    assert len(commands) == 1
    assert result["fit_returncode"] == 7
    assert result["sampling_status"] == "FIT_ERROR"
    assert result["comparison_excluded_reason"] == "fit command failed"
