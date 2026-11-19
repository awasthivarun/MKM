"""Read-only profiling helpers and stable experimental-point selection."""

from dataclasses import fields
import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

module_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_agpd_frumkin.py"
spec = importlib.util.spec_from_file_location("profile_agpd_frumkin", module_path)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
from mkm.models.agpd_registry import get_agpd_model_definition


def test_profile_point_selection_preserves_full_data_and_stiff_point():
    np.testing.assert_array_equal(profile._select_point_ids(492, 492), np.arange(492))
    ids = profile._select_point_ids(16, 492)
    assert len(ids) == 16
    assert len(np.unique(ids)) == 16
    assert 115 in ids
    np.testing.assert_array_equal(profile._select_point_ids(1, 492), [115])
    np.testing.assert_array_equal(profile._select_point_ids(10, 8), np.arange(8))


def test_profile_reads_exact_physical_parameters(monkeypatch, tmp_path):
    definition = get_agpd_model_definition("c_ER")
    names = [f.name for f in fields(definition.parameter_class)]
    dataset = xr.Dataset({name: (("chain", "draw"), np.array([[i + 0.125]]))
                          for i, name in enumerate(names)})
    calls = []

    def open_dataset(path, **kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(profile.xr, "open_dataset", open_dataset)
    values = profile.saved_parameters(tmp_path / "anything.nc", "c_ER", 0, 0, "posterior")
    assert values == {name: i + 0.125 for i, name in enumerate(names)}
    assert calls == [{"group": "posterior", "engine": "h5netcdf"}]


def test_profile_coverage_summary_reports_failures(capsys):
    result = profile._summarize_coverages([
        np.array([-1.2, np.nan]), np.array([38.0, 384.0]), np.array([1e-12, 0.01]),
        np.array([1e-10, 0.2]), np.array([0.8, 0.2]),
    ], "test")
    output = capsys.readouterr().out
    assert "finite_rates=1/2" in output
    assert "iterations > 96: 1/2" in output
    assert "[1]" in output
    assert result["iterations"].shape == (2,)
