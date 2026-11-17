"""Unified PtRu provenance and posterior reconstruction contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from mkm.project_paths import ProjectPaths
from mkm.provenance import PTRU_MODEL_SOURCE_FILES
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE, RUN_STATUS_SAMPLED
from mkm.workflows.ptru_fit import PtRuFitSpecification
from mkm.workflows import ptru_posterior


def fake_fit(paths, pooled=False):
    data_path = paths.root / "selected.parquet"
    model_path = paths.root / "ptru.yaml"
    prior_path = paths.root / "priors.yaml"
    data_path.write_bytes(b"synthetic selected data")
    model_path.write_text("model: test\n", encoding="utf-8")
    prior_path.write_text("priors: test\n", encoding="utf-8")
    materials = ("Pt100", "Pt50Ru50", "Pt66Ru33") if pooled else ("Pt100",)
    spec = PtRuFitSpecification(
        "acidic", "CO_ER_LH", materials, "all_materials" if pooled else "individual",
        None if pooled else "Pt100", "shared" if pooled else None, "shared",
    )
    return SimpleNamespace(
        specification=spec, inputs=SimpleNamespace(materials=materials),
        config={
            "mechanism_assumptions": {"CO_adsorption": "finite_rate_SSA"},
            "surface_composition": {material: {"Ru_fraction": 0 if material == "Pt100" else 1 / 3}
                                    for material in materials},
            "parameterizations": {"shared": {}},
            "temperature_K": 293.15,
            "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
            "electrolyte": {"activity_model": "ideal_molarity", "standard_state_concentration_M": 1.0},
        },
        parameter_specs={"Gact1_0": {"distribution": "normal", "mu": 0.5, "sigma": 0.1}},
        data_path=data_path, model_config_path=model_path, prior_config_path=prior_path,
    )


def test_source_identity_tracks_unified_pt_ru_model_code():
    sources = set(PTRU_MODEL_SOURCE_FILES)
    assert {
        "src/mkm/mechanisms/co_ssa.py", "src/mkm/mechanisms/electrochem.py",
        "src/mkm/mechanisms/ptru.py", "src/mkm/models/ptru.py", "src/mkm/workflows/ptru_fit.py",
        "src/mkm/workflows/likelihood_configuration.py", "src/mkm/inference/model.py",
    } <= sources


@pytest.mark.parametrize("pooled", [False, True])
def test_checkpoint_validates_and_detects_modifications(tmp_path, monkeypatch, pooled):
    paths = ProjectPaths(root=tmp_path)
    fit = fake_fit(paths, pooled)
    monkeypatch.setattr(ptru_posterior, "PTRU_MODEL_SOURCE_FILES", ())
    metadata = ptru_posterior.build_ptru_run_metadata(paths, fit, sampler={})
    metadata["status"] = RUN_STATUS_COMPLETE
    ptru_posterior.validate_ptru_run_metadata(metadata, paths, fit)
    metadata["status"] = RUN_STATUS_SAMPLED
    ptru_posterior.validate_ptru_run_metadata(
        metadata, paths, fit, allowed_statuses=(RUN_STATUS_COMPLETE, RUN_STATUS_SAMPLED),
    )
    metadata["status"] = RUN_STATUS_COMPLETE
    fit.data_path.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="inputs.data_sha256"):
        ptru_posterior.validate_ptru_run_metadata(metadata, paths, fit)


def test_checkpoint_ignores_unrelated_configuration_file_edits(tmp_path, monkeypatch):
    paths = ProjectPaths(root=tmp_path)
    fit = fake_fit(paths)
    monkeypatch.setattr(ptru_posterior, "PTRU_MODEL_SOURCE_FILES", ())
    metadata = ptru_posterior.build_ptru_run_metadata(paths, fit, sampler={})
    metadata["status"] = RUN_STATUS_COMPLETE
    fit.prior_config_path.write_text("priors: unrelated_new_model\n", encoding="utf-8")
    ptru_posterior.validate_ptru_run_metadata(metadata, paths, fit)


def test_checkpoint_detects_changed_resolved_kinetic_config(tmp_path, monkeypatch):
    paths = ProjectPaths(root=tmp_path)
    fit = fake_fit(paths)
    monkeypatch.setattr(ptru_posterior, "PTRU_MODEL_SOURCE_FILES", ())
    metadata = ptru_posterior.build_ptru_run_metadata(paths, fit, sampler={})
    metadata["status"] = RUN_STATUS_COMPLETE
    fit.config["temperature_K"] = 310.0
    with pytest.raises(ValueError, match="model_specification"):
        ptru_posterior.validate_ptru_run_metadata(metadata, paths, fit)


def _idata(n_chains, n_draws, stats_draws=None):
    stats_draws = n_draws if stats_draws is None else stats_draws
    posterior = xr.Dataset({"Gact1_0": (("chain", "draw"), np.full((n_chains, n_draws), 0.5))})
    divergent = np.zeros((n_chains, stats_draws), dtype=bool)
    divergent[0, -3] = True
    sample_stats = xr.Dataset({"diverging": (("chain", "draw"), divergent)})
    likelihood = xr.Dataset({
        "rate_observed": (("chain", "draw", "observation"), np.zeros((n_chains, n_draws, 2))),
    })
    return xr.DataTree.from_dict({
        "/posterior": posterior, "/sample_stats": sample_stats, "/log_likelihood": likelihood,
    })


def _mock_loaded_fit(tmp_path, monkeypatch, idata):
    built = SimpleNamespace(
        model=SimpleNamespace(free_RVs=(SimpleNamespace(name="Gact1_0"),)),
        mechanism_result=SimpleNamespace(pointwise={"theta_CO": None}),
    )
    fit = SimpleNamespace(
        output_dir=tmp_path, built_model=built, specification=SimpleNamespace(medium="acidic"),
        model_data=None, inputs=None, parameter_specs={},
    )
    (tmp_path / "posterior.nc").touch()
    monkeypatch.setattr(ptru_posterior, "build_ptru_fit", lambda *_: fit)
    monkeypatch.setattr(ptru_posterior, "read_run_metadata", lambda *_: {})
    monkeypatch.setattr(ptru_posterior, "validate_ptru_run_metadata", lambda *_: None)
    monkeypatch.setattr(ptru_posterior, "load_inference_data", lambda *_: idata)
    return fit


def test_loader_retains_all_draws_and_divergences(tmp_path, monkeypatch):
    idata = _idata(2, 1000)
    _mock_loaded_fit(tmp_path, monkeypatch, idata)

    def deterministics(source, *_args, **_kwargs):
        return source.posterior.to_dataset().assign(
            ln_rate_model=(("chain", "draw"), np.zeros((2, 1000))),
            theta_CO=(("chain", "draw"), np.full((2, 1000), 0.5)),
        )

    monkeypatch.setattr(ptru_posterior, "compute_posterior_deterministics", deterministics)
    run = ptru_posterior.load_ptru_posterior_run(
        SimpleNamespace(), SimpleNamespace(medium="acidic"), reconstruct_pointwise=True, progressbar=False,
    )
    for group in ("posterior", "sample_stats", "log_likelihood"):
        assert getattr(run.inference_data, group).sizes["draw"] == 1000
    assert run.inference_data.sample_stats.to_dataset()["diverging"].to_numpy().sum() == 1
    assert "theta_CO" in run.inference_data.posterior


def test_loader_rejects_mismatched_draw_dimensions(tmp_path, monkeypatch):
    _mock_loaded_fit(tmp_path, monkeypatch, _idata(2, 125, 1000))
    with pytest.raises(ValueError, match="different draw dimensions"):
        ptru_posterior.load_ptru_posterior_run(SimpleNamespace(), SimpleNamespace())
