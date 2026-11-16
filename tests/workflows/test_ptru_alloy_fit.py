"""Individual and pooled PtRu fitting contracts without sampling."""

from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from mkm.mechanisms.ptru import PTRU_PARAMETER_CLASSES
from mkm.models.ptru import ptru_all_parameter_specs, ptru_parameter_specs
from mkm.project_paths import ProjectPaths
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE
from mkm.workflows.ptru_fit import build_ptru_fit, load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import build_ptru_run_metadata, validate_ptru_run_metadata


def selected_data(medium, materials):
    records = []
    for material in materials:
        for c in (0.25, 0.5):
            for pco in (0.01, 0.1):
                for index, E in enumerate((0.02, 0.05)):
                    records.append({
                        "material": material, "medium": medium, "C_electrolyte_M": c,
                        "CO_mole_fraction": pco, "E_V_SHE": E, "replicate": "A",
                        "rate_s_inv": 0.01, "ln_rate": np.log(0.01), "analysis_grid_index": index,
                    })
    return pd.DataFrame(records)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model", list(PTRU_PARAMETER_CLASSES))
@pytest.mark.parametrize("material", ["Pt66Ru33", "Pt50Ru50"])
def test_all_individual_alloy_priors(medium, model, material):
    config, priors = load_ptru_config(ProjectPaths.discover(__file__))
    spec = resolve_ptru_fit_specification(config, medium, model, material=material)
    expected = {field.name for field in fields(PTRU_PARAMETER_CLASSES[model])}
    actual = ptru_parameter_specs(priors, medium, model, material)
    assert spec.fit_scope == "individual" and spec.material == material
    assert set(actual) == expected
    assert "q" not in actual and "beta_2_BF" not in actual


@pytest.mark.parametrize("parameterization", ["shared", "independent", "linear_xRu"])
@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_pooled_parameter_profiles_are_correct(parameterization, medium):
    config, priors = load_ptru_config(ProjectPaths.discover(__file__))
    spec = resolve_ptru_fit_specification(
        config, medium, "CO_BF_ER_LH", all_materials=True, parameterization=parameterization,
    )
    parameters = ptru_all_parameter_specs(priors, config, spec)
    assert "q" not in parameters and "beta_2_BF" not in parameters
    if parameterization == "independent":
        assert "Gact2_BF_0_Pt100" not in parameters
        assert "deltaG5_0_Pt100" not in parameters
        assert "Gact2_LH_0_Pt100" in parameters
        assert "Gact2_BF_0_Pt66Ru33" in parameters
    elif parameterization == "linear_xRu":
        assert "deltaG5_0_xRu_slope" in parameters
        assert "beta_2_ER_xRu_slope" in parameters
    else:
        assert "deltaG5_0" in parameters
        assert all(not name.endswith("_xRu_slope") for name in parameters)


def test_pooled_model_and_posterior_provenance(monkeypatch):
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(config, "basic", "CO_BF_ER_LH", all_materials=True)
    monkeypatch.setattr(pd, "read_parquet", lambda _: selected_data("basic", spec.materials))
    fit = build_ptru_fit(paths, spec)
    assert fit.inputs.materials == spec.materials
    assert len(fit.model_data.observations) == 24
    assert fit.output_dir.parts[-4:] == ("all_materials", "shared", "shared", "CO_BF_ER_LH")
    with TemporaryDirectory(dir=paths.root) as temporary:
        path = Path(temporary) / "synthetic.parquet"
        path.write_bytes(b"synthetic")
        fit = replace(fit, data_path=path)
        metadata = build_ptru_run_metadata(paths, fit, sampler={})
        metadata["status"] = RUN_STATUS_COMPLETE
        validate_ptru_run_metadata(metadata, paths, fit)
        metadata["model_specification"]["surface_composition"]["Pt100"]["Ru_fraction"] = 0.1
        with pytest.raises(ValueError, match="model_specification"):
            validate_ptru_run_metadata(metadata, paths, fit)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("material", ["Pt66Ru33", "Pt50Ru50"])
def test_individual_fit_writes_same_path(monkeypatch, medium, material):
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(config, medium, "CO_BF_ER_LH", material=material)
    monkeypatch.setattr(pd, "read_parquet", lambda _: selected_data(medium, [material]))
    fit = build_ptru_fit(paths, spec)
    assert fit.output_dir.parts[-3:] == ("individual", material, "CO_BF_ER_LH")
    assert "Gact2_BF_0" in fit.parameter_specs
    assert "q" not in fit.parameter_specs


@pytest.mark.parametrize("parameterization", ["shared", "independent", "linear_xRu"])
def test_pooled_models_can_be_built_from_selected_data(monkeypatch, parameterization):
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, "acidic", "CO_BF_ER_LH", all_materials=True, parameterization=parameterization,
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: selected_data("acidic", spec.materials))
    fit = build_ptru_fit(paths, spec)
    names = {variable.name for variable in fit.built_model.model.free_RVs}
    assert {"sigma_rate_abs", "sigma_rate_rel"} <= names
    assert set(fit.parameter_specs) == names


def test_acid_and_base_cannot_be_combined(monkeypatch):
    paths = ProjectPaths.discover(__file__)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(config, "acidic", "CO_ER_LH", all_materials=True)
    wrong = selected_data("acidic", spec.materials)
    wrong.loc[wrong.index[0], "medium"] = "basic"
    monkeypatch.setattr(pd, "read_parquet", lambda _: wrong)
    with pytest.raises(ValueError, match="wrong electrolyte medium"):
        build_ptru_fit(paths, spec)


def test_bf_only_with_pt100_is_rejected_but_alloy_only_is_allowed():
    config, _ = load_ptru_config(ProjectPaths.discover(__file__))
    with pytest.raises(ValueError, match="BF-only"):
        resolve_ptru_fit_specification(config, "basic", "CO_BF", all_materials=True)
    spec = resolve_ptru_fit_specification(
        config, "basic", "CO_BF", all_materials=True, excluded_materials=("Pt100",),
    )
    assert spec.materials == ("Pt50Ru50", "Pt66Ru33")


def test_independent_requires_shared_error_structure():
    config, _ = load_ptru_config(ProjectPaths.discover(__file__))
    with pytest.raises(ValueError, match="requires shared"):
        resolve_ptru_fit_specification(
            config, "basic", "CO_ER_LH", all_materials=True,
            parameterization="independent", error_structure="material",
        )
