import pytest

from mkm.provenance import build_fit_metadata, read_run_metadata, sha256_file, write_run_metadata


def test_sha256_file_is_content_stable(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same content")
    second.write_text("same content")

    assert sha256_file(first) == sha256_file(second)


def test_rate_normal_run_metadata_round_trips_yaml(tmp_path):
    path = tmp_path / "run_metadata.yaml"
    metadata = {
        "status": "complete",
        "fit_scope": "all_materials",
        "materials": ["Ag10Pd90", "Pd100"],
        "model": "CO_BF_ER_LH",
        "likelihood": "rate_normal",
        "error_structure": "material",
        "parameterization": "linear_beta_er",
        "parameterization_specification": {
            "name": "linear_beta_er",
            "x_reference": 0.5,
            "slopes": {"beta_2_ER": {"distribution": "normal", "mu": 0.0, "sigma": 0.25}},
        },
        "stored_posterior_variables": [
            "beta_2_ER",
            "beta_2_ER_xAg_slope",
            "sigma_rate_abs",
            "sigma_rate_rel",
        ],
    }

    write_run_metadata(metadata, path)
    assert read_run_metadata(path) == metadata


def test_all_material_fit_metadata_requires_resolved_parameterization_contents(tmp_path, monkeypatch):
    data_path = tmp_path / "data.parquet"
    config_path = tmp_path / "config.yaml"
    data_path.write_text("data")
    config_path.write_text("config")
    monkeypatch.setattr("mkm.provenance.git_commit", lambda root: "abc123")
    monkeypatch.setattr("mkm.provenance.package_versions", lambda: {})

    with pytest.raises(ValueError, match="resolved parameterization"):
        build_fit_metadata(
            root=tmp_path,
            fit_scope="all_materials",
            materials=("Ag10Pd90", "Pd100"),
            model_name="CO_BF_ER_LH",
            error_structure="shared",
            data_path=data_path,
            model_config_path=config_path,
            sampler={},
            parameterization="linear_xAg",
            parameterization_specification=None,
            prior_material="Ag10Pd90",
        )

    specification = {
        "name": "linear_xAg",
        "x_reference": 0.5,
        "slopes": {"deltaG1_0": {"distribution": "normal", "mu": 0.0, "sigma": 0.1}},
    }
    metadata = build_fit_metadata(
        root=tmp_path,
        fit_scope="all_materials",
        materials=("Ag10Pd90", "Pd100"),
        model_name="CO_BF_ER_LH",
        error_structure="shared",
        data_path=data_path,
        model_config_path=config_path,
        sampler={},
        parameterization="linear_xAg",
        parameterization_specification=specification,
        prior_material="Ag10Pd90",
    )
    assert metadata["parameterization_specification"] == specification


def test_sha256_mapping_is_key_order_stable():
    from mkm.provenance import sha256_mapping

    assert sha256_mapping({"b": 2, "a": 1}) == sha256_mapping({"a": 1, "b": 2})


def test_fit_metadata_hashes_complete_config_bundle(tmp_path, monkeypatch):
    from mkm.provenance import sha256_mapping

    data_path = tmp_path / "data.parquet"
    base = tmp_path / "model.yaml"
    priors = tmp_path / "priors.yaml"
    data_path.write_text("data")
    base.write_text("base")
    priors.write_text("priors")
    monkeypatch.setattr("mkm.provenance.git_commit", lambda root: "abc123")
    monkeypatch.setattr("mkm.provenance.package_versions", lambda: {})

    metadata = build_fit_metadata(
        root=tmp_path,
        fit_scope="individual",
        materials=("Ag10Pd90",),
        model_name="CO_ER",
        error_structure="material",
        data_path=data_path,
        model_config_path=base,
        model_config_paths=(base, priors),
        sampler={},
        prior_material="Ag10Pd90",
    )
    expected = sha256_mapping({"model.yaml": sha256_file(base), "priors.yaml": sha256_file(priors)})
    assert metadata["inputs"]["model_config_sha256"] == expected
    assert metadata["inputs"]["model_config_paths"] == ["model.yaml", "priors.yaml"]


def test_atomic_metadata_write_preserves_previous_on_serialization_error(tmp_path):
    import yaml
    path = tmp_path / "metadata.yaml"
    write_run_metadata({"status": "complete"}, path)
    class Unserializable:
        pass
    with pytest.raises(yaml.representer.RepresenterError):
        write_run_metadata({"x": Unserializable()}, path)
    assert read_run_metadata(path) == {"status": "complete"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_preprocessing_manifest_hashes_actual_outputs(tmp_path):
    from mkm.provenance import write_preprocessing_manifest
    source = tmp_path / "source.py"
    source.write_text("def preprocess(): pass")
    raw = tmp_path / "raw.xlsx"
    raw.write_bytes(b"raw workbook")
    config = tmp_path / "config.yaml"
    config.write_text("method: test")
    output = tmp_path / "selected.parquet"
    output.write_bytes(b"row contents")
    manifest = write_preprocessing_manifest(
        tmp_path / "manifest.yaml", root=tmp_path, raw_paths=[raw], config_paths=[config],
        source_files=["source.py"], products={"selected": (output, 3)},
    )
    assert manifest["outputs"]["selected"]["rows"] == 3
    assert manifest["outputs"]["selected"]["sha256"] == sha256_file(output)
    assert manifest["raw_inputs"][0]["sha256"] == sha256_file(raw)
    assert read_run_metadata(tmp_path / "manifest.yaml") == manifest


def test_tracked_source_and_configuration_mutations_change_run_identity(tmp_path, monkeypatch):
    from mkm.provenance import source_file_hashes, validate_fit_input_identity

    source = tmp_path / "mechanism.py"
    data_path = tmp_path / "data.parquet"
    config_path = tmp_path / "model.yaml"
    source.write_text("rate = 1")
    data_path.write_text("observations")
    config_path.write_text("parameter: 1")
    monkeypatch.setattr("mkm.provenance.git_commit", lambda root: None)
    monkeypatch.setattr("mkm.provenance.package_versions", lambda: {})
    metadata = build_fit_metadata(
        root=tmp_path, fit_scope="individual", materials=("Pt100",), model_name="CO_ER",
        error_structure="shared", data_path=data_path, model_config_path=config_path, sampler={},
    )
    before = source_file_hashes(tmp_path, ("mechanism.py",))
    assert not validate_fit_input_identity(metadata, tmp_path, data_path, config_path, (config_path,))

    source.write_text("rate = 2")
    assert source_file_hashes(tmp_path, ("mechanism.py",)) != before
    config_path.write_text("parameter: 2")
    # Schema 2 retains the complete YAML digest for audit but validates resolved
    # kinetic and prior specifications at the model level instead of all YAML.
    assert not validate_fit_input_identity(metadata, tmp_path, data_path, config_path, (config_path,))
    # Older schema 1 checkpoints still require the exact configuration source.
    legacy_metadata = dict(metadata)
    legacy_metadata.pop("provenance_schema")
    assert "inputs.model_config_sha256" in validate_fit_input_identity(
        legacy_metadata, tmp_path, data_path, config_path, (config_path,),
    )
    data_path.write_text("different observations")
    assert "inputs.data_sha256" in validate_fit_input_identity(
        metadata, tmp_path, data_path, config_path, (config_path,),
    )
