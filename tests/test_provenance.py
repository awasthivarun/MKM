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
