from mkm.provenance import read_run_metadata, sha256_file, write_run_metadata


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
        "stored_posterior_variables": [
            "beta_2_ER",
            "beta_2_ER_xAg_slope",
            "sigma_rate_abs",
            "sigma_rate_rel",
        ],
    }

    write_run_metadata(metadata, path)
    assert read_run_metadata(path) == metadata
