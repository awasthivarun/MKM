from pathlib import Path

import pytest

from mkm.project_paths import ProjectPaths


def test_discover_finds_repository_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"test\"\nversion = \"0.0.0\"\n")
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "run.py"
    script.touch()

    assert ProjectPaths.discover(script).root == tmp_path


def test_agpd_paths_are_centralized_under_root(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    assert paths.agpd_analysis_dir == tmp_path / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
    assert paths.agpd_selected_path.name == "AgPd_COOx_basic_selected.parquet"
    assert paths.agpd_model_config_path == tmp_path / "config" / "models" / "agpd_basic.yaml"
    assert paths.agpd_preprocessing_config_path == tmp_path / "config" / "preprocessing" / "agpd_basic.yaml"
    assert paths.agpd_posterior_root == tmp_path / "results" / "AgPd_COOx_basic" / "posterior"


def test_posterior_output_path_always_uses_canonical_layout(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    result = paths.agpd_posterior_output_dir("Ag10Pd90", "BF_LH", "setup_intercept")

    assert result == (
        tmp_path / "results" / "AgPd_COOx_basic" / "posterior"
        / "Ag10Pd90" / "setup_intercept" / "BF_LH"
    )


def test_iid_reader_supports_legacy_layout(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    legacy = tmp_path / "results" / "AgPd_COOx_basic" / "posterior" / "Ag10Pd90" / "BF"
    legacy.mkdir(parents=True)
    (legacy / "posterior.nc").touch()

    assert paths.agpd_posterior_dir("Ag10Pd90", "BF", "iid") == legacy


def test_missing_posterior_is_reported(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    with pytest.raises(FileNotFoundError, match="Posterior not found"):
        paths.agpd_posterior_dir("Ag10Pd90", "BF", "setup_intercept")


def test_invalid_likelihood_is_rejected(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    with pytest.raises(ValueError, match="Unsupported likelihood"):
        paths.agpd_posterior_output_dir("Ag10Pd90", "BF", "unknown")
