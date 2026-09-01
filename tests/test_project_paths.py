from pathlib import Path

import pytest

from mkm.project_paths import ProjectPaths


def test_discover_finds_repository_root(tmp_path):
    root = tmp_path / "repo"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")

    assert ProjectPaths.discover(nested).root == root


def test_individual_posterior_path_is_compact_and_unambiguous(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    result = paths.agpd_posterior_output_dir(
        fit_scope="individual",
        material="Ag10Pd90",
        model_name="BF_LH",
        parameterization=None,
        error_structure="material",
    )
    assert result == (
        tmp_path
        / "results"
        / "AgPd_COOx_basic"
        / "posterior"
        / "individual"
        / "Ag10Pd90"
        / "BF_LH"
    )


def test_all_material_path_labels_parameterization_and_error_structure(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    result = paths.agpd_posterior_output_dir(
        fit_scope="all_materials",
        model_name="CO_BF_ER_LH",
        parameterization="linear_beta_er",
        error_structure="shared",
    )
    assert result == (
        tmp_path
        / "results"
        / "AgPd_COOx_basic"
        / "posterior"
        / "all_materials"
        / "linear_beta_er"
        / "shared"
        / "CO_BF_ER_LH"
    )


def test_posterior_reader_requires_single_posterior_file(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    kwargs = {
        "fit_scope": "individual",
        "material": "Ag10Pd90",
        "model_name": "BF",
        "parameterization": None,
        "error_structure": "material",
    }
    with pytest.raises(FileNotFoundError, match="posterior.nc"):
        paths.agpd_posterior_dir(**kwargs)

    directory = paths.agpd_posterior_output_dir(**kwargs)
    directory.mkdir(parents=True)
    (directory / "posterior.nc").touch()
    assert paths.agpd_posterior_dir(**kwargs) == directory


def test_validation_paths_distinguish_loco_and_lomo(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    lomo = paths.agpd_validation_output_dir(
        scheme="lomo",
        model_name="CO_BF_ER_LH",
        parameterization="shared",
        error_structure="shared",
        material="Pd100",
    )
    loco = paths.agpd_validation_output_dir(
        scheme="loco",
        model_name="CO_BF_ER_LH",
        parameterization="linear_xAg",
        error_structure="material",
        material="Ag50Pd50",
        koh_M=0.5,
        co_mole_fraction=0.1,
    )

    assert lomo.name == "Pd100"
    assert loco.name == "KOH_0.5_CO_0.1"


def test_invalid_scope_or_error_structure_is_rejected(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    with pytest.raises(ValueError, match="Unknown fit scope"):
        paths.agpd_posterior_output_dir(
            fit_scope="unknown",
            model_name="BF",
        )
    with pytest.raises(ValueError, match="Unsupported error structure"):
        paths.agpd_posterior_output_dir(
            fit_scope="all_materials",
            model_name="BF_LH",
            parameterization="shared",
            error_structure="unknown",
        )
