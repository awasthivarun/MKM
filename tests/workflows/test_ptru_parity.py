"""Contract tests for PtRu per-run comparison and all-material holdout workflows."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_validation import ptru_lomo_output_dir, split_ptru_lomo


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("test_script_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ptru_comparison_supports_candidate_specific_parameterizations_and_errors():
    parse = load_script("compare_ptru_models").parse_candidates
    candidates = parse((
        "a,CO_ER_LH,shared,shared", "b,CO_ER_LH,linear_xRu,material",
        "c,CO_ER_LH,independent,shared",
    ), all_materials=True)
    assert candidates[1] == ("b", "CO_ER_LH", "linear_xRu", "material")
    assert candidates[2][-2:] == ("independent", "shared")
    with pytest.raises(ValueError, match="unique run labels"):
        parse(("a,CO_ER", "a,CO_LH"))
    with pytest.raises(ValueError, match="LABEL,MODEL"):
        parse(("a,CO_ER,linear_xRu,material", "b,CO_LH"), all_materials=False)


def test_ptru_lomo_holds_out_entire_material_without_mixing_media():
    frame = pd.DataFrame({
        "material": np.repeat(("Pt100", "Pt66Ru33", "Pt50Ru50"), 4),
        "C_electrolyte_M": [0.1, 0.1, 0.5, 0.5] * 3,
        "CO_mole_fraction": [0.01, 0.01, 0.10, 0.10] * 3,
        "E_V_SHE": [0.0, 0.1, 0.0, 0.1] * 3,
    })
    train, heldout = split_ptru_lomo(frame, "Pt66Ru33")
    assert set(train.material) == {"Pt100", "Pt50Ru50"}
    assert set(heldout.material) == {"Pt66Ru33"}
    assert len(train) == 8 and len(heldout) == 4
    with pytest.raises(ValueError, match="absent"):
        split_ptru_lomo(frame, "Pt75Ru25")
    with pytest.raises(ValueError, match="two training materials"):
        split_ptru_lomo(frame.loc[frame.material != "Pt100"], "Pt66Ru33")


def test_ptru_lomo_training_specification_and_output_location():
    paths = ProjectPaths(root=ROOT)
    config, _ = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, "acidic", "CO_ER_LH", all_materials=True,
        parameterization="linear_xRu", excluded_materials=("Pt66Ru33",),
    )
    assert spec.error_structure == "shared"
    assert spec.materials == ("Pt100", "Pt50Ru50")
    directory = ptru_lomo_output_dir(paths, "acidic", "CO_ER_LH", "Pt66Ru33", "linear_xRu", "shared")
    assert directory.parts[-5:] == ("lomo", "Pt66Ru33", "CO_ER_LH", "linear_xRu", "shared")


def test_ptru_lomo_cli_rejects_independent_error_parameters_and_uses_target_accept_090(monkeypatch):
    import sys
    script = load_script("validate_ptru_lomo")
    monkeypatch.setattr(sys, "argv", [
        "validate_ptru_lomo.py", "CO_ER_LH", "--medium", "basic", "--holdout-material", "Pt50Ru50",
    ])
    args = script.parse_args()
    assert args.parameterization == "shared" and args.error_structure == "shared"
    assert args.target_accept == 0.90
    monkeypatch.setattr(sys, "argv", [
        "validate_ptru_lomo.py", "CO_ER_LH", "--medium", "basic", "--holdout-material", "Pt50Ru50",
        "--parameterization", "independent",
    ])
    with pytest.raises(SystemExit):
        script.parse_args()
