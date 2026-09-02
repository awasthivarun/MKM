import numpy as np
import pandas as pd
import pymc as pm
import pytest
import yaml

from mkm.model_inputs import ModelInputArrays
from mkm.workflows.agpd_fit import build_fit_mechanism, resolve_agpd_fit_specification
from mkm.workflows.agpd_validation import (
    build_mechanism_prediction_model,
    split_agpd_loco,
    split_agpd_lomo,
    validate_heldout_error_support,
    validate_validation_error_structure,
)


def _config():
    with open("config/models/agpd_basic.yaml", "r") as file:
        return yaml.safe_load(file)


def _selected_frame():
    rows = []
    for material in ("Ag10Pd90", "Ag50Pd50"):
        for koh in (0.25, 0.50):
            for co in (0.01, 0.10):
                for replicate in ("A", "B", "C"):
                    rows.append(
                        {
                            "material": material,
                            "C_KOH_M": koh,
                            "CO_mole_fraction": co,
                            "replicate": replicate,
                        }
                    )
    return pd.DataFrame(rows)


def _pd100_inputs():
    return ModelInputArrays(
        materials=("Pd100",),
        condition_material_index=np.array([0], dtype=np.int64),
        condition_ln_electrolyte_concentration=np.log(np.array([0.5])),
        condition_ln_CO_mole_fraction=np.log(np.array([0.1])),
        model_point_condition_index=np.array([0], dtype=np.int64),
        model_point_E_V_SHE=np.array([0.3]),
        observation_model_point_index=np.array([0], dtype=np.int64),
        observation_rate=np.array([1.0e-3]),
    )


def test_loco_removes_all_replicates_for_one_condition():
    train, heldout = split_agpd_loco(_selected_frame(), "Ag10Pd90", 0.25, 0.01)

    assert set(heldout["replicate"]) == {"A", "B", "C"}
    assert len(heldout) == 3
    assert not (
        train["material"].eq("Ag10Pd90")
        & train["C_KOH_M"].eq(0.25)
        & train["CO_mole_fraction"].eq(0.01)
    ).any()


def test_lomo_removes_entire_material():
    train, heldout = split_agpd_lomo(_selected_frame(), "Ag10Pd90")

    assert set(heldout["material"]) == {"Ag10Pd90"}
    assert "Ag10Pd90" not in set(train["material"])
    assert set(train["material"]) == {"Ag50Pd50"}


def test_lomo_requires_shared_error_without_hierarchy():
    validate_validation_error_structure("lomo", "shared")
    validate_validation_error_structure("loco", "material")

    with pytest.raises(ValueError, match="no fitted sigma_rate_abs"):
        validate_validation_error_structure("lomo", "material")


def test_material_error_requires_every_heldout_material_in_training():
    validate_heldout_error_support(
        "material",
        training_materials=("Ag10Pd90", "Ag50Pd50"),
        heldout_materials=("Ag10Pd90",),
    )

    with pytest.raises(ValueError, match="absent from training"):
        validate_heldout_error_support(
            "material",
            training_materials=("Ag50Pd50",),
            heldout_materials=("Ag10Pd90",),
        )


def test_lomo_prediction_model_can_evaluate_pd100_with_full_co_model():
    config = _config()
    specification = resolve_agpd_fit_specification(
        config,
        model_name="CO_BF_ER_LH",
        all_materials=True,
        parameterization="linear_xAg",
        error_structure="shared",
        prior_material="Ag10Pd90",
    )
    inputs = _pd100_inputs()
    mechanism = build_fit_mechanism(
        specification,
        inputs,
        config,
        prediction_only=True,
    )
    prediction_model = build_mechanism_prediction_model(inputs, mechanism)

    assert isinstance(prediction_model, pm.Model)
    assert "ln_rate_model" in prediction_model.named_vars
    free_names = {variable.name for variable in prediction_model.free_RVs}
    assert "Gact2_BF_0" in free_names
    assert "Gact2_BF_0_xAg_slope" in free_names
