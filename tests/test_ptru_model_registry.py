"""Unified PtRu registry and pure-Pt pathway constraints."""

from dataclasses import fields

import pytest

from mkm.mechanisms.ptru import PURE_PT_MODELS, PTRU_PARAMETER_CLASSES
from mkm.models.ptru import available_ptru_models, ptru_parameter_names


def test_ptru_registry_contains_pure_and_alloy_models():
    assert available_ptru_models() == tuple(PTRU_PARAMETER_CLASSES)
    assert PURE_PT_MODELS == ("CO_LH", "CO_ER", "CO_ER_LH")
    for name in available_ptru_models():
        expected = {field.name for field in fields(PTRU_PARAMETER_CLASSES[name])}
        assert set(ptru_parameter_names(name)) == expected
        assert {"deltaG1_0", "deltaG4_0", "Gact1_0"} <= expected
        assert "q" not in expected and "beta_2_BF" not in expected


@pytest.mark.parametrize("model", available_ptru_models())
def test_pure_pt_only_needs_parameters_for_active_non_bf_paths(model):
    if model == "CO_BF":
        with pytest.raises(ValueError, match="BF-only"):
            ptru_parameter_names(model, "Pt100")
        return
    names = set(ptru_parameter_names(model, "Pt100"))
    assert "deltaG5_0" not in names
    assert "Gact2_BF_0" not in names
    assert ("Gact2_ER_0" in names) == ("ER" in model.split("_"))
    assert ("Gact2_LH_0" in names) == ("LH" in model.split("_"))


def test_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown PtRu model"):
        ptru_parameter_names("CO_UNKNOWN")
