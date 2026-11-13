from dataclasses import fields

import pytest

from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES, evaluate_pt100_co
from mkm.models.ptru_registry import (
    available_ptru_models,
    get_ptru_model_definition,
    ptru_model_metadata,
)


def test_ptru_registry_contains_current_pt100_models():
    assert available_ptru_models() == ("CO_LH", "CO_ER", "CO_ER_LH")
    assert available_ptru_models(medium="acidic") == available_ptru_models()
    assert available_ptru_models(medium="basic") == available_ptru_models()
    assert available_ptru_models(fit_scope="all_materials") == ()

    for name in available_ptru_models():
        definition = get_ptru_model_definition(name)

        assert definition.parameter_class is PT100_PARAMETER_CLASSES[name]
        assert definition.evaluator is evaluate_pt100_co
        assert definition.supports_individual
        assert not definition.supports_all_materials
        assert definition.material == "Pt100"
        assert definition.pathways == tuple(name.removeprefix("CO_").split("_"))

        names = {field.name for field in fields(definition.parameter_class)}
        assert {"deltaG1_0", "deltaG4_0", "Gact1_0"} <= names


def test_ptru_registry_metadata_is_consistent():
    metadata = ptru_model_metadata("CO_ER_LH")

    assert metadata["material"] == "Pt100"
    assert metadata["pathways"] == ["ER", "LH"]
    assert metadata["supported_media"] == ["acidic", "basic"]
    assert metadata["fit_scopes"] == ["individual"]


def test_ptru_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown PtRu model"):
        get_ptru_model_definition("CO_BF")