"""Settings dispatch is identical for fitting, DRC, and provenance metadata."""

from types import SimpleNamespace

import pytest

from mkm.models.agpd_evaluator_settings import agpd_evaluator_settings
from mkm.provenance import AGPD_MODEL_SOURCE_FILES, PTRU_MODEL_SOURCE_FILES


@pytest.mark.parametrize("evaluator_name,key,argument", [
    ("evaluate_agpd_mixed_adsorption", "mixed_adsorption", "mixed_config"),
    ("evaluate_agpd_reversible_subset", "reversible_ssa", "reversible_config"),
    ("evaluate_agpd_lateral_interactions", "lateral_interactions", "lateral_config"),
])
def test_dispatch_passes_only_active_evaluator_settings(evaluator_name, key, argument):
    def evaluator():
        pass

    evaluator.__name__ = evaluator_name
    definition = SimpleNamespace(evaluator=evaluator, config_model_name="CO_BF_ER_LH")
    config = {key: {"sentinel": 123}, "unrelated": 456}
    kwargs, selected = agpd_evaluator_settings(definition, config)
    assert kwargs == {argument: {"sentinel": 123}}
    assert selected == (key,)


def test_legacy_full_ssa_settings_are_preserved():
    def evaluate_agpd_full_ssa():
        pass

    definition = SimpleNamespace(evaluator=evaluate_agpd_full_ssa, config_model_name="CO_BF_ER_LH_3_OH")
    kwargs, selected = agpd_evaluator_settings(definition, {"full_ssa": {"enabled": True}})
    assert kwargs == {"full_ssa_config": {"enabled": True}}
    assert selected == ("full_ssa",)


def test_shared_config_and_composition_helpers_are_part_of_scoped_provenance():
    assert "src/mkm/models/agpd_evaluator_settings.py" in AGPD_MODEL_SOURCE_FILES
    assert "src/mkm/models/composition_bounds.py" in AGPD_MODEL_SOURCE_FILES
    assert "src/mkm/models/composition_bounds.py" in PTRU_MODEL_SOURCE_FILES
