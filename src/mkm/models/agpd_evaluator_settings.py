"""Resolve chemistry-specific evaluator settings in fitting, DRC, and run provenance."""


_EVALUATOR_SETTINGS = {
    "evaluate_agpd_mixed_adsorption": ("mixed_adsorption", "mixed_config"),
    "evaluate_agpd_reversible_subset": ("reversible_ssa", "reversible_config"),
    "evaluate_agpd_lateral_interactions": ("lateral_interactions", "lateral_config"),
}


def agpd_evaluator_settings(definition, config):
    """Return (evaluator kwargs, associated source config keys) without changing mechanism selection."""
    kwargs = {}
    keys = []
    if definition.config_model_name == "CO_BF_ER_LH_3_OH":
        kwargs["full_ssa_config"] = config.get("full_ssa", {})
        keys.append("full_ssa")
    evaluator = getattr(definition.evaluator, "func", definition.evaluator)
    selected = _EVALUATOR_SETTINGS.get(evaluator.__name__)
    if selected is not None:
        config_key, argument = selected
        kwargs[argument] = config.get(config_key, {})
        keys.append(config_key)
    return kwargs, tuple(keys)
