"""Declarative registry for AgPd kinetic model variants."""

from dataclasses import dataclass
from typing import Callable

from mkm.mechanisms.agpd_basic import (
    AgPdCOBFERRLHParameters,
    AgPdCOBFParameters,
    AgPdCOBFERParameters,
    AgPdCOBFLHParameters,
    AgPdCOERLHParameters,
    AgPdCOERParameters,
    AgPdCOLHParameters,
    evaluate_agpd_co_bf,
    evaluate_agpd_co_bf_er,
    evaluate_agpd_co_bf_er_lh,
    evaluate_agpd_co_bf_er_lh_ag10_no_bf,
    evaluate_agpd_co_bf_er_lh_ag10_no_er,
    evaluate_agpd_co_bf_er_lh_capped,
    evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf,
    evaluate_agpd_co_bf_lh,
    evaluate_agpd_co_er,
    evaluate_agpd_co_er_lh,
    evaluate_agpd_co_lh,
)


@dataclass(frozen=True)
class AgPdModelDefinition:
    parameter_class: type
    evaluator: Callable
    pathways: tuple[str, ...]
    config_model_name: str = "CO_BF_ER_LH"
    fit_scopes: tuple[str, ...] = ("individual",)
    fixed_parameters: tuple[tuple[str, float], ...] = ()
    prior_variant: str | None = None
    coverage_cap_mode: str = "none"
    bf_disabled_materials: tuple[str, ...] = ()
    er_disabled_materials: tuple[str, ...] = ()
    pure_pd_prediction_reduction: bool = False

    @property
    def supports_individual(self):
        return "individual" in self.fit_scopes

    @property
    def supports_all_materials(self):
        return "all_materials" in self.fit_scopes


FULL_SCOPES = ("individual", "all_materials")
ALL_ONLY = ("all_materials",)
FITTED_CAPS_MODEL = "CO_BF_ER_LH_fitted_caps_Ag10_no_BF"


AGPD_MODEL_REGISTRY = {
    "CO_LH": AgPdModelDefinition(
        parameter_class=AgPdCOLHParameters,
        evaluator=evaluate_agpd_co_lh,
        pathways=("LH",),
    ),
    "CO_ER": AgPdModelDefinition(
        parameter_class=AgPdCOERParameters,
        evaluator=evaluate_agpd_co_er,
        pathways=("ER",),
    ),
    "CO_BF": AgPdModelDefinition(
        parameter_class=AgPdCOBFParameters,
        evaluator=evaluate_agpd_co_bf,
        pathways=("BF",),
    ),
    "CO_ER_LH": AgPdModelDefinition(
        parameter_class=AgPdCOERLHParameters,
        evaluator=evaluate_agpd_co_er_lh,
        pathways=("ER", "LH"),
    ),
    "CO_BF_LH": AgPdModelDefinition(
        parameter_class=AgPdCOBFLHParameters,
        evaluator=evaluate_agpd_co_bf_lh,
        pathways=("BF", "LH"),
    ),
    "CO_BF_ER": AgPdModelDefinition(
        parameter_class=AgPdCOBFERParameters,
        evaluator=evaluate_agpd_co_bf_er,
        pathways=("BF", "ER"),
    ),
    "CO_BF_ER_LH": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh,
        pathways=("BF", "ER", "LH"),
        fit_scopes=FULL_SCOPES,
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_q1": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        fixed_parameters=(("q", 1.0), ("beta_2_BF", 0.0)),
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_neg": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        prior_variant="negative_beta_BF",
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_Ag10_no_BF": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh_ag10_no_bf,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        bf_disabled_materials=("Ag10Pd90",),
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_Ag10_no_ER": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh_ag10_no_er,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        er_disabled_materials=("Ag10Pd90",),
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_Ag10_no_BF_q1": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh_ag10_no_bf,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        fixed_parameters=(("q", 1.0), ("beta_2_BF", 0.0)),
        bf_disabled_materials=("Ag10Pd90",),
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_Ag10_no_BF_neg": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh_ag10_no_bf,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        prior_variant="negative_beta_BF",
        bf_disabled_materials=("Ag10Pd90",),
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_capped": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh_capped,
        pathways=("BF", "ER", "LH"),
        fit_scopes=FULL_SCOPES,
        coverage_cap_mode="config",
        pure_pd_prediction_reduction=True,
    ),
    "CO_BF_ER_LH_capped_Ag10_no_BF": AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        coverage_cap_mode="config",
        bf_disabled_materials=("Ag10Pd90",),
        pure_pd_prediction_reduction=True,
    ),
    FITTED_CAPS_MODEL: AgPdModelDefinition(
        parameter_class=AgPdCOBFERRLHParameters,
        evaluator=evaluate_agpd_co_bf_er_lh,
        pathways=("BF", "ER", "LH"),
        fit_scopes=ALL_ONLY,
        coverage_cap_mode="fitted",
        bf_disabled_materials=("Ag10Pd90",),
        pure_pd_prediction_reduction=True,
    ),
}


def available_agpd_models():
    return tuple(AGPD_MODEL_REGISTRY)


def available_agpd_all_material_models():
    return tuple(name for name, definition in AGPD_MODEL_REGISTRY.items() if definition.supports_all_materials)


def get_agpd_model_definition(model_name):
    try:
        return AGPD_MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown AgPd model '{model_name}'. Available models: {available_agpd_models()}.") from error


def agpd_model_metadata(model_name):
    definition = get_agpd_model_definition(model_name)
    return {
        "name": model_name,
        "config_model_name": definition.config_model_name,
        "pathways": list(definition.pathways),
        "fit_scopes": list(definition.fit_scopes),
        "fixed_parameters": dict(definition.fixed_parameters),
        "prior_variant": definition.prior_variant,
        "coverage_cap_mode": definition.coverage_cap_mode,
        "bf_disabled_materials": list(definition.bf_disabled_materials),
        "er_disabled_materials": list(definition.er_disabled_materials),
    }
