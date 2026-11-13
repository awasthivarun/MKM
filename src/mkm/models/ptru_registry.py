"""Registry for implemented PtRu-family kinetic models."""

from dataclasses import dataclass
from typing import Callable

from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES, evaluate_pt100_co


@dataclass(frozen=True)
class PtRuModelDefinition:
    parameter_class: type
    evaluator: Callable
    pathways: tuple[str, ...]
    media: tuple[str, ...] = ("acidic", "basic")
    fit_scopes: tuple[str, ...] = ("individual",)
    material: str = "Pt100"

    @property
    def supports_individual(self):
        return "individual" in self.fit_scopes

    @property
    def supports_all_materials(self):
        return "all_materials" in self.fit_scopes


PTRU_MODEL_REGISTRY = {
    name: PtRuModelDefinition(
        parameter_class=parameter_class,
        evaluator=evaluate_pt100_co,
        pathways=tuple(name.removeprefix("CO_").split("_")),
    )
    for name, parameter_class in PT100_PARAMETER_CLASSES.items()
}


def available_ptru_models(*, medium=None, fit_scope="individual"):
    return tuple(
        name for name, definition in PTRU_MODEL_REGISTRY.items()
        if fit_scope in definition.fit_scopes
        and (medium is None or medium in definition.media)
    )


def get_ptru_model_definition(model_name):
    try:
        return PTRU_MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown PtRu model '{model_name}'. Available models: {tuple(PTRU_MODEL_REGISTRY)}."
        ) from error


def ptru_model_metadata(model_name):
    definition = get_ptru_model_definition(model_name)
    return {
        "material": definition.material,
        "pathways": list(definition.pathways),
        "supported_media": list(definition.media),
        "fit_scopes": list(definition.fit_scopes),
    }