"""Canonical, inference-independent names for AgPd reaction physics.

Legacy fitted model names remain in agpd_registry.py for reproducibility.
"""

from dataclasses import dataclass


COUPLING_PATHWAYS = ("BF", "ER", "LH")
INTERACTION_MARKERS = ("c", "x", "o")
FINITE_STEPS = ("CO", "3", "pOH", "sOH")


@dataclass(frozen=True)
class AgPdPhysics:
    pathways: tuple[str, ...]
    finite_steps: tuple[str, ...]
    interactions: tuple[str, ...]
    modifier: str | None = None

    @property
    def name(self):
        terms = [*(["".join(self.interactions)] if self.interactions else []),
                 *(["CO"] if "CO" in self.finite_steps else []), *self.pathways,
                 *(["3"] if "3" in self.finite_steps else []),
                 *[step for step in ("pOH", "sOH") if step in self.finite_steps]]
        if self.modifier is not None:
            terms.append(self.modifier)
        return "_".join(terms)


def parse_agpd_physics(model_name):
    """Parse a canonical name, rejecting reordered or ambiguous labels.

    Parser support is not a claim that an evaluator exists for every valid name.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Model name must be a nonempty string.")
    tokens = model_name.split("_")
    if any(not token for token in tokens):
        raise ValueError(f"Invalid AgPd model name '{model_name}'.")

    interactions = ()
    if tokens[0].islower() and set(tokens[0]) <= set(INTERACTION_MARKERS):
        interactions = tuple(tokens.pop(0))
        if not interactions or interactions != tuple(c for c in INTERACTION_MARKERS if c in interactions):
            raise ValueError(f"Interactions must appear once each in c-x-o order: '{model_name}'.")

    finite_steps = []
    if tokens and tokens[0] == "CO":
        finite_steps.append(tokens.pop(0))
    pathways = []
    while tokens and tokens[0] in COUPLING_PATHWAYS:
        pathways.append(tokens.pop(0))
    if not pathways or tuple(pathways) != tuple(p for p in COUPLING_PATHWAYS if p in pathways):
        raise ValueError(f"Coupling pathways must appear in BF-ER-LH order: '{model_name}'.")
    if tokens and tokens[0] == "3":
        finite_steps.append(tokens.pop(0))
    for step in ("pOH", "sOH"):
        if tokens and tokens[0] == step:
            finite_steps.append(tokens.pop(0))
    modifier = None
    if tokens:
        if tokens == ["capped"]:
            modifier = tokens.pop(0)
        else:
            raise ValueError(f"Unrecognized or misplaced AgPd model tokens: {tokens}.")

    physics = AgPdPhysics(tuple(pathways), tuple(finite_steps), interactions, modifier)
    if physics.name != model_name:
        raise ValueError(f"Noncanonical AgPd model name '{model_name}'; expected '{physics.name}'.")
    return physics
