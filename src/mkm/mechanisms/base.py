from dataclasses import dataclass, field
from typing import Mapping, Protocol

import pytensor.tensor as pt

from mkm.model_inputs import ModelPointInputs


@dataclass(frozen=True)
class MechanismResult:
    ln_rate: object
    pointwise: Mapping[str, object] = field(default_factory=dict)


class Mechanism(Protocol):
    def __call__(self, inputs: ModelPointInputs) -> MechanismResult: ...


def validate_mechanism_result(result: MechanismResult):
    ln_rate = pt.as_tensor_variable(result.ln_rate)

    if ln_rate.ndim != 1:
        raise ValueError("Mechanism ln_rate output must be one-dimensional.")

    if "ln_rate" in result.pointwise:
        raise ValueError("'ln_rate' is reserved as the primary mechanism output and must not also appear in pointwise outputs.")

    for name, value in result.pointwise.items():
        if not isinstance(name, str):
            raise ValueError("Pointwise observable names must be strings.")

        if not name:
            raise ValueError("Pointwise observable names must not be empty.")

        tensor = pt.as_tensor_variable(value)

        if tensor.ndim != 1:
            raise ValueError(f"Pointwise observable '{name}' must be one-dimensional.")

    return result