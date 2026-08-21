import numpy as np
import pytensor.tensor as pt
import pytest

from mkm.mechanisms.base import MechanismResult, validate_mechanism_result


def test_mechanism_result_accepts_log_rate_vector():
    result = MechanismResult(ln_rate=pt.vector("ln_rate"))
    validated = validate_mechanism_result(result)

    assert validated is result


def test_mechanism_result_accepts_pointwise_observables():
    result = MechanismResult(
        ln_rate=pt.vector("ln_rate"),
        pointwise={"theta_CO": pt.vector("theta_CO"), "theta_free": pt.vector("theta_free")},
    )

    validated = validate_mechanism_result(result)

    assert set(validated.pointwise) == {"theta_CO", "theta_free"}


def test_mechanism_result_rejects_scalar_log_rate():
    result = MechanismResult(ln_rate=pt.scalar("ln_rate"))

    with pytest.raises(ValueError, match="one-dimensional"):
        validate_mechanism_result(result)


def test_mechanism_result_rejects_nonvector_pointwise_output():
    result = MechanismResult(ln_rate=pt.vector("ln_rate"), pointwise={"theta_CO": pt.matrix("theta_CO")})

    with pytest.raises(ValueError, match="one-dimensional"):
        validate_mechanism_result(result)


def test_mechanism_result_reserves_ln_rate_name():
    result = MechanismResult(ln_rate=pt.vector("ln_rate"), pointwise={"ln_rate": pt.vector("duplicate")})

    with pytest.raises(ValueError, match="reserved"):
        validate_mechanism_result(result)