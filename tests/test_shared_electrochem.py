"""Regression guards for the AgPd/Pt100 shared electrochemical primitives."""

import numpy as np
import pytest
import pytensor.tensor as pt

from mkm.mechanisms import electrochem as shared
from mkm.mechanisms import pt100
from mkm.mechanisms.agpd_basic import electrochem as legacy
from mkm.mechanisms.agpd_basic import finite_co, full_ssa, legacy_qea


@pytest.mark.parametrize("function_name", (
    "thermal_energy_eV", "log_eyring_prefactor_s_inv", "log_equilibrium_constant",
    "log_tst_rate_constant", "electrochemical_free_energy", "electrochemical_activation_energy",
))
def test_legacy_electrochem_exports_are_shared_objects(function_name):
    function = getattr(shared, function_name)
    assert getattr(legacy, function_name) is function
    for consumer in (pt100, finite_co, full_ssa, legacy_qea):
        if hasattr(consumer, function_name):
            assert getattr(consumer, function_name) is function


def test_shared_electrochemical_potential_gradient_is_unchanged():
    potentials = pt.vector("potentials")
    energy = shared.electrochemical_free_energy(0.2, 0.4, potentials)
    activation = shared.electrochemical_activation_energy(0.7, 0.3, 1.0, potentials)
    data = np.array([-0.25, 0.0, 0.15], dtype=float)

    np.testing.assert_allclose(energy.eval({potentials: data}), 0.2 - 0.4 * data)
    np.testing.assert_allclose(activation.eval({potentials: data}), 0.7 - 0.3 * data)
    np.testing.assert_allclose(pt.grad(pt.sum(energy), potentials).eval({potentials: data}), -0.4)
    np.testing.assert_allclose(pt.grad(pt.sum(activation), potentials).eval({potentials: data}), -0.3)
