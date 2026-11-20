"""Regression for distinct residuals hidden behind nearly identical coverage branches."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from mkm.mechanisms.agpd_basic.lateral_interactions import _branch_acceptance


def test_both_final_branch_residuals_are_required():
    residuals = pt.matrix("residuals")
    disagreements = pt.vector("disagreements")
    maximum, valid = _branch_acceptance(residuals, disagreements, 1e-7, 1e-5)
    evaluate = pytensor.function([residuals, disagreements], [maximum, valid])
    largest, accepted = evaluate(
        np.array([[4e-8, 4e-8, 1.2e-7], [2e-7, 4e-8, 2e-8]], dtype=float),
        np.array([1e-9, 2e-5, 1e-10], dtype=float),
    )
    np.testing.assert_allclose(largest, [2e-7, 4e-8, 1.2e-7])
    np.testing.assert_array_equal(accepted, [False, False, False])


def test_branches_with_valid_residuals_and_matching_coverages_are_accepted():
    residuals = pt.matrix("residuals")
    differences = pt.vector("differences")
    _, valid = _branch_acceptance(residuals, differences, 1e-7, 1e-5)
    evaluate = pytensor.function([residuals, differences], valid)
    assert evaluate(np.array([[1e-8, 9e-8]], dtype=float).T, np.array([1e-6]))[0]
    assert not evaluate(np.array([[1e-8, np.nan]], dtype=float).T, np.array([0.0]))[0]
