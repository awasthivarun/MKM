"""Shared finite-rate CO SSA with OH quasi-equilibrium and a single-site balance."""

from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt


@dataclass(frozen=True)
class COSSACoverages:
    log_theta_empty: object
    log_theta_CO: object
    log_theta_OH: object


def solve_co_ssa_qea_oh(
    log_term_oh,
    log_k1_a_co,
    log_k_minus_1,
    *,
    log_k_other=(),
    log_k_lh_app=None,
    theta_co_max=None,
):
    """Solve CO SSA while retaining the existing analytic capped/uncapped formulations."""
    term_oh = pt.as_tensor_variable(log_term_oh)
    zeros = pt.zeros_like(term_oh)

    log_k1_a_co = pt.as_tensor_variable(log_k1_a_co) + zeros
    log_k_minus_1 = pt.as_tensor_variable(log_k_minus_1) + zeros
    log_A = pt.logsumexp(pt.stack([zeros, term_oh]), axis=0)

    consumption_terms = [log_k_minus_1]
    consumption_terms.extend(pt.as_tensor_variable(rate) + zeros for rate in log_k_other)
    log_k_cons = pt.logsumexp(pt.stack(consumption_terms), axis=0)

    log_rho = log_k1_a_co - log_k_cons
    A = pt.exp(log_A)
    rho = pt.exp(log_rho)

    if log_k_lh_app is None:
        lambda_lh = pt.zeros_like(rho)
    else:
        log_k_lh_app = pt.as_tensor_variable(log_k_lh_app) + zeros
        lambda_lh = pt.exp(log_k_lh_app - log_k_cons)

    if theta_co_max is None:
        if log_k_lh_app is None:
            log_denominator = pt.logsumexp(pt.stack([log_A, log_rho]), axis=0)
            log_theta_empty = -log_denominator
            log_theta_CO = log_rho - log_denominator
        else:
            quadratic_a = A * lambda_lh
            quadratic_b = rho + A - lambda_lh
            sqrt_disc = pt.sqrt(quadratic_b**2 + 4.0 * quadratic_a)

            theta_empty = pt.where(
                quadratic_b >= 0,
                2.0 / (quadratic_b + sqrt_disc),
                (sqrt_disc - quadratic_b) / (2.0 * quadratic_a),
            )
            log_theta_empty = pt.log(theta_empty)
            log_theta_CO = log_rho + log_theta_empty - pt.log1p(lambda_lh * theta_empty)
    else:
        theta_co_max = pt.as_tensor_variable(theta_co_max)
        inv_theta_co_max = 1.0 / theta_co_max

        quadratic_b = rho * (1.0 + inv_theta_co_max) + A + lambda_lh
        discriminant = (
            (A + lambda_lh)**2
            + 2.0 * rho * (A * (1.0 + inv_theta_co_max) + lambda_lh * (inv_theta_co_max - 1.0))
            + rho**2 * (1.0 - inv_theta_co_max)**2
        )
        sqrt_disc = pt.sqrt(discriminant)

        log_theta_CO = np.log(2.0) + log_rho - pt.log(quadratic_b + sqrt_disc)
        theta_CO = pt.exp(log_theta_CO)
        log_theta_empty = pt.log1p(-theta_CO) - log_A

    return COSSACoverages(
        log_theta_empty=log_theta_empty,
        log_theta_CO=log_theta_CO,
        log_theta_OH=term_oh + log_theta_empty,
    )