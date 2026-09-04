"""Pure-Pd limit of the finite-CO AgPd mechanism.

This module contains the reduced parameterization needed for individual Pd100
fits. It is also the exact x_Ag = 0 prediction limit of CO_BF_ER_LH: the BF
channel is absent, while finite CO adsorption/desorption, ER, LH, and Pd-OH
chemistry remain unchanged. An optional CO packing cap is used for the capped
all-material model's pure-Pd prediction limit.
"""

from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from mkm.mechanisms.agpd_basic import (
    AgPdPointState,
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_surface_fraction,
    log_tst_rate_constant,
    logsumexp_pathways,
    solve_pd_co_ssa_qea_oh,
)
from mkm.mechanisms.base import MechanismResult


@dataclass(frozen=True)
class PdCOERLHParameters:
    deltaG1_0: object
    deltaG4_0: object

    beta_2_ER: object

    Gact1_0: object
    Gact2_ER_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class PdCOERLHResult:
    mechanism_result: MechanismResult

    log_rate_ER: object
    log_rate_LH: object
    log_rate_total: object

    log_K1: object
    log_K4: object

    log_k1: object
    log_k_minus_1: object
    log_k2_ER: object
    log_k2_LH: object


def evaluate_pd_co_er_lh(
    state: AgPdPointState,
    parameters: PdCOERLHParameters,
    temperature_K,
    theta_CO_max=None,
):
    """Evaluate finite-CO ER+LH chemistry on pure Pd.

    The equations are the x_Ag = 0 limit of CO_BF_ER_LH. No Ag adsorption
    quantity or BF-specific parameter appears in this reduced model.
    """
    if np.any(state.Ag_fraction != 0.0):
        raise ValueError("CO_ER_LH is defined only for pure-Pd model points.")
    if np.any(state.Pd_fraction <= 0.0):
        raise ValueError("CO_ER_LH requires a positive Pd fraction.")

    E = pt.as_tensor_variable(state.E_V_SHE)

    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(
        delta_G_0_eV=parameters.deltaG4_0,
        electron_transfer=1.0,
        E_V_SHE=E,
    )

    Gact1 = pt.as_tensor_variable(parameters.Gact1_0)
    Gact2_ER = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_ER_0,
        beta=parameters.beta_2_ER,
        electron_transfer=1.0,
        E_V_SHE=E,
    )
    Gact2_LH = pt.as_tensor_variable(parameters.Gact2_LH_0)

    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)

    log_k1 = log_tst_rate_constant(activation_G_eV=Gact1, temperature_K=temperature_K)
    log_k_minus_1 = log_k1 - log_K1

    log_k2_ER = log_tst_rate_constant(
        activation_G_eV=Gact2_ER,
        temperature_K=temperature_K,
    )
    log_k2_LH = log_tst_rate_constant(
        activation_G_eV=Gact2_LH,
        temperature_K=temperature_K,
    )

    log_k1_a_CO = log_k1 + pt.as_tensor_variable(state.ln_a_CO)
    log_k_ER_app = log_k2_ER + pt.as_tensor_variable(state.ln_a_OH)

    term_OH_Pd = log_K4 + pt.as_tensor_variable(state.ln_a_OH)
    log_k_LH_app = (
        log_k2_LH
        + term_OH_Pd
        + log_surface_fraction(state.Pd_fraction)
    )

    point_template = pt.zeros_like(log_k_ER_app)
    log_k_BF_app = point_template - np.inf

    solver_theta_CO_max = 1.0 if theta_CO_max is None else theta_CO_max

    pd_coverages = solve_pd_co_ssa_qea_oh(
        log_K_OH_Pd=log_K4,
        log_k1_a_CO=log_k1_a_CO,
        log_k_minus_1=log_k_minus_1,
        log_k_BF_app=log_k_BF_app,
        log_k_ER_app=log_k_ER_app,
        log_k_LH_app=log_k_LH_app,
        state=state,
        theta_CO_max=solver_theta_CO_max,
    )

    log_rate_ER = log_k_ER_app + pd_coverages.log_theta_CO
    log_rate_LH = (
        log_k_LH_app
        + pd_coverages.log_theta_empty_Pd
        + pd_coverages.log_theta_CO
    )
    log_rate_total = logsumexp_pathways(log_rate_ER, log_rate_LH)

    rate_fraction_ER = pt.exp(log_rate_ER - log_rate_total)
    rate_fraction_LH = pt.exp(log_rate_LH - log_rate_total)

    theta_CO = pt.exp(pd_coverages.log_theta_CO)
    pointwise = {
        "theta_CO": theta_CO,
        "theta_OH_Pd": pt.exp(pd_coverages.log_theta_OH_Pd),
        "theta_empty_Pd": pt.exp(pd_coverages.log_theta_empty_Pd),
        "ln_rate_ER": log_rate_ER,
        "ln_rate_LH": log_rate_LH,
        "rate_fraction_ER": rate_fraction_ER,
        "rate_fraction_LH": rate_fraction_LH,
    }

    if theta_CO_max is not None:
        pointwise["theta_CO_site_occupation"] = (
            theta_CO / pt.as_tensor_variable(theta_CO_max)
        )

    mechanism_result = MechanismResult(
        ln_rate=log_rate_total,
        pointwise=pointwise,
    )

    return PdCOERLHResult(
        mechanism_result=mechanism_result,
        log_rate_ER=log_rate_ER,
        log_rate_LH=log_rate_LH,
        log_rate_total=log_rate_total,
        log_K1=log_K1,
        log_K4=log_K4,
        log_k1=log_k1,
        log_k_minus_1=log_k_minus_1,
        log_k2_ER=log_k2_ER,
        log_k2_LH=log_k2_LH,
    )
