
"""Finite-rate CO oxidation on Pt100: acidic/basic ER, LH, and ER+LH mechanisms."""

from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from mkm.mechanisms.agpd_basic.electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_tst_rate_constant,
)
from mkm.mechanisms.base import MechanismResult


@dataclass(frozen=True)
class PtCOLHParameters:
    deltaG1_0: object
    deltaG4_0: object
    Gact1_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class PtCOERParameters:
    deltaG1_0: object
    deltaG4_0: object
    Gact1_0: object
    Gact2_ER_0: object
    beta_2_ER: object


@dataclass(frozen=True)
class PtCOERLHParameters:
    deltaG1_0: object
    deltaG4_0: object
    Gact1_0: object
    Gact2_ER_0: object
    beta_2_ER: object
    Gact2_LH_0: object


PT100_PARAMETER_CLASSES = {
    "CO_LH": PtCOLHParameters,
    "CO_ER": PtCOERParameters,
    "CO_ER_LH": PtCOERLHParameters,
}


def evaluate_pt100_co(state, parameters, temperature_K, medium, model_name):
    """Evaluate Pt100 with CO SSA, OH QEA, and negligible COOH coverage."""
    if medium not in ("acidic", "basic"):
        raise ValueError(f"Unsupported medium: {medium}")
    if model_name not in PT100_PARAMETER_CLASSES:
        raise ValueError(f"Unknown Pt100 mechanism: {model_name}")
    if tuple(state.materials) != ("Pt100",):
        raise ValueError("Pt100 mechanisms require only Pt100 model points.")

    E = pt.as_tensor_variable(state.E_V_SHE)
    ln_a_ion = pt.as_tensor_variable(state.ln_electrolyte_concentration)
    ln_a_CO = pt.as_tensor_variable(state.ln_CO_mole_fraction)
    zeros = pt.zeros_like(E)

    # CO + * <-> CO*: finite adsorption/desorption, consistent with AgPd.
    log_K1 = log_equilibrium_constant(parameters.deltaG1_0, temperature_K)
    log_k1 = log_tst_rate_constant(parameters.Gact1_0, temperature_K)
    log_k_minus_1 = log_k1 - log_K1
    log_k1_a_CO = log_k1 + ln_a_CO

    # Water-derived OH in acid (inverse H+ dependence); OH--derived OH in base.
    deltaG4 = electrochemical_free_energy(parameters.deltaG4_0, 1.0, E)
    log_K4 = log_equilibrium_constant(deltaG4, temperature_K)
    log_OH_activity = -ln_a_ion if medium == "acidic" else ln_a_ion
    term_OH = log_K4 + log_OH_activity
    log_A = pt.logsumexp(pt.stack([zeros, term_OH]), axis=0)

    pathways = model_name.removeprefix("CO_").split("_")
    log_k_cons_terms = [log_k_minus_1 + zeros]
    log_k_ER_app = None
    log_k_LH = None

    if "ER" in pathways:
        Gact_ER = electrochemical_activation_energy(
            parameters.Gact2_ER_0, parameters.beta_2_ER, 1.0, E,
        )
        log_k_ER = log_tst_rate_constant(Gact_ER, temperature_K)
        log_k_ER_app = log_k_ER + (zeros if medium == "acidic" else ln_a_ion)
        log_k_cons_terms.append(log_k_ER_app)

    if "LH" in pathways:
        log_k_LH = log_tst_rate_constant(parameters.Gact2_LH_0, temperature_K)

    log_k_cons = pt.logsumexp(pt.stack(log_k_cons_terms), axis=0)
    log_rho = log_k1_a_CO - log_k_cons

    # Site balance: theta_* + theta_CO + theta_OH = 1.
    # CO SSA: k1*a_CO*theta_* = (k-1 + kER_app + kLH*theta_OH)*theta_CO.
    if log_k_LH is None:
        log_den = pt.logsumexp(pt.stack([log_A, log_rho]), axis=0)
        log_theta_empty = -log_den
        log_theta_CO = log_rho - log_den
    else:
        log_lambda = log_k_LH + term_OH - log_k_cons
        A = pt.exp(log_A)
        rho = pt.exp(log_rho)
        lam = pt.exp(log_lambda)

        a = A * lam
        b = rho + A - lam
        sqrt_disc = pt.sqrt(b**2 + 4.0 * a)

        # Stable positive root of a*theta_empty**2 + b*theta_empty - 1 = 0.
        theta_empty = pt.where(
            b >= 0,
            2.0 / (b + sqrt_disc),
            (sqrt_disc - b) / (2.0 * a),
        )
        log_theta_empty = pt.log(theta_empty)
        log_theta_CO = log_rho + log_theta_empty - pt.log1p(lam * theta_empty)

    log_theta_OH = term_OH + log_theta_empty

    log_rates = {}
    if log_k_ER_app is not None:
        log_rates["ER"] = log_k_ER_app + log_theta_CO
    if log_k_LH is not None:
        log_rates["LH"] = log_k_LH + log_theta_CO + log_theta_OH

    log_rate_total = pt.logsumexp(pt.stack(list(log_rates.values())), axis=0)

    pointwise = {
        "theta_empty": pt.exp(log_theta_empty),
        "theta_CO": pt.exp(log_theta_CO),
        "theta_OH": pt.exp(log_theta_OH),
    }
    for pathway, log_rate in log_rates.items():
        pointwise[f"ln_rate_{pathway}"] = log_rate
        pointwise[f"rate_fraction_{pathway}"] = pt.exp(log_rate - log_rate_total)

    return MechanismResult(ln_rate=log_rate_total, pointwise=pointwise)
