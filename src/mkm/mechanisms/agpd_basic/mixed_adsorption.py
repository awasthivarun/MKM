"""Mixed QEA/finite-rate adsorption when COOH removal is instantaneous.

CO is the only nonlinear coverage unknown after eliminating both OH balances. A
vectorized, differentiable Newton iteration handles arbitrary BF/ER/LH subsets.
There are no lateral interactions or explicit COOH sites in this limit.
"""

from dataclasses import dataclass

import numpy as np
import pytensor
import pytensor.tensor as pt

from mkm.mechanisms.base import MechanismResult
from mkm.mechanisms.electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_tst_rate_constant,
)


@dataclass(frozen=True)
class AgPdMixedAdsorptionResult:
    mechanism_result: MechanismResult
    log_rate_total: object


def _adsorption_state(theta_CO, *, a4, b4, lh, a5, b5, bf, er, k1a, km1, K1a, co_finite):
    """Return coverages, pathway rates, and CO balance and its analytic derivative."""
    z = pt.as_tensor_variable(theta_CO)
    primary_denom = a4 + b4 + lh * z
    theta_pOH = a4 * (1.0 - z) / primary_denom
    dpOH_dCO = -a4 * (a4 + b4 + lh) / primary_denom**2
    theta_empty_primary = 1.0 - z - theta_pOH
    dempty_dCO = -1.0 - dpOH_dCO

    secondary_denom = a5 + b5 + bf * z
    theta_sOH = a5 / secondary_denom
    dsOH_dCO = -a5 * bf / secondary_denom**2
    theta_empty_secondary = 1.0 - theta_sOH

    rate_BF = bf * z * theta_sOH
    rate_ER = er * z
    rate_LH = lh * z * theta_pOH
    rate_total = rate_BF + rate_ER + rate_LH
    d_rate_dCO = bf * (theta_sOH + z * dsOH_dCO) + er + lh * (theta_pOH + z * dpOH_dCO)

    if co_finite:
        residual_CO = k1a * theta_empty_primary - km1 * z - rate_total
        derivative_CO = k1a * dempty_dCO - km1 - d_rate_dCO
    else:
        residual_CO = K1a * theta_empty_primary - z
        derivative_CO = K1a * dempty_dCO - 1.0
    return {
        "theta_CO": z,
        "theta_pOH": theta_pOH,
        "theta_sOH": theta_sOH,
        "theta_empty_primary": theta_empty_primary,
        "theta_empty_secondary": theta_empty_secondary,
        "rate_BF": rate_BF,
        "rate_ER": rate_ER,
        "rate_LH": rate_LH,
        "rate_total": rate_total,
        "residual_CO": residual_CO,
        "derivative_CO": derivative_CO,
    }


def evaluate_agpd_mixed_adsorption(state, parameters, temperature_K, *, pathways, finite_steps,
                                    mixed_config=None):
    """Evaluate finite CO/pOH/sOH subsets with fast irreversible COOH removal."""
    if mixed_config is None:
        mixed_config = {}
    iterations = int(mixed_config.get("newton_iterations", 32))
    max_step = float(mixed_config.get("newton_max_step", 0.25))
    tolerance = float(mixed_config.get("relative_residual_tolerance", 1e-7))
    if iterations < 1 or not np.isfinite(max_step) or not 0 < max_step <= 1:
        raise ValueError("Mixed-adsorption Newton settings must have iterations >= 1 and step in (0, 1].")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Mixed-adsorption relative residual tolerance must be positive and finite.")
    if np.any(state.Pd_fraction <= 0.0):
        raise ValueError("Mixed adsorption requires positive primary-site fraction.")

    pathways, finite_steps = tuple(pathways), tuple(finite_steps)
    E = pt.as_tensor_variable(state.E_V_SHE)
    zeros = pt.zeros_like(E)
    a_CO = pt.exp(pt.as_tensor_variable(state.ln_a_CO))
    a_OH = pt.exp(pt.as_tensor_variable(state.ln_a_OH))
    fp = pt.as_tensor_variable(state.Pd_fraction)
    fs = pt.as_tensor_variable(state.Ag_fraction)

    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(parameters.deltaG4_0, 1.0, E)
    K1a = pt.exp(log_equilibrium_constant(deltaG1, temperature_K)) * a_CO
    K4a = pt.exp(log_equilibrium_constant(deltaG4, temperature_K)) * a_OH

    if "CO" in finite_steps:
        log_k1 = log_tst_rate_constant(parameters.Gact1_0, temperature_K)
        k1a = pt.exp(log_k1) * a_CO
        km1 = pt.exp(log_k1 - log_equilibrium_constant(deltaG1, temperature_K))
    else:
        k1a, km1 = zeros, zeros

    if "pOH" in finite_steps:
        g4act = electrochemical_activation_energy(parameters.Gact4_0, parameters.beta_4, 1.0, E)
        log_k4 = log_tst_rate_constant(g4act, temperature_K)
        a4 = pt.exp(log_k4) * a_OH
        b4 = pt.exp(log_k4 - log_equilibrium_constant(deltaG4, temperature_K))
    else:
        a4, b4 = K4a, pt.ones_like(E)

    if "BF" in pathways or "sOH" in finite_steps:
        deltaG5 = electrochemical_free_energy(parameters.deltaG5_0, parameters.q, E)
        K5a = pt.exp(log_equilibrium_constant(deltaG5, temperature_K)) * a_OH
        if "sOH" in finite_steps:
            g5act = electrochemical_activation_energy(parameters.Gact5_0, parameters.beta_5, parameters.q, E)
            log_k5 = log_tst_rate_constant(g5act, temperature_K)
            a5 = pt.exp(log_k5) * a_OH
            b5 = pt.exp(log_k5 - log_equilibrium_constant(deltaG5, temperature_K))
        else:
            a5, b5 = K5a, pt.ones_like(E)
    else:
        a5, b5 = zeros, pt.ones_like(E)

    if "BF" in pathways:
        gbf = electrochemical_activation_energy(
            parameters.Gact2_BF_0, parameters.beta_2_BF, 1.0 - parameters.q, E
        )
        kbf = pt.exp(log_tst_rate_constant(gbf, temperature_K))
        bf = fs * kbf  # BF rate normalized per primary site.
        bf_secondary = pt.where(fs > 0.0, fp * kbf, 0.0)
    else:
        bf, bf_secondary = zeros, zeros
    if "ER" in pathways:
        ger = electrochemical_activation_energy(parameters.Gact2_ER_0, parameters.beta_2_ER, 1.0, E)
        er = pt.exp(log_tst_rate_constant(ger, temperature_K)) * a_OH
    else:
        er = zeros
    if "LH" in pathways:
        lh = fp * pt.exp(log_tst_rate_constant(parameters.Gact2_LH_0, temperature_K)) + zeros
    else:
        lh = zeros

    # Only finite-rate OH adsorption reacts to OH consumption; QEA ignores the corresponding net flux.
    reactive_lh = lh if "pOH" in finite_steps else zeros
    # A BF event consumes one secondary OH per primary CO. Per-secondary-site consumption
    # is fp * kBF * theta_CO * theta_sOH, whereas measured rate per primary site uses fs * kBF.
    reactive_bf = bf_secondary if "sOH" in finite_steps else zeros
    kwargs = dict(a4=a4, b4=b4, lh=reactive_lh, a5=a5, b5=b5, bf=reactive_bf,
                  er=er, k1a=k1a, km1=km1, K1a=K1a, co_finite="CO" in finite_steps)

    # The OH balances use reactive coefficients. CO consumption must use the actual coupling
    # rates even when OH is in QEA, so construct total flux separately below.
    def _trial(z):
        trial = _adsorption_state(z, **kwargs)
        rate_BF = bf * z * trial["theta_sOH"]
        rate_ER = er * z
        rate_LH = lh * z * trial["theta_pOH"]
        rate = rate_BF + rate_ER + rate_LH
        if "CO" in finite_steps:
            # Differentiate actual coupling flux with respect to z, including reactive OH denominators.
            dp = -a4 * (a4 + b4 + reactive_lh) / (a4 + b4 + reactive_lh * z)**2
            ds = -a5 * reactive_bf / (a5 + b5 + reactive_bf * z)**2
            dr = bf * (trial["theta_sOH"] + z * ds) + er + lh * (trial["theta_pOH"] + z * dp)
            residual = k1a * trial["theta_empty_primary"] - km1 * z - rate
            derivative = k1a * (-1.0 - dp) - km1 - dr
        else:
            residual = K1a * trial["theta_empty_primary"] - z
            derivative = trial["derivative_CO"]
        trial.update({"rate_BF": rate_BF, "rate_ER": rate_ER, "rate_LH": rate_LH,
                      "rate_total": rate, "residual_CO": residual, "derivative_CO": derivative})
        return trial

    initial = pt.clip(K1a / (1.0 + K1a + K4a), 1e-14, 1.0 - 1e-14)

    def _newton_step(previous):
        trial = _trial(previous)
        derivative = trial["derivative_CO"]
        safe_derivative = pt.where(pt.abs(derivative) > 1e-30, derivative, -1e-30)
        step = pt.clip(trial["residual_CO"] / safe_derivative, -max_step, max_step)
        return pt.clip(previous - step, 1e-14, 1.0 - 1e-14)

    sequence = pytensor.scan(fn=_newton_step, outputs_info=initial, n_steps=iterations,
                             strict=False, return_updates=False, name="agpd_mixed_oh_newton")
    solved = _trial(sequence[-1])
    z = solved["theta_CO"]
    theta_pOH = solved["theta_pOH"]
    theta_sOH = pt.where(fs > 0.0, solved["theta_sOH"], 0.0)
    theta_empty_s = pt.where(fs > 0.0, solved["theta_empty_secondary"], 1.0)
    rate = solved["rate_total"]
    norm = pt.maximum(pt.abs(k1a * solved["theta_empty_primary"]) + pt.abs(km1 * z) + rate, 1e-30)
    if "CO" not in finite_steps:
        norm = pt.maximum(K1a * solved["theta_empty_primary"] + z, 1e-30)
    relative_residual = pt.abs(solved["residual_CO"]) / norm

    # Do not silently substitute an unconverged coverage solution into the likelihood.
    log_rate = pt.where(relative_residual <= tolerance,
                        pt.where(rate > 0.0, pt.log(pt.maximum(rate, 1e-300)), -np.inf), np.nan)
    pointwise = {
        "theta_CO": z,
        "theta_OH_Pd": theta_pOH,
        "theta_empty_Pd": solved["theta_empty_primary"],
        "theta_OH_Ag": theta_sOH,
        "theta_empty_Ag": theta_empty_s,
        "coverage_solver_relative_residual": relative_residual,
    }
    for pathway in pathways:
        rate_part = solved[f"rate_{pathway}"]
        pointwise[f"ln_rate_{pathway}"] = pt.where(rate_part > 0.0,
                                                    pt.log(pt.maximum(rate_part, 1e-300)), -np.inf)
        pointwise[f"rate_fraction_{pathway}"] = pt.where(rate > 0.0, rate_part / pt.maximum(rate, 1e-300), 0.0)
    return AgPdMixedAdsorptionResult(MechanismResult(ln_rate=log_rate, pointwise=pointwise), log_rate)
