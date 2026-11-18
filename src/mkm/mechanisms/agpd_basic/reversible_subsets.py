"""Reversible BF/ER/LH coupling with finite COOH oxidation (step 3).

CO, primary OH and secondary OH can independently be in QEA or SSA. Their
balances eliminate analytically, leaving one COOH-forming affinity to solve.
The final COOH + OH -> CO2 step is irreversible (zero product activity).
"""

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


def _coverages_from_affinity(affinity, *, K1a, K4a, K5a, k3a, k1_reverse,
                             co_finite, pOH_finite, sOH_finite, k4_reverse,
                             k5_forward_aOH, k5_reverse, kBF, kBF_reverse,
                             kER_reverse, kLH, kLH_reverse, K_ER_aOH, fp, fs):
    """Analytic adsorption balances given A = ln(K_ER aOH theta_CO/theta_COOH).

    Coupling rates are per primary site. BF consumption per secondary site is
    fp * kBF * theta_CO * theta_sOH, preserving site-count stoichiometry.
    """
    fp = pt.as_tensor_variable(fp)
    fs = pt.as_tensor_variable(fs)
    t = K_ER_aOH * pt.exp(-affinity)       # theta_COOH / theta_CO
    R = K1a / (1.0 + (k3a / k1_reverse) * t) if co_finite else K1a
    S = t * R                             # theta_COOH / theta_empty_primary
    D = 1.0 + R + S

    if pOH_finite:
        u = fp * kLH / k4_reverse
        v = fp * kLH_reverse / k4_reverse
        A = D * u * R + v * S
        B = D + K4a - u * R
        root = pt.sqrt(pt.maximum(B * B + 4.0 * A, 0.0))
        # Stable quadratic root for A*theta_empty**2 + B*theta_empty - 1 = 0.
        empty_p = pt.where(B >= 0.0, 2.0 / pt.maximum(B + root, 1e-300), (root - B) / pt.maximum(2.0 * A, 1e-300))
        co = R * empty_p
        cooh = S * empty_p
        oh_p = empty_p * (K4a + v * cooh) / (1.0 + u * co)
        coefficient_lh = fp * kLH_reverse * empty_p / (1.0 + u * co)
    else:
        empty_p = 1.0 / (D + K4a)
        co, cooh = R * empty_p, S * empty_p
        oh_p = K4a * empty_p
        coefficient_lh = fp * kLH_reverse * empty_p

    if sOH_finite:
        denominator = k5_forward_aOH + k5_reverse + fp * (kBF * co + kBF_reverse * cooh)
        oh_s = (k5_forward_aOH + fp * kBF_reverse * cooh) / denominator
        empty_s = (k5_reverse + fp * kBF * co) / denominator
        coefficient_bf = fs * kBF_reverse * k5_reverse / denominator
    else:
        empty_s = 1.0 / (1.0 + K5a)
        oh_s = K5a * empty_s
        coefficient_bf = fs * kBF_reverse * empty_s

    # Each route has the same net COOH-forming affinity after OH SSA elimination.
    # Factoring net rates as coefficient * theta_COOH * expm1(A) avoids cancellation.
    return dict(
        theta_CO=co, theta_COOH=cooh, theta_OH_Pd=oh_p, theta_empty_Pd=empty_p,
        theta_OH_Ag=pt.where(fs > 0.0, oh_s, 0.0),
        theta_empty_Ag=pt.where(fs > 0.0, empty_s, 1.0),
        coefficient_BF=coefficient_bf, coefficient_ER=pt.as_tensor_variable(kER_reverse),
        coefficient_LH=coefficient_lh,
        coefficient_total=coefficient_bf + kER_reverse + coefficient_lh,
    )


def evaluate_agpd_reversible_subset(state, parameters, temperature_K, *, pathways, finite_steps,
                                    reversible_config=None):
    """Solve a finite-step-3 network with independently selected QEA/SSA steps."""
    if reversible_config is None:
        reversible_config = {}
    iterations = int(reversible_config.get("affinity_iterations", 48))
    damping = float(reversible_config.get("affinity_damping", 1.0))
    tolerance = float(reversible_config.get("relative_residual_tolerance", 1e-8))
    if iterations < 1 or not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("Reversible SSA requires positive iterations and damping in (0, 1].")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Reversible SSA relative residual tolerance must be positive and finite.")
    if np.any(state.Pd_fraction <= 0.0):
        raise ValueError("Reversible SSA requires a positive primary-site fraction.")
    pathways, finite_steps = tuple(pathways), tuple(finite_steps)
    if "3" not in finite_steps or not pathways or set(pathways) - {"BF", "ER", "LH"}:
        raise ValueError("Reversible subset requires finite step 3 and at least one coupling pathway.")

    E = pt.as_tensor_variable(state.E_V_SHE)
    zeros = pt.zeros_like(E)
    ln_aOH = pt.as_tensor_variable(state.ln_a_OH)
    fp, fs = pt.as_tensor_variable(state.Pd_fraction), pt.as_tensor_variable(state.Ag_fraction)
    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0) + zeros
    deltaG4 = electrochemical_free_energy(parameters.deltaG4_0, 1.0, E)
    deltaG3 = electrochemical_free_energy(parameters.deltaG3_0, 1.0, E)
    logK1 = log_equilibrium_constant(deltaG1, temperature_K)
    logK4 = log_equilibrium_constant(deltaG4, temperature_K)
    K1a = pt.exp(logK1 + pt.as_tensor_variable(state.ln_a_CO))
    K4a = pt.exp(logK4 + ln_aOH)
    K_ER_aOH = pt.exp(log_equilibrium_constant(deltaG3 - deltaG1, temperature_K) + ln_aOH)

    g3 = electrochemical_activation_energy(parameters.Gact3_0, parameters.beta_3, 1.0, E)
    k3a = pt.exp(log_tst_rate_constant(g3, temperature_K) + ln_aOH)
    if "CO" in finite_steps:
        logk1 = log_tst_rate_constant(parameters.Gact1_0, temperature_K)
        k1_reverse = pt.exp(logk1 - logK1) + zeros
    else:
        k1_reverse = pt.ones_like(E)  # unused in the CO-QEA branch

    if "pOH" in finite_steps:
        g4 = electrochemical_activation_energy(parameters.Gact4_0, parameters.beta_4, 1.0, E)
        logk4 = log_tst_rate_constant(g4, temperature_K)
        k4_reverse = pt.exp(logk4 - logK4)
    else:
        k4_reverse = pt.ones_like(E)

    if "BF" in pathways or "sOH" in finite_steps:
        deltaG5 = electrochemical_free_energy(parameters.deltaG5_0, parameters.q, E)
        logK5 = log_equilibrium_constant(deltaG5, temperature_K)
        K5a = pt.exp(logK5 + ln_aOH)
    else:
        deltaG5 = zeros
        K5a = zeros
        logK5 = zeros
    if "sOH" in finite_steps:
        g5 = electrochemical_activation_energy(parameters.Gact5_0, parameters.beta_5, parameters.q, E)
        logk5 = log_tst_rate_constant(g5, temperature_K)
        k5a = pt.exp(logk5 + ln_aOH)
        k5_reverse = pt.exp(logk5 - logK5)
    else:
        k5a, k5_reverse = K5a, pt.ones_like(E)

    if "BF" in pathways:
        gBF = electrochemical_activation_energy(
            parameters.Gact2_BF_0, parameters.beta_2_BF, 1.0 - parameters.q, E
        )
        logkBF = log_tst_rate_constant(gBF, temperature_K)
        kBF = pt.exp(logkBF)
        kBF_reverse = pt.exp(logkBF - log_equilibrium_constant(deltaG3 - deltaG1 - deltaG5,
                                                                  temperature_K))
    else:
        kBF, kBF_reverse = zeros, zeros
    if "ER" in pathways:
        gER = electrochemical_activation_energy(parameters.Gact2_ER_0, parameters.beta_2_ER, 1.0, E)
        logkER = log_tst_rate_constant(gER, temperature_K)
        kER_reverse = pt.exp(logkER - log_equilibrium_constant(deltaG3 - deltaG1, temperature_K))
    else:
        kER_reverse = zeros
    if "LH" in pathways:
        kLH = pt.exp(log_tst_rate_constant(parameters.Gact2_LH_0, temperature_K)) + zeros
        kLH_reverse = kLH * pt.exp(-log_equilibrium_constant(deltaG3 - deltaG1 - deltaG4,
                                                             temperature_K))
    else:
        kLH, kLH_reverse = zeros, zeros

    kwargs = dict(
        K1a=K1a, K4a=K4a, K5a=K5a, k3a=k3a, k1_reverse=k1_reverse,
        co_finite="CO" in finite_steps, pOH_finite="pOH" in finite_steps,
        sOH_finite="sOH" in finite_steps, k4_reverse=k4_reverse,
        k5_forward_aOH=k5a, k5_reverse=k5_reverse,
        kBF=kBF, kBF_reverse=kBF_reverse, kER_reverse=kER_reverse,
        kLH=kLH, kLH_reverse=kLH_reverse, K_ER_aOH=K_ER_aOH, fp=fp, fs=fs,
    )

    def _trial(affinity):
        return _coverages_from_affinity(affinity, **kwargs)

    initial_state = _trial(zeros)
    initial = pt.log1p(k3a / pt.maximum(initial_state["coefficient_total"], 1e-300))

    def _step(previous):
        solved = _trial(previous)
        target = pt.log1p(k3a / pt.maximum(solved["coefficient_total"], 1e-300))
        return (1.0 - damping) * previous + damping * target

    sequence = pytensor.scan(
        fn=_step, outputs_info=initial, n_steps=iterations,
        name="agpd_reversible_affinity", strict=False, return_updates=False,
    )
    affinity = sequence[-1]
    solved = _trial(affinity)
    departure = pt.expm1(affinity)
    coefficient = solved["coefficient_total"]
    relative_residual = pt.abs(coefficient * departure - k3a) / pt.maximum(k3a, 1e-300)
    inactive = coefficient <= 0.0  # e.g., BF-only chemistry on a pure-Pd surface
    # A missing coupling route must give exactly zero production, not a numerical floor.
    rate = pt.where(inactive, 0.0, k3a * solved["theta_COOH"])
    valid = pt.or_(inactive, relative_residual <= tolerance)
    log_rate = pt.where(valid, pt.where(rate > 0.0, pt.log(pt.maximum(rate, 1e-300)), -np.inf), np.nan)

    pointwise = {
        key: value for key, value in solved.items() if key.startswith("theta_")
    }
    pointwise.update({
        "cooh_formation_affinity": affinity,
        "ssa_relative_residual_COOH": pt.where(inactive, 0.0, relative_residual),
        "coverage_solver_relative_residual": pt.where(inactive, 0.0, relative_residual),
    })
    electron_flux = rate  # Final COOH oxidation releases one electron.
    for pathway in pathways:
        flux = solved[f"coefficient_{pathway}"] * solved["theta_COOH"] * departure
        pointwise[f"net_rate_{pathway}"] = flux
        # BF: q from OH adsorption plus (1-q) from coupling; LH and ER: one from coupling/OH.
        electron_flux = electron_flux + flux
        fraction = pt.where(rate > 0.0, flux / pt.maximum(rate, 1e-300), 0.0)
        pointwise[f"net_pathway_fraction_{pathway}"] = fraction
        pointwise[f"rate_fraction_{pathway}"] = fraction
        pointwise[f"ln_rate_{pathway}"] = pt.where(flux > 0.0, pt.log(pt.maximum(flux, 1e-300)), -np.inf)
    pointwise["electron_flux"] = electron_flux
    return MechanismResult(ln_rate=log_rate, pointwise=pointwise)
