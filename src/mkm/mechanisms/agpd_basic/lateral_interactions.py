"""Thermodynamically reciprocal lateral interactions for AgPd CO oxidation.

Three pair energies define an excess surface free energy (per total surface site).
The secondary OH population enters the interacting OH density with PCT weight q.
Both QEA and finite-rate steps are solved from the same steady-state balances.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.scan.utils import until

from mkm.mechanisms.base import MechanismResult
from mkm.mechanisms.electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_tst_rate_constant,
)


INTERACTION_PARAMETERS = {"c": "z_CO_CO_eV", "x": "z_CO_OH_eV", "o": "z_OH_OH_eV"}


def _excess_chemical_potentials(c, p, s, fp, fs, q, zc, zx, zo):
    """Derivatives of W = zc*C^2/2 + zx*C*H + zo*H^2/2.

    C = fp*c, H = fp*p + q*fs*s. This guarantees reciprocal CO/OH cross terms
    with correct site-population and partial-charge-transfer weighting.
    """
    C = fp * c
    H = fp * p + q * fs * s
    mu_co = zc * C + zx * H
    mu_p = zx * C + zo * H
    mu_s = q * mu_p
    return mu_co, mu_p, mu_s


def evaluate_agpd_lateral_interactions(state, parameters, temperature_K, *, pathways, finite_steps,
                                       interactions, lateral_config=None):
    """Solve Frumkin-coupled adsorption/oxidation with differentiable fixed-point iterations.

    The configured iteration count is a maximum when adaptive convergence is enabled.
    Both initial coverage branches must satisfy the actual balance tolerance before stopping;
    non-converged or distinct branches are rejected, not silently used in the likelihood.
    """
    config = {} if lateral_config is None else lateral_config
    iterations = int(config.get("fixed_point_iterations", 96))
    adaptive = bool(config.get("adaptive_convergence", False))
    damping = float(config.get("fixed_point_damping", 0.5))
    tolerance = float(config.get("relative_residual_tolerance", 1e-7))
    branch_tolerance = float(config.get("branch_tolerance", 1e-5))
    gamma = float(config.get("bep_coefficient", 0.65))
    if iterations < 1 or not 0.0 < damping <= 1.0 or not np.isfinite(damping):
        raise ValueError("Lateral solver iterations must be positive and damping must lie in (0, 1].")
    if not all(np.isfinite(x) and x > 0.0 for x in (tolerance, branch_tolerance)):
        raise ValueError("Lateral solver tolerances must be positive and finite.")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("The BEP coefficient must be finite and lie in [0, 1].")
    if np.any(state.Pd_fraction <= 0.0):
        raise ValueError("Lateral interaction models require positive primary-site populations.")
    pathways, finite_steps, interactions = tuple(pathways), tuple(finite_steps), tuple(interactions)
    if not pathways or set(pathways) - {"BF", "ER", "LH"} or not interactions:
        raise ValueError("Lateral solver requires at least one pathway and one interaction.")

    E = pt.as_tensor_variable(state.E_V_SHE)
    zeros = pt.zeros_like(E)
    neg_inf = zeros - np.inf
    ln_a_co, ln_a_oh = pt.as_tensor_variable(state.ln_a_CO), pt.as_tensor_variable(state.ln_a_OH)
    fp, fs = pt.as_tensor_variable(state.Pd_fraction), pt.as_tensor_variable(state.Ag_fraction)
    log_fp = pt.where(fp > 0.0, pt.log(pt.maximum(fp, 1e-300)), neg_inf)
    log_fs = pt.where(fs > 0.0, pt.log(pt.maximum(fs, 1e-300)), neg_inf)
    log_fp_secondary = pt.where(fs > 0.0, log_fp, neg_inf)
    has_secondary_oh = "BF" in pathways or "sOH" in finite_steps or bool(set(interactions) & {"x", "o"})
    q = pt.as_tensor_variable(parameters.q) if has_secondary_oh else 0.0
    zc = pt.as_tensor_variable(parameters.z_CO_CO_eV) if "c" in interactions else 0.0
    zx = pt.as_tensor_variable(parameters.z_CO_OH_eV) if "x" in interactions else 0.0
    zo = pt.as_tensor_variable(parameters.z_OH_OH_eV) if "o" in interactions else 0.0

    dg1_0 = pt.as_tensor_variable(parameters.deltaG1_0) + zeros
    dg4_0 = electrochemical_free_energy(parameters.deltaG4_0, 1.0, E)
    dg5_0 = electrochemical_free_energy(parameters.deltaG5_0, q, E) if has_secondary_oh else zeros
    dg3_0 = electrochemical_free_energy(parameters.deltaG3_0, 1.0, E) if "3" in finite_steps else zeros

    g1_0 = pt.as_tensor_variable(parameters.Gact1_0) + zeros if "CO" in finite_steps else zeros
    g4_0 = (electrochemical_activation_energy(parameters.Gact4_0, parameters.beta_4, 1.0, E)
            if "pOH" in finite_steps else zeros)
    g5_0 = (electrochemical_activation_energy(parameters.Gact5_0, parameters.beta_5, q, E)
            if "sOH" in finite_steps else zeros)
    gbf_0 = (electrochemical_activation_energy(parameters.Gact2_BF_0, parameters.beta_2_BF, 1.0-q, E)
             if "BF" in pathways else zeros)
    ger_0 = (electrochemical_activation_energy(parameters.Gact2_ER_0, parameters.beta_2_ER, 1.0, E)
             if "ER" in pathways else zeros)
    glh_0 = pt.as_tensor_variable(parameters.Gact2_LH_0) + zeros if "LH" in pathways else zeros
    g3_0 = (electrochemical_activation_energy(parameters.Gact3_0, parameters.beta_3, 1.0, E)
            if "3" in finite_steps else zeros)
    logk3a = log_tst_rate_constant(g3_0, temperature_K) + ln_a_oh if "3" in finite_steps else neg_inf

    # Explicit Scan inputs keep PyMC prior random variables and their RNGs outside the loop.
    scan_constants = tuple(pt.as_tensor_variable(value) for value in (
        neg_inf, fp, fs, log_fp, log_fs, log_fp_secondary, q, zc, zx, zo,
        dg1_0, dg4_0, dg5_0, dg3_0, g1_0, g4_0, g5_0, gbf_0, ger_0, glh_0,
        logk3a, ln_a_co, ln_a_oh,
    ))

    scan_unique = []
    scan_indices = []
    for value in scan_constants:
        index = next((i for i, existing in enumerate(scan_unique) if existing is value), None)
        if index is None:
            index = len(scan_unique)
            scan_unique.append(value)
        scan_indices.append(index)

    def _constants(c, p, s, constants):
        (neg_inf, fp, fs, log_fp, log_fs, log_fp_secondary, q, zc, zx, zo,
         dg1_0, dg4_0, dg5_0, dg3_0, g1_0, g4_0, g5_0, gbf_0, ger_0, glh_0,
         logk3a, ln_a_co, ln_a_oh) = constants
        neg = neg_inf + pt.zeros_like(c)
        mu_c, mu_p, mu_s = _excess_chemical_potentials(c, p, s, fp, fs, q, zc, zx, zo)
        dg1, dg4, dg5 = dg1_0 + mu_c, dg4_0 + mu_p, dg5_0 + mu_s
        logK1a = log_equilibrium_constant(dg1, temperature_K) + ln_a_co
        logK4a = log_equilibrium_constant(dg4, temperature_K) + ln_a_oh
        logK5a = log_equilibrium_constant(dg5, temperature_K) + ln_a_oh

        if "CO" in finite_steps:
            lk1a = log_tst_rate_constant(g1_0 + gamma * mu_c, temperature_K) + ln_a_co
            lk1b = lk1a - logK1a
        else:
            lk1a = lk1b = neg
        if "pOH" in finite_steps:
            lk4a = log_tst_rate_constant(g4_0 + gamma * mu_p, temperature_K) + ln_a_oh
            lk4b = lk4a - logK4a
        else:
            lk4a = lk4b = neg
        if "sOH" in finite_steps:
            lk5a = log_tst_rate_constant(g5_0 + gamma * mu_s, temperature_K) + ln_a_oh
            lk5b = lk5a - logK5a
        else:
            lk5a = lk5b = neg
        if "BF" in pathways:
            lbf = log_tst_rate_constant(gbf_0 - gamma * (mu_c + mu_s), temperature_K)
            lbf_rev = (lbf - log_equilibrium_constant(dg3_0 - dg1 - dg5, temperature_K)
                       if "3" in finite_steps else neg)
        else:
            lbf = lbf_rev = neg
        if "ER" in pathways:
            ler = log_tst_rate_constant(ger_0 - gamma * mu_c, temperature_K) + ln_a_oh
            ler_rev = (ler - log_equilibrium_constant(dg3_0 - dg1, temperature_K) - ln_a_oh
                       if "3" in finite_steps else neg)
        else:
            ler = ler_rev = neg
        if "LH" in pathways:
            llh = log_tst_rate_constant(glh_0 - gamma * (mu_c + mu_p), temperature_K)
            llh_rev = (llh - log_equilibrium_constant(dg3_0 - dg1 - dg4, temperature_K)
                       if "3" in finite_steps else neg)
        else:
            llh = llh_rev = neg
        return (logK1a, logK4a, logK5a, lk1a, lk1b, lk4a, lk4b, lk5a, lk5b,
                lbf, lbf_rev, ler, ler_rev, llh, llh_rev, mu_c, mu_p, mu_s)

    def _log(x, neg_inf):
        return pt.where(x > 0.0, pt.log(pt.maximum(x, 1e-300)), neg_inf)

    def _relative_balance(y, constants):
        """Maximum normalized reaction/QEA balance error for each branch and model point."""
        c, p, h, s, e = (y[:, i] for i in range(5))
        (neg_inf, fp, fs, log_fp, log_fs, log_fp_secondary, q, zc, zx, zo,
         dg1_0, dg4_0, dg5_0, dg3_0, g1_0, g4_0, g5_0, gbf_0, ger_0, glh_0,
         logk3a, ln_a_co, ln_a_oh) = constants
        (lco, lp, ls, lk1a, lk1b, lk4a, lk4b, lk5a, lk5b,
         lbf, lbfr, ler, lerr, llh, llhr, *_) = _constants(c, p, s, constants)
        rbf = pt.exp(log_fs+lbf)*c*s - pt.exp(log_fs+lbfr)*h*(1.0-s)
        rer = pt.exp(ler)*c - pt.exp(lerr)*h
        rlh = pt.exp(log_fp+llh)*c*p - pt.exp(log_fp+llhr)*h*e
        total = rbf + rer + rlh
        rate = pt.exp(logk3a)*h if "3" in finite_steps else total
        if "CO" in finite_steps:
            forward, backward = pt.exp(lk1a)*e, pt.exp(lk1b)*c
            rco = pt.abs(forward-backward-total) / pt.maximum(forward+backward+pt.abs(total), 1e-300)
        else:
            rco = pt.abs(pt.log(pt.maximum(c, 1e-300)/pt.maximum(e, 1e-300))-lco)
        if "pOH" in finite_steps:
            forward, backward = pt.exp(lk4a)*e, pt.exp(lk4b)*p
            rpo = pt.abs(forward-backward-rlh) / pt.maximum(forward+backward+pt.abs(rlh), 1e-300)
        else:
            rpo = pt.abs(pt.log(pt.maximum(p, 1e-300)/pt.maximum(e, 1e-300))-lp)
        if not has_secondary_oh:
            rso = pt.zeros_like(c)
        elif "sOH" in finite_steps:
            forward, backward = pt.exp(lk5a)*(1.0-s), pt.exp(lk5b)*s
            bf_on_secondary = fp*rbf/pt.maximum(fs, 1e-300)
            rso = (pt.abs(forward-backward-bf_on_secondary) /
                   pt.maximum(forward+backward+pt.abs(bf_on_secondary), 1e-300))
        else:
            rso = pt.abs(pt.log(pt.maximum(s, 1e-300)/pt.maximum(1.0-s, 1e-300))-ls)
        rh = pt.abs(total-rate)/pt.maximum(pt.abs(total)+rate, 1e-300) if "3" in finite_steps else pt.zeros_like(c)
        residual = pt.maximum(pt.maximum(rco, rpo), pt.maximum(rso, rh))
        inactive = fs <= 0.0 if pathways == ("BF",) else pt.zeros_like(fs, dtype="bool")
        return pt.where(inactive, 0.0, residual)

    def _trial(y, *unique_values):
        constants = tuple(unique_values[i] for i in scan_indices)
        (neg_inf, fp, fs, log_fp, log_fs, log_fp_secondary, q, zc, zx, zo,
         dg1_0, dg4_0, dg5_0, dg3_0, g1_0, g4_0, g5_0, gbf_0, ger_0, glh_0,
         logk3a, ln_a_co, ln_a_oh) = constants
        c, p, h, s, e = (y[:, i] for i in range(5))
        # Carry the vacant-site fraction independently: 1-c-p-h loses precision near CO saturation.
        (lco, lp, ls, lk1a, lk1b, lk4a, lk4b, lk5a, lk5b,
         lbf, lbfr, ler, lerr, llh, llhr, *_) = _constants(c, p, s, constants)
        lc, lp_cov, lh, le, ls_cov, ls_empty = (_log(x, neg_inf) for x in (c, p, h, e, s, 1.0-s))
        if "CO" in finite_steps:
            log_con = pt.logsumexp(pt.stack([lk1b, log_fs+lbf+ls_cov, ler, log_fp+llh+lp_cov]), axis=0)
            log_rev = pt.logsumexp(pt.stack([log_fs+lbfr+ls_empty, lerr, log_fp+llhr+le]), axis=0)
            ratio_co = pt.logaddexp(lk1a, lh-le+log_rev) - log_con
        else:
            ratio_co = lco
        if "pOH" in finite_steps:
            ratio_p = (pt.logaddexp(lk4a, log_fp+llhr+lh)
                       - pt.logaddexp(lk4b, log_fp+llh+lc))
        else:
            ratio_p = lp
        if "sOH" in finite_steps:
            ratio_s = (pt.logaddexp(lk5a, log_fp_secondary+lbfr+lh)
                       - pt.logaddexp(lk5b, log_fp_secondary+lbf+lc))
        else:
            ratio_s = ls if has_secondary_oh else pt.zeros_like(lco) - np.inf
        if "3" in finite_steps:
            log_form = pt.logsumexp(pt.stack([log_fs+lbf+ls_cov, ler, log_fp+llh+lp_cov]), axis=0)
            log_remove = pt.logsumexp(
                pt.stack([logk3a+pt.zeros_like(c), log_fs+lbfr+ls_empty, lerr, log_fp+llhr+le]), axis=0
            )
            ratio_h = ratio_co + log_form - log_remove
        else:
            ratio_h = pt.zeros_like(ratio_co) - np.inf
        logZ = pt.logsumexp(pt.stack([pt.zeros_like(ratio_co), ratio_co, ratio_p, ratio_h]), axis=0)
        c_next, p_next, h_next = pt.exp(ratio_co-logZ), pt.exp(ratio_p-logZ), pt.exp(ratio_h-logZ)
        s_next = pt.sigmoid(ratio_s)
        e_next = pt.exp(-logZ)
        result = pt.stack([c_next, p_next, h_next, s_next, e_next], axis=1)
        return (1.0-damping)*y + damping*result

    shape = pt.ones_like(E)
    start_co = pt.stack([0.88*shape, 0.05*shape, 0.01*shape, 0.10*shape], axis=0)
    start_oh = pt.stack([0.05*shape, 0.88*shape, 0.01*shape, 0.90*shape], axis=0)
    if "3" not in finite_steps:
        start_co = pt.set_subtensor(start_co[2], zeros)
        start_oh = pt.set_subtensor(start_oh[2], zeros)
    # Vacancy is also normalized by logZ at every iteration, avoiding subtractive cancellation.
    start_co = pt.concatenate([start_co, (1.0-pt.sum(start_co[:3], axis=0))[None, :]], axis=0)
    start_oh = pt.concatenate([start_oh, (1.0-pt.sum(start_oh[:3], axis=0))[None, :]], axis=0)
    initial = pt.stack([start_co, start_oh], axis=0)
    if adaptive:
        # Adaptive scan stops only when the actual kinetic/QEA residuals pass for BOTH starts.
        # A small fixed-point movement alone can hide a large kinetic residual in stiff regimes.
        def _adaptive_trial(y, *unique_values):
            next_y = _trial(y, *unique_values)
            constants = tuple(unique_values[i] for i in scan_indices)
            residuals = _relative_balance(next_y, constants)
            valid_residuals = pt.and_(~pt.isnan(residuals), ~pt.isinf(residuals))
            return next_y, until(pt.all(valid_residuals & (residuals <= tolerance)))

        step = _adaptive_trial
    else:
        step = _trial
    sequence = pytensor.scan(fn=step, outputs_info=initial, n_steps=iterations,
                             non_sequences=scan_unique, strict=True, return_updates=False,
                             name="agpd_frumkin_coverages")
    both = sequence[-1]
    branch_difference = pt.max(pt.abs(both[0] - both[1]), axis=0)
    c, p, h, s, e = (both[0, i] for i in range(5))
    (lco, lp, ls, lk1a, lk1b, lk4a, lk4b, lk5a, lk5b,
     lbf, lbfr, ler, lerr, llh, llhr, mc, mp, ms) = _constants(c, p, s, scan_constants)
    bf_f = pt.exp(log_fs+lbf)*c*s
    bf_b = pt.exp(log_fs+lbfr)*h*(1.0-s)
    er_f = pt.exp(ler)*c
    er_b = pt.exp(lerr)*h
    lh_f = pt.exp(log_fp+llh)*c*p
    lh_b = pt.exp(log_fp+llhr)*h*e
    rbf, rer, rlh = bf_f-bf_b, er_f-er_b, lh_f-lh_b
    total = rbf+rer+rlh
    rate = pt.exp(logk3a)*h if "3" in finite_steps else total
    residual = _relative_balance(both, scan_constants)[0]
    diff = branch_difference
    # BF-only chemistry has no productive coupling route when secondary sites are absent.
    inactive = fs <= 0.0 if pathways == ("BF",) else pt.zeros_like(fs, dtype="bool")
    residual = pt.where(inactive, 0.0, residual)
    diff = pt.where(inactive, 0.0, diff)
    h = pt.where(inactive, 0.0, h)
    finite_residual = pt.and_(~pt.isnan(residual), ~pt.isinf(residual))
    valid = pt.and_(pt.and_(finite_residual, residual <= tolerance), diff <= branch_tolerance)
    rate = pt.where(inactive, 0.0, rate)
    log_rate = pt.where(valid, pt.where(rate > 0.0, pt.log(pt.maximum(rate, 1e-300)), -np.inf), np.nan)
    pointwise = {
        "theta_CO": c, "theta_OH_Pd": p, "theta_empty_Pd": e, "theta_COOH": h,
        "theta_OH_Ag": pt.where(fs > 0.0, s, 0.0),
        "theta_empty_Ag": pt.where(fs > 0.0, 1.0-s, 1.0),
        "lateral_mu_CO_eV": mc, "lateral_mu_pOH_eV": mp, "lateral_mu_sOH_eV": ms,
        "coverage_solver_relative_residual": residual,
        "coverage_branch_disagreement": diff,
        "coverage_solver_iterations": pt.zeros_like(E) + sequence.shape[0],
        "electron_flux": 2.0*rate,
    }
    if config.get("diagnose_branches", False):
        # Diagnostic only: how much would one additional fixed-point iteration move each branch?
        # Small update errors support stationarity, but do not establish dynamical stability.
        updated = _trial(both, *scan_unique)
        update_errors = pt.max(pt.abs(updated - both), axis=1)
        pointwise["branch0_balance_residual"] = residual
        pointwise["branch1_balance_residual"] = _relative_balance(both, scan_constants)[1]
        for index in (0, 1):
            branch = both[index]
            pointwise[f"branch{index}_update_error"] = update_errors[index]
            for label, coverage in zip(("CO", "pOH", "COOH", "sOH"), (branch[i] for i in range(4))):
                pointwise[f"branch{index}_theta_{label}"] = coverage
            pointwise[f"branch{index}_theta_empty_Pd"] = branch[4]
    for pathway, flux in (("BF", rbf), ("ER", rer), ("LH", rlh)):
        if pathway in pathways:
            pointwise[f"net_rate_{pathway}"] = flux
            pointwise[f"ln_rate_{pathway}"] = pt.where(flux > 0.0, pt.log(pt.maximum(flux,1e-300)), -np.inf)
            pointwise[f"rate_fraction_{pathway}"] = pt.where(rate > 0.0, flux/pt.maximum(rate,1e-300), 0.0)
    return MechanismResult(ln_rate=log_rate, pointwise=pointwise)
