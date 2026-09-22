"""Fully finite-rate AgPd CO oxidation with CO, COOH, OH* and OH# in SSA."""

from dataclasses import dataclass

import numpy as np
import pytensor
import pytensor.tensor as pt

from mkm.mechanisms.base import MechanismResult
from .electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_tst_rate_constant,
)
from .state import AgPdPointState


@dataclass(frozen=True)
class AgPdCOBFERRLH3OHParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object
    deltaG3_0: object
    q: object

    Gact1_0: object
    Gact4_0: object
    Gact5_0: object
    Gact2_BF_0: object
    Gact2_ER_0: object
    Gact2_LH_0: object
    Gact3_0: object

    beta_4: object
    beta_5: object
    beta_2_BF: object
    beta_2_ER: object
    beta_3: object


@dataclass(frozen=True)
class AgPdCOBFERRLH3OHResult:
    mechanism_result: MechanismResult
    deltaG_CO: object
    deltaG_OH_Pd: object
    deltaG_OH_Ag: object
    deltaG_COOH: object
    deltaG_BF: object
    deltaG_ER: object
    deltaG_LH: object
    deltaG_3: object
    electron_flux: object
    ssa_residual_CO: object
    ssa_residual_OH_Pd: object
    ssa_residual_OH_Ag: object
    ssa_residual_COOH: object


def _reverse_log_rate(log_k_forward, delta_G_eV, temperature_K):
    log_K = log_equilibrium_constant(delta_G_eV=delta_G_eV, temperature_K=temperature_K)
    return log_k_forward - log_K


def _safe_divide(numerator, denominator, floor=1e-30):
    denominator = pt.as_tensor_variable(denominator)
    fallback = pt.where(denominator >= 0.0, floor, -floor)
    safe = pt.where(pt.abs(denominator) >= floor, denominator, fallback)
    return pt.as_tensor_variable(numerator) / safe


def _full_ssa_state_from_affinity(
    affinity,
    *,
    k1_forward_a_CO,
    k1_reverse,
    k_OH_Pd_forward_a_OH,
    k_OH_Pd_reverse,
    k_OH_Ag_forward_a_OH,
    k_OH_Ag_reverse,
    k_BF_forward,
    k_BF_reverse,
    k_ER_forward,
    k_ER_reverse,
    k_LH_forward,
    k_LH_reverse,
    k3_forward_a_OH,
    a_OH,
    Ag_fraction,
    Pd_fraction,
):
    """Evaluate the analytically reduced SSA state at a trial common COOH-forming affinity."""
    affinity = pt.as_tensor_variable(affinity)
    driving = pt.expm1(affinity)

    # All three reversible COOH-forming routes connect the same thermodynamic states. Exact
    # detailed balance therefore gives them one common dimensionless affinity. Parameterizing
    # that small departure from equilibrium directly avoids subtracting nearly equal gross rates.
    K_ER = _safe_divide(k_ER_forward, k_ER_reverse)
    cooh_per_co = K_ER * a_OH * pt.exp(-affinity)

    # CO SSA with v_CO = v_3 and theta_COOH = cooh_per_co * theta_CO.
    theta_CO_per_empty = _safe_divide(
        k1_forward_a_CO,
        k1_reverse + k3_forward_a_OH * cooh_per_co,
    )
    theta_COOH_per_empty = cooh_per_co * theta_CO_per_empty

    # OH* SSA can be eliminated analytically together with the Pd site balance. The resulting
    # site-balance equation is quadratic in theta_empty_Pd.
    K_OH_Pd_a = _safe_divide(k_OH_Pd_forward_a_OH, k_OH_Pd_reverse)
    lh_loading = _safe_divide(
        Pd_fraction * k_LH_forward * theta_CO_per_empty,
        k_OH_Pd_reverse,
    )
    lh_flux_loading = _safe_divide(
        Pd_fraction * k_LH_reverse * theta_COOH_per_empty * driving,
        k_OH_Pd_reverse,
    )
    site_linear = 1.0 + theta_CO_per_empty + theta_COOH_per_empty + K_OH_Pd_a
    quad_a = site_linear * lh_loading - lh_flux_loading
    quad_b = site_linear - lh_loading
    discriminant = pt.maximum(quad_b**2 + 4.0 * quad_a, 1e-30)
    theta_empty_Pd = _safe_divide(2.0, quad_b + pt.sqrt(discriminant))

    theta_CO = theta_CO_per_empty * theta_empty_Pd
    theta_COOH = theta_COOH_per_empty * theta_empty_Pd
    common_departure = theta_COOH * driving

    # Stable net-pathway coefficients. With exact thermodynamic cycle closure, each pathway net
    # flux is C_j * common_departure; no subtraction of nearly equal forward/reverse gross rates
    # is required.
    coefficient_ER = k_ER_reverse
    coefficient_LH = _safe_divide(
        Pd_fraction * theta_empty_Pd * k_LH_reverse,
        1.0 + _safe_divide(Pd_fraction * k_LH_forward * theta_CO, k_OH_Pd_reverse),
    )
    ag_denominator = (
        k_OH_Ag_forward_a_OH
        + k_OH_Ag_reverse
        + k_BF_forward * theta_CO
        + k_BF_reverse * theta_COOH
    )
    coefficient_BF = _safe_divide(
        Ag_fraction * k_BF_reverse * k_OH_Ag_reverse,
        ag_denominator,
    )
    coefficient_total = coefficient_BF + coefficient_ER + coefficient_LH

    flux_BF = coefficient_BF * common_departure
    flux_ER = coefficient_ER * common_departure
    flux_LH = coefficient_LH * common_departure
    flux_3 = k3_forward_a_OH * theta_COOH

    # These expressions follow directly from the OH steady-state balances and avoid cancellation
    # of gross adsorption/desorption fluxes near equilibrium.
    theta_OH_Pd = _safe_divide(
        k_OH_Pd_forward_a_OH * theta_empty_Pd - flux_LH,
        k_OH_Pd_reverse,
    )
    theta_OH_Ag = _safe_divide(
        k_OH_Ag_forward_a_OH + k_BF_reverse * theta_COOH,
        ag_denominator,
    )
    theta_empty_Ag = 1.0 - theta_OH_Ag

    return {
        "theta_empty_Pd": theta_empty_Pd,
        "theta_CO": theta_CO,
        "theta_COOH": theta_COOH,
        "theta_OH_Pd": theta_OH_Pd,
        "theta_OH_Ag": theta_OH_Ag,
        "theta_empty_Ag": theta_empty_Ag,
        "coefficient_BF": coefficient_BF,
        "coefficient_ER": coefficient_ER,
        "coefficient_LH": coefficient_LH,
        "coefficient_total": coefficient_total,
        "flux_BF": flux_BF,
        "flux_ER": flux_ER,
        "flux_LH": flux_LH,
        "flux_3": flux_3,
    }


def _solver_settings(solver_config):
    iterations = int(solver_config.get("affinity_iterations", 8))
    damping = float(solver_config.get("affinity_damping", 1.0))
    if iterations < 1:
        raise ValueError("full_ssa affinity_iterations must be positive.")
    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("full_ssa affinity_damping must lie in (0, 1].")
    return iterations, damping


def _solve_full_ssa(
    *,
    k1_forward_a_CO,
    k1_reverse,
    k_OH_Pd_forward_a_OH,
    k_OH_Pd_reverse,
    k_OH_Ag_forward_a_OH,
    k_OH_Ag_reverse,
    k_BF_forward,
    k_BF_reverse,
    k_ER_forward,
    k_ER_reverse,
    k_LH_forward,
    k_LH_reverse,
    k3_forward_a_OH,
    a_OH,
    Ag_fraction,
    Pd_fraction,
    solver_config,
):
    iterations, damping = _solver_settings(solver_config)
    state_kwargs = dict(
        k1_forward_a_CO=k1_forward_a_CO,
        k1_reverse=k1_reverse,
        k_OH_Pd_forward_a_OH=k_OH_Pd_forward_a_OH,
        k_OH_Pd_reverse=k_OH_Pd_reverse,
        k_OH_Ag_forward_a_OH=k_OH_Ag_forward_a_OH,
        k_OH_Ag_reverse=k_OH_Ag_reverse,
        k_BF_forward=k_BF_forward,
        k_BF_reverse=k_BF_reverse,
        k_ER_forward=k_ER_forward,
        k_ER_reverse=k_ER_reverse,
        k_LH_forward=k_LH_forward,
        k_LH_reverse=k_LH_reverse,
        k3_forward_a_OH=k3_forward_a_OH,
        a_OH=a_OH,
        Ag_fraction=Ag_fraction,
        Pd_fraction=Pd_fraction,
    )

    # If ER were the only COOH-forming route, this would be the exact affinity. Because BF and LH
    # add positive conductance, it is a conservative starting value for the fixed-point iteration.
    affinity_initial = pt.log1p(_safe_divide(k3_forward_a_OH, k_ER_reverse))

    # Keep the repeated fixed-point map inside a single Scan Op. A Python for-loop duplicates the
    # complete SSA graph once per iteration; Scan keeps one differentiable copy of the iteration rule.
    state_keys = tuple(state_kwargs)
    non_sequences = [state_kwargs[key] for key in state_keys]

    def _affinity_step(affinity_previous, *state_values):
        iteration_kwargs = dict(zip(state_keys, state_values))
        iteration_state = _full_ssa_state_from_affinity(affinity_previous, **iteration_kwargs)
        target = pt.log1p(
            _safe_divide(iteration_kwargs["k3_forward_a_OH"], iteration_state["coefficient_total"])
        )
        return (1.0 - damping) * affinity_previous + damping * target

    affinity_sequence = pytensor.scan(
        fn=_affinity_step,
        outputs_info=affinity_initial,
        non_sequences=non_sequences,
        n_steps=iterations,
        name="agpd_full_ssa_affinity",
        strict=True,
        return_updates=False,
    )
    affinity = affinity_sequence[-1]

    state = _full_ssa_state_from_affinity(affinity, **state_kwargs)
    return affinity, state


def evaluate_agpd_co_bf_er_lh_3_oh(
    state: AgPdPointState,
    parameters: AgPdCOBFERRLH3OHParameters,
    temperature_K,
    *,
    full_ssa_config=None,
):
    if np.any(state.Pd_fraction <= 0.0):
        raise ValueError("CO_BF_ER_LH_3_OH requires a positive Pd fraction.")
    if full_ssa_config is None:
        full_ssa_config = {}

    E = pt.as_tensor_variable(state.E_V_SHE)
    ln_a_CO = pt.as_tensor_variable(state.ln_a_CO)
    ln_a_OH = pt.as_tensor_variable(state.ln_a_OH)
    a_CO = pt.exp(ln_a_CO)
    a_OH = pt.exp(ln_a_OH)
    Ag_fraction = pt.as_tensor_variable(state.Ag_fraction)
    Pd_fraction = pt.as_tensor_variable(state.Pd_fraction)

    deltaG_CO = pt.as_tensor_variable(parameters.deltaG1_0) + pt.zeros_like(E)
    deltaG_OH_Pd = electrochemical_free_energy(parameters.deltaG4_0, 1.0, E)
    deltaG_OH_Ag = electrochemical_free_energy(parameters.deltaG5_0, parameters.q, E)
    deltaG_COOH = electrochemical_free_energy(parameters.deltaG3_0, 1.0, E)
    deltaG_BF = deltaG_COOH - deltaG_CO - deltaG_OH_Ag
    deltaG_ER = deltaG_COOH - deltaG_CO
    deltaG_LH = deltaG_COOH - deltaG_CO - deltaG_OH_Pd
    overall_deltaG_0 = float(full_ssa_config.get("overall_reaction_deltaG_0_eV", -1.86))
    deltaG_overall = electrochemical_free_energy(overall_deltaG_0, 2.0, E)
    deltaG_3 = deltaG_overall - deltaG_COOH

    Gact1 = pt.as_tensor_variable(parameters.Gact1_0) + pt.zeros_like(E)
    Gact_OH_Pd = electrochemical_activation_energy(
        parameters.Gact4_0, parameters.beta_4, 1.0, E
    )
    Gact_OH_Ag = electrochemical_activation_energy(
        parameters.Gact5_0, parameters.beta_5, parameters.q, E
    )
    Gact_BF = electrochemical_activation_energy(
        parameters.Gact2_BF_0, parameters.beta_2_BF, 1.0 - parameters.q, E
    )
    Gact_ER = electrochemical_activation_energy(parameters.Gact2_ER_0, parameters.beta_2_ER, 1.0, E)
    Gact_LH = pt.as_tensor_variable(parameters.Gact2_LH_0) + pt.zeros_like(E)
    Gact_3 = electrochemical_activation_energy(parameters.Gact3_0, parameters.beta_3, 1.0, E)

    log_k1 = log_tst_rate_constant(Gact1, temperature_K)
    log_k1_reverse = _reverse_log_rate(log_k1, deltaG_CO, temperature_K)
    log_k_OH_Pd = log_tst_rate_constant(Gact_OH_Pd, temperature_K)
    log_k_OH_Pd_reverse = _reverse_log_rate(log_k_OH_Pd, deltaG_OH_Pd, temperature_K)
    log_k_OH_Ag = log_tst_rate_constant(Gact_OH_Ag, temperature_K)
    log_k_OH_Ag_reverse = _reverse_log_rate(log_k_OH_Ag, deltaG_OH_Ag, temperature_K)
    log_k_BF = log_tst_rate_constant(Gact_BF, temperature_K)
    log_k_BF_reverse = _reverse_log_rate(log_k_BF, deltaG_BF, temperature_K)
    log_k_ER = log_tst_rate_constant(Gact_ER, temperature_K)
    log_k_ER_reverse = _reverse_log_rate(log_k_ER, deltaG_ER, temperature_K)
    log_k_LH = log_tst_rate_constant(Gact_LH, temperature_K)
    log_k_LH_reverse = _reverse_log_rate(log_k_LH, deltaG_LH, temperature_K)
    log_k3 = log_tst_rate_constant(Gact_3, temperature_K)

    k1 = pt.exp(log_k1)
    k1_reverse = pt.exp(log_k1_reverse)
    k_OH_Pd = pt.exp(log_k_OH_Pd)
    k_OH_Pd_reverse = pt.exp(log_k_OH_Pd_reverse)
    k_OH_Ag = pt.exp(log_k_OH_Ag)
    k_OH_Ag_reverse = pt.exp(log_k_OH_Ag_reverse)
    k_BF = pt.exp(log_k_BF)
    k_BF_reverse = pt.exp(log_k_BF_reverse)
    k_ER = pt.exp(log_k_ER)
    k_ER_reverse = pt.exp(log_k_ER_reverse)
    k_LH = pt.exp(log_k_LH)
    k_LH_reverse = pt.exp(log_k_LH_reverse)
    k3 = pt.exp(log_k3)

    k1_a_CO = k1 * a_CO
    k_OH_Pd_a_OH = k_OH_Pd * a_OH
    k_OH_Ag_a_OH = k_OH_Ag * a_OH
    k_ER_a_OH = k_ER * a_OH
    k3_a_OH = k3 * a_OH

    affinity, solved = _solve_full_ssa(
        k1_forward_a_CO=k1_a_CO,
        k1_reverse=k1_reverse,
        k_OH_Pd_forward_a_OH=k_OH_Pd_a_OH,
        k_OH_Pd_reverse=k_OH_Pd_reverse,
        k_OH_Ag_forward_a_OH=k_OH_Ag_a_OH,
        k_OH_Ag_reverse=k_OH_Ag_reverse,
        k_BF_forward=k_BF,
        k_BF_reverse=k_BF_reverse,
        k_ER_forward=k_ER,
        k_ER_reverse=k_ER_reverse,
        k_LH_forward=k_LH,
        k_LH_reverse=k_LH_reverse,
        k3_forward_a_OH=k3_a_OH,
        a_OH=a_OH,
        Ag_fraction=Ag_fraction,
        Pd_fraction=Pd_fraction,
        solver_config=full_ssa_config,
    )
    theta_empty_Pd = solved["theta_empty_Pd"]
    theta_CO = solved["theta_CO"]
    theta_COOH = solved["theta_COOH"]
    theta_OH_Pd = solved["theta_OH_Pd"]
    theta_OH_Ag_raw = solved["theta_OH_Ag"]
    theta_empty_Ag_raw = solved["theta_empty_Ag"]
    theta_OH_Ag = pt.where(Ag_fraction > 0.0, theta_OH_Ag_raw, 0.0)
    theta_empty_Ag = pt.where(Ag_fraction > 0.0, theta_empty_Ag_raw, 1.0)

    flux_BF = solved["flux_BF"]
    flux_ER = solved["flux_ER"]
    flux_LH = solved["flux_LH"]
    flux_3 = solved["flux_3"]

    # CO, OH* and OH# SSAs were eliminated analytically. Their net fluxes are therefore the
    # corresponding steady-state fluxes by construction; only the final COOH balance is solved
    # iteratively through the common affinity.
    flux_CO = flux_3
    flux_OH_Pd = flux_LH
    flux_OH_Ag = flux_BF
    residual_COOH = flux_BF + flux_ER + flux_LH - flux_3
    residual_CO = -residual_COOH
    residual_OH_Pd = pt.zeros_like(residual_COOH)
    residual_OH_Ag = pt.zeros_like(residual_COOH)

    electron_flux = (
        flux_OH_Pd
        + parameters.q * flux_OH_Ag
        + (1.0 - parameters.q) * flux_BF
        + flux_ER
        + flux_3
    )
    log_rate_total = log_k3 + ln_a_OH + pt.log(pt.maximum(theta_COOH, 1e-300))
    net_fraction_BF = _safe_divide(flux_BF, flux_3)
    net_fraction_ER = _safe_divide(flux_ER, flux_3)
    net_fraction_LH = _safe_divide(flux_LH, flux_3)
    ssa_relative_residual = _safe_divide(residual_COOH, flux_3)

    pointwise = {
        "theta_CO": theta_CO,
        "theta_COOH": theta_COOH,
        "theta_OH_Pd": theta_OH_Pd,
        "theta_empty_Pd": theta_empty_Pd,
        "theta_OH_Ag": theta_OH_Ag,
        "theta_empty_Ag": theta_empty_Ag,
        "net_pathway_fraction_BF": net_fraction_BF,
        "net_pathway_fraction_ER": net_fraction_ER,
        "net_pathway_fraction_LH": net_fraction_LH,
        "ssa_relative_residual_COOH": ssa_relative_residual,
        "cooh_formation_affinity": affinity,
    }
    return AgPdCOBFERRLH3OHResult(
        mechanism_result=MechanismResult(ln_rate=log_rate_total, pointwise=pointwise),
        deltaG_CO=deltaG_CO,
        deltaG_OH_Pd=deltaG_OH_Pd,
        deltaG_OH_Ag=deltaG_OH_Ag,
        deltaG_COOH=deltaG_COOH,
        deltaG_BF=deltaG_BF,
        deltaG_ER=deltaG_ER,
        deltaG_LH=deltaG_LH,
        deltaG_3=deltaG_3,
        electron_flux=electron_flux,
        ssa_residual_CO=residual_CO,
        ssa_residual_OH_Pd=residual_OH_Pd,
        ssa_residual_OH_Ag=residual_OH_Ag,
        ssa_residual_COOH=residual_COOH,
    )
