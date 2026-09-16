from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from mkm.mechanisms.base import MechanismResult
from .electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_tst_rate_constant,
)
from .state import AgPdPointState
from .surface import calculate_ag_qea_coverages, log_surface_fraction, logsumexp_pathways

@dataclass(frozen=True)
class PdCOSSACoverages:
    log_theta_empty_Pd: object
    log_theta_CO: object
    log_theta_OH_Pd: object


@dataclass(frozen=True)
class AgPdCOLHParameters:
    deltaG1_0: object
    deltaG4_0: object

    Gact1_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class AgPdCOERParameters:
    deltaG1_0: object
    deltaG4_0: object

    beta_2_ER: object

    Gact1_0: object
    Gact2_ER_0: object


@dataclass(frozen=True)
class AgPdCOBFParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2_BF: object
    q: object

    Gact1_0: object
    Gact2_BF_0: object


@dataclass(frozen=True)
class AgPdCOERLHParameters:
    deltaG1_0: object
    deltaG4_0: object

    beta_2_ER: object

    Gact1_0: object
    Gact2_ER_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class AgPdCOBFLHParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2_BF: object
    q: object

    Gact1_0: object
    Gact2_BF_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class AgPdCOBFERParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2_BF: object
    beta_2_ER: object
    q: object

    Gact1_0: object
    Gact2_BF_0: object
    Gact2_ER_0: object


@dataclass(frozen=True)
class AgPdCOBFERRLHParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2_BF: object
    beta_2_ER: object
    q: object

    Gact1_0: object
    Gact2_BF_0: object
    Gact2_ER_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class AgPdCOPathwayResult:
    mechanism_result: MechanismResult
    log_rate_total: object


@dataclass(frozen=True)
class AgPdCOBFERRLHResult:
    mechanism_result: MechanismResult

    log_rate_BF: object
    log_rate_ER: object
    log_rate_LH: object
    log_rate_total: object
    log_K1: object
    log_K4: object
    log_K5: object

    log_k1: object
    log_k_minus_1: object

    log_k2_BF: object
    log_k2_ER: object
    log_k2_LH: object


def solve_pd_co_ssa_qea_oh(
    log_K_OH_Pd,
    log_k1_a_CO,
    log_k_minus_1,
    log_k_BF_app=None,
    log_k_ER_app=None,
    log_k_LH_app=None,
    state: AgPdPointState | None = None,
    theta_CO_max=None,
):
    if state is None:
        raise ValueError("Pd CO SSA requires an AgPd point state.")

    log_K_OH_Pd = pt.as_tensor_variable(log_K_OH_Pd)
    term_OH_Pd = log_K_OH_Pd + pt.as_tensor_variable(state.ln_a_OH)
    point_template = pt.zeros_like(term_OH_Pd)

    log_k1_a_CO = pt.as_tensor_variable(log_k1_a_CO) + point_template
    log_k_minus_1 = pt.as_tensor_variable(log_k_minus_1) + point_template
    log_A_Pd = pt.logsumexp(pt.stack([point_template, term_OH_Pd]), axis=0)

    log_k_cons_terms = [log_k_minus_1]
    if log_k_BF_app is not None:
        log_k_cons_terms.append(pt.as_tensor_variable(log_k_BF_app) + point_template)
    if log_k_ER_app is not None:
        log_k_cons_terms.append(pt.as_tensor_variable(log_k_ER_app) + point_template)
    log_k_cons = pt.logsumexp(pt.stack(log_k_cons_terms), axis=0)

    log_rho = log_k1_a_CO - log_k_cons
    A_Pd = pt.exp(log_A_Pd)
    rho = pt.exp(log_rho)

    if log_k_LH_app is None:
        lambda_LH = pt.zeros_like(rho)
    else:
        log_k_LH_app = pt.as_tensor_variable(log_k_LH_app) + point_template
        lambda_LH = pt.exp(log_k_LH_app - log_k_cons)

    if theta_CO_max is None:
        if log_k_LH_app is None:
            # With no LH consumption, theta_CO = rho * theta_empty and
            # theta_CO + A_Pd * theta_empty = 1.
            log_denominator = pt.logsumexp(pt.stack([log_A_Pd, log_rho]), axis=0)
            log_theta_empty_Pd = -log_denominator
            log_theta_CO = log_rho - log_denominator
        else:
            quadratic_a = A_Pd * lambda_LH
            quadratic_b = rho + A_Pd - lambda_LH
            discriminant = quadratic_b**2 + 4.0 * quadratic_a
            sqrt_discriminant = pt.sqrt(discriminant)

            root_positive_b = 2.0 / (quadratic_b + sqrt_discriminant)
            root_negative_b = (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a)
            theta_empty_Pd = pt.where(quadratic_b >= 0, root_positive_b, root_negative_b)
            log_theta_empty_Pd = pt.log(theta_empty_Pd)
            log_theta_CO = log_rho + log_theta_empty_Pd - pt.log1p(lambda_LH * theta_empty_Pd)
    else:
        theta_CO_max = pt.as_tensor_variable(theta_CO_max)
        inv_theta_CO_max = 1.0 / theta_CO_max

        # r_ads = k1 a_CO theta_empty_Pd (1 - theta_CO / theta_CO_max).
        # With theta_CO + A_Pd theta_empty_Pd = 1, the CO SSA remains quadratic.
        quadratic_a = rho * inv_theta_CO_max + lambda_LH
        quadratic_b = rho * (1.0 + inv_theta_CO_max) + A_Pd + lambda_LH
        discriminant = (
            (A_Pd + lambda_LH) ** 2
            + 2.0
            * rho
            * (A_Pd * (1.0 + inv_theta_CO_max) + lambda_LH * (inv_theta_CO_max - 1.0))
            + rho**2 * (1.0 - inv_theta_CO_max) ** 2
        )
        sqrt_discriminant = pt.sqrt(discriminant)

        # Stable smaller root of a theta_CO^2 - b theta_CO + rho = 0.
        log_theta_CO = np.log(2.0) + log_rho - pt.log(quadratic_b + sqrt_discriminant)
        theta_CO = pt.exp(log_theta_CO)
        log_theta_empty_Pd = pt.log1p(-theta_CO) - log_A_Pd

    log_theta_OH_Pd = term_OH_Pd + log_theta_empty_Pd
    return PdCOSSACoverages(
        log_theta_empty_Pd=log_theta_empty_Pd,
        log_theta_CO=log_theta_CO,
        log_theta_OH_Pd=log_theta_OH_Pd,
    )


def _pathway_activity(activity, fraction, pathway_name):
    if activity is None:
        return np.ones_like(fraction, dtype=float)
    activity = np.asarray(activity, dtype=float)
    if activity.shape != fraction.shape:
        raise ValueError(f"{pathway_name} pathway activity must have the same shape as the model points.")
    if not np.all(np.isfinite(activity)) or np.any(activity < 0.0) or np.any(activity > 1.0):
        raise ValueError(f"{pathway_name} pathway activity must be finite and lie in [0, 1].")
    return activity


def _evaluate_agpd_co(
    state,
    parameters,
    temperature_K,
    pathways,
    *,
    theta_CO_max=None,
    bf_activity=None,
    er_activity=None,
):
    pathways = tuple(pathways)
    if not pathways or any(pathway not in {"BF", "ER", "LH"} for pathway in pathways):
        raise ValueError(f"Unsupported CO oxidation pathway set: {pathways}.")
    if np.any(state.Pd_fraction <= 0):
        raise ValueError("Finite-rate CO models require a positive Pd fraction.")

    E = pt.as_tensor_variable(state.E_V_SHE)
    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(
        delta_G_0_eV=parameters.deltaG4_0,
        electron_transfer=1.0,
        E_V_SHE=E,
    )
    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)

    Gact1 = pt.as_tensor_variable(parameters.Gact1_0)
    log_k1 = log_tst_rate_constant(activation_G_eV=Gact1, temperature_K=temperature_K)
    log_k_minus_1 = log_k1 - log_K1
    log_k1_a_CO = log_k1 + pt.as_tensor_variable(state.ln_a_CO)

    log_K5 = None
    log_k2_BF = None
    log_k2_ER = None
    log_k2_LH = None
    log_k_BF_app = None
    log_k_ER_app = None
    log_k_LH_app = None
    ag_coverages = None

    if "BF" in pathways:
        bf_activity = _pathway_activity(bf_activity, state.Ag_fraction, "BF")
        if np.all(state.Ag_fraction * bf_activity <= 0.0):
            raise ValueError(
                "BF-containing finite-rate CO models require an active BF pathway at positive Ag fraction."
            )
        deltaG5 = electrochemical_free_energy(
            delta_G_0_eV=parameters.deltaG5_0,
            electron_transfer=parameters.q,
            E_V_SHE=E,
        )
        Gact2_BF = electrochemical_activation_energy(
            activation_G_0_eV=parameters.Gact2_BF_0,
            beta=parameters.beta_2_BF,
            electron_transfer=1.0 - parameters.q,
            E_V_SHE=E,
        )
        log_K5 = log_equilibrium_constant(delta_G_eV=deltaG5, temperature_K=temperature_K)
        log_k2_BF = log_tst_rate_constant(activation_G_eV=Gact2_BF, temperature_K=temperature_K)
        ag_coverages = calculate_ag_qea_coverages(log_K_OH_Ag=log_K5, state=state)
        log_k_BF_app = (
            log_k2_BF
            + ag_coverages.log_theta_OH_Ag
            + log_surface_fraction(state.Ag_fraction)
            + log_surface_fraction(bf_activity)
        )

    if "ER" in pathways:
        er_activity = _pathway_activity(er_activity, state.Pd_fraction, "ER")
        Gact2_ER = electrochemical_activation_energy(
            activation_G_0_eV=parameters.Gact2_ER_0,
            beta=parameters.beta_2_ER,
            electron_transfer=1.0,
            E_V_SHE=E,
        )
        log_k2_ER = log_tst_rate_constant(activation_G_eV=Gact2_ER, temperature_K=temperature_K)
        log_k_ER_app = (
            log_k2_ER
            + pt.as_tensor_variable(state.ln_a_OH)
            + log_surface_fraction(er_activity)
        )

    if "LH" in pathways:
        Gact2_LH = pt.as_tensor_variable(parameters.Gact2_LH_0)
        log_k2_LH = log_tst_rate_constant(activation_G_eV=Gact2_LH, temperature_K=temperature_K)
        term_OH_Pd = log_K4 + pt.as_tensor_variable(state.ln_a_OH)
        log_k_LH_app = log_k2_LH + term_OH_Pd + log_surface_fraction(state.Pd_fraction)

    pd_coverages = solve_pd_co_ssa_qea_oh(
        log_K_OH_Pd=log_K4,
        log_k1_a_CO=log_k1_a_CO,
        log_k_minus_1=log_k_minus_1,
        log_k_BF_app=log_k_BF_app,
        log_k_ER_app=log_k_ER_app,
        log_k_LH_app=log_k_LH_app,
        state=state,
        theta_CO_max=theta_CO_max,
    )

    log_rates = {}
    if "BF" in pathways:
        log_rates["BF"] = log_k_BF_app + pd_coverages.log_theta_CO
    if "ER" in pathways:
        log_rates["ER"] = log_k_ER_app + pd_coverages.log_theta_CO
    if "LH" in pathways:
        log_rates["LH"] = log_k_LH_app + pd_coverages.log_theta_empty_Pd + pd_coverages.log_theta_CO

    log_rate_total = logsumexp_pathways(*(log_rates[pathway] for pathway in pathways))
    theta_CO = pt.exp(pd_coverages.log_theta_CO)
    pointwise = {
        "theta_CO": theta_CO,
        "theta_OH_Pd": pt.exp(pd_coverages.log_theta_OH_Pd),
        "theta_empty_Pd": pt.exp(pd_coverages.log_theta_empty_Pd),
    }
    if ag_coverages is not None:
        pointwise["theta_OH_Ag"] = pt.exp(ag_coverages.log_theta_OH_Ag)
        pointwise["theta_empty_Ag"] = pt.exp(ag_coverages.log_theta_empty_Ag)
    if theta_CO_max is not None:
        pointwise["theta_CO_site_occupation"] = theta_CO / pt.as_tensor_variable(theta_CO_max)

    for pathway in pathways:
        pointwise[f"ln_rate_{pathway}"] = log_rates[pathway]
        pointwise[f"rate_fraction_{pathway}"] = pt.exp(log_rates[pathway] - log_rate_total)

    return {
        "mechanism_result": MechanismResult(ln_rate=log_rate_total, pointwise=pointwise),
        "log_rates": log_rates,
        "log_rate_total": log_rate_total,
        "log_K1": log_K1,
        "log_K4": log_K4,
        "log_K5": log_K5,
        "log_k1": log_k1,
        "log_k_minus_1": log_k_minus_1,
        "log_k2_BF": log_k2_BF,
        "log_k2_ER": log_k2_ER,
        "log_k2_LH": log_k2_LH,
    }


def _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways):
    evaluated = _evaluate_agpd_co(state, parameters, temperature_K, pathways)
    return AgPdCOPathwayResult(
        mechanism_result=evaluated["mechanism_result"],
        log_rate_total=evaluated["log_rate_total"],
    )


def evaluate_agpd_co_lh(state: AgPdPointState, parameters: AgPdCOLHParameters, temperature_K):
    return _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways=("LH",))


def evaluate_agpd_co_er(state: AgPdPointState, parameters: AgPdCOERParameters, temperature_K):
    return _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways=("ER",))


def evaluate_agpd_co_bf(state: AgPdPointState, parameters: AgPdCOBFParameters, temperature_K):
    return _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways=("BF",))


def evaluate_agpd_co_er_lh(state: AgPdPointState, parameters: AgPdCOERLHParameters, temperature_K):
    return _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways=("ER", "LH"))


def evaluate_agpd_co_bf_lh(state: AgPdPointState, parameters: AgPdCOBFLHParameters, temperature_K):
    return _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways=("BF", "LH"))


def evaluate_agpd_co_bf_er(state: AgPdPointState, parameters: AgPdCOBFERParameters, temperature_K):
    return _evaluate_agpd_co_subset(state, parameters, temperature_K, pathways=("BF", "ER"))


def evaluate_agpd_co_bf_er_lh(
    state: AgPdPointState,
    parameters: AgPdCOBFERRLHParameters,
    temperature_K,
    theta_CO_max=None,
    bf_activity=None,
    er_activity=None,
):
    evaluated = _evaluate_agpd_co(
        state,
        parameters,
        temperature_K,
        pathways=("BF", "ER", "LH"),
        theta_CO_max=theta_CO_max,
        bf_activity=bf_activity,
        er_activity=er_activity,
    )
    log_rates = evaluated["log_rates"]
    return AgPdCOBFERRLHResult(
        mechanism_result=evaluated["mechanism_result"],
        log_rate_BF=log_rates["BF"],
        log_rate_ER=log_rates["ER"],
        log_rate_LH=log_rates["LH"],
        log_rate_total=evaluated["log_rate_total"],
        log_K1=evaluated["log_K1"],
        log_K4=evaluated["log_K4"],
        log_K5=evaluated["log_K5"],
        log_k1=evaluated["log_k1"],
        log_k_minus_1=evaluated["log_k_minus_1"],
        log_k2_BF=evaluated["log_k2_BF"],
        log_k2_ER=evaluated["log_k2_ER"],
        log_k2_LH=evaluated["log_k2_LH"],
    )


def _material_pathway_activity(state, *, material, active_value=0.0):
    if state.materials is None or state.material_index is None:
        raise ValueError("Material-specific pathway modifiers require material identity in the AgPd point state.")
    activity_by_material = np.asarray(
        [active_value if name == material else 1.0 for name in state.materials],
        dtype=float,
    )
    return activity_by_material[state.material_index]


def evaluate_agpd_co_bf_er_lh_ag10_no_bf(state, parameters, temperature_K):
    return evaluate_agpd_co_bf_er_lh(
        state=state,
        parameters=parameters,
        temperature_K=temperature_K,
        bf_activity=_material_pathway_activity(state, material="Ag10Pd90"),
    )


def evaluate_agpd_co_bf_er_lh_ag10_no_er(state, parameters, temperature_K):
    return evaluate_agpd_co_bf_er_lh(
        state=state,
        parameters=parameters,
        temperature_K=temperature_K,
        er_activity=_material_pathway_activity(state, material="Ag10Pd90"),
    )


def evaluate_agpd_co_bf_er_lh_capped(state, parameters, temperature_K):
    if state.theta_CO_max is None:
        raise ValueError("CO_BF_ER_LH_capped requires material-resolved CO coverage caps in the model configuration.")
    return evaluate_agpd_co_bf_er_lh(
        state=state,
        parameters=parameters,
        temperature_K=temperature_K,
        theta_CO_max=state.theta_CO_max,
    )


def evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf(state, parameters, temperature_K):
    if state.theta_CO_max is None:
        raise ValueError(
            "CO_BF_ER_LH_capped_Ag10_no_BF requires material-resolved CO coverage caps in the model configuration."
        )
    return evaluate_agpd_co_bf_er_lh(
        state=state,
        parameters=parameters,
        temperature_K=temperature_K,
        theta_CO_max=state.theta_CO_max,
        bf_activity=_material_pathway_activity(state, material="Ag10Pd90"),
    )
