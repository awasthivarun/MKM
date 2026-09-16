from dataclasses import dataclass

import pytensor.tensor as pt

from mkm.mechanisms.base import MechanismResult
from .electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_tst_rate_constant,
)
from .state import AgPdPointState
from .surface import calculate_ag_qea_coverages, calculate_pd_qea_coverages, log_surface_fraction, logsumexp_pathways

@dataclass(frozen=True)
class AgPdERParameters:
    deltaG1_0: object
    deltaG4_0: object

    beta_2_ER: object
    Gact2_ER_0: object


@dataclass(frozen=True)
class AgPdERResult:
    mechanism_result: MechanismResult

    log_rate_ER: object

    log_K1: object
    log_K4: object
    log_k2_ER: object


@dataclass(frozen=True)
class AgPdBFERRLHParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2_BF: object
    beta_2_ER: object
    q: object

    Gact2_BF_0: object
    Gact2_ER_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class AgPdBFERRLHResult:
    mechanism_result: MechanismResult
    log_rate_BF: object
    log_rate_ER: object
    log_rate_LH: object
    log_rate_total: object

    log_K1: object
    log_K4: object
    log_K5: object

    log_k2_BF: object
    log_k2_ER: object
    log_k2_LH: object


def calculate_log_rate_bf(log_k2_BF, log_theta_CO, log_theta_OH_Ag, state: AgPdPointState):
    return (
        pt.as_tensor_variable(log_k2_BF)
        + pt.as_tensor_variable(log_theta_CO)
        + pt.as_tensor_variable(log_theta_OH_Ag)
        + log_surface_fraction(state.Ag_fraction)
    )


def calculate_log_rate_er(log_k2_ER, log_theta_CO, state: AgPdPointState):
    return pt.as_tensor_variable(log_k2_ER) + pt.as_tensor_variable(log_theta_CO) + pt.as_tensor_variable(state.ln_a_OH)


def calculate_log_rate_lh(log_k2_LH, log_theta_CO, log_theta_OH_Pd, state: AgPdPointState):
    return (
        pt.as_tensor_variable(log_k2_LH)
        + pt.as_tensor_variable(log_theta_CO)
        + pt.as_tensor_variable(log_theta_OH_Pd)
        + log_surface_fraction(state.Pd_fraction)
    )


def evaluate_agpd_er(state: AgPdPointState, parameters: AgPdERParameters, temperature_K):
    E = pt.as_tensor_variable(state.E_V_SHE)
    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(delta_G_0_eV=parameters.deltaG4_0, electron_transfer=1.0, E_V_SHE=E)
    Gact2_ER = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_ER_0,
        beta=parameters.beta_2_ER,
        electron_transfer=1.0,
        E_V_SHE=E,
    )

    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)
    log_k2_ER = log_tst_rate_constant(activation_G_eV=Gact2_ER, temperature_K=temperature_K)
    pd_coverages = calculate_pd_qea_coverages(log_K_CO=log_K1, log_K_OH_Pd=log_K4, state=state)

    log_rate_ER = calculate_log_rate_er(log_k2_ER=log_k2_ER, log_theta_CO=pd_coverages.log_theta_CO, state=state)
    mechanism_result = MechanismResult(
        ln_rate=log_rate_ER,
        pointwise={
            "theta_CO": pt.exp(pd_coverages.log_theta_CO),
            "theta_OH_Pd": pt.exp(pd_coverages.log_theta_OH_Pd),
            "theta_empty_Pd": pt.exp(pd_coverages.log_theta_empty_Pd),
            "ln_rate_ER": log_rate_ER,
        },
    )
    return AgPdERResult(
        mechanism_result=mechanism_result,
        log_rate_ER=log_rate_ER,
        log_K1=log_K1,
        log_K4=log_K4,
        log_k2_ER=log_k2_ER,
    )


def evaluate_agpd_bf_er_lh(state: AgPdPointState, parameters: AgPdBFERRLHParameters, temperature_K):
    E = pt.as_tensor_variable(state.E_V_SHE)
    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(delta_G_0_eV=parameters.deltaG4_0, electron_transfer=1.0, E_V_SHE=E)
    deltaG5 = electrochemical_free_energy(
        delta_G_0_eV=parameters.deltaG5_0, electron_transfer=parameters.q, E_V_SHE=E
    )
    Gact2_BF = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_BF_0,
        beta=parameters.beta_2_BF,
        electron_transfer=1.0 - parameters.q,
        E_V_SHE=E,
    )
    Gact2_ER = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_ER_0,
        beta=parameters.beta_2_ER,
        electron_transfer=1.0,
        E_V_SHE=E,
    )
    Gact2_LH = pt.as_tensor_variable(parameters.Gact2_LH_0)
    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)
    log_K5 = log_equilibrium_constant(delta_G_eV=deltaG5, temperature_K=temperature_K)

    log_k2_BF = log_tst_rate_constant(activation_G_eV=Gact2_BF, temperature_K=temperature_K)
    log_k2_ER = log_tst_rate_constant(activation_G_eV=Gact2_ER, temperature_K=temperature_K)
    log_k2_LH = log_tst_rate_constant(activation_G_eV=Gact2_LH, temperature_K=temperature_K)
    pd_coverages = calculate_pd_qea_coverages(log_K_CO=log_K1, log_K_OH_Pd=log_K4, state=state)
    ag_coverages = calculate_ag_qea_coverages(log_K_OH_Ag=log_K5, state=state)

    log_rate_BF = calculate_log_rate_bf(
        log_k2_BF=log_k2_BF,
        log_theta_CO=pd_coverages.log_theta_CO,
        log_theta_OH_Ag=ag_coverages.log_theta_OH_Ag,
        state=state,
    )
    log_rate_ER = calculate_log_rate_er(log_k2_ER=log_k2_ER, log_theta_CO=pd_coverages.log_theta_CO, state=state)
    log_rate_LH = calculate_log_rate_lh(
        log_k2_LH=log_k2_LH,
        log_theta_CO=pd_coverages.log_theta_CO,
        log_theta_OH_Pd=pd_coverages.log_theta_OH_Pd,
        state=state,
    )
    log_rate_total = logsumexp_pathways(log_rate_BF, log_rate_ER, log_rate_LH)

    rate_fraction_BF = pt.exp(log_rate_BF - log_rate_total)
    rate_fraction_ER = pt.exp(log_rate_ER - log_rate_total)
    rate_fraction_LH = pt.exp(log_rate_LH - log_rate_total)
    mechanism_result = MechanismResult(
        ln_rate=log_rate_total,
        pointwise={
            "theta_CO": pt.exp(pd_coverages.log_theta_CO),
            "theta_OH_Pd": pt.exp(pd_coverages.log_theta_OH_Pd),
            "theta_empty_Pd": pt.exp(pd_coverages.log_theta_empty_Pd),
            "theta_OH_Ag": pt.exp(ag_coverages.log_theta_OH_Ag),
            "theta_empty_Ag": pt.exp(ag_coverages.log_theta_empty_Ag),
            "ln_rate_BF": log_rate_BF,
            "ln_rate_ER": log_rate_ER,
            "ln_rate_LH": log_rate_LH,
            "rate_fraction_BF": rate_fraction_BF,
            "rate_fraction_ER": rate_fraction_ER,
            "rate_fraction_LH": rate_fraction_LH,
        },
    )
    return AgPdBFERRLHResult(
        mechanism_result=mechanism_result,
        log_rate_BF=log_rate_BF,
        log_rate_ER=log_rate_ER,
        log_rate_LH=log_rate_LH,
        log_rate_total=log_rate_total,
        log_K1=log_K1,
        log_K4=log_K4,
        log_K5=log_K5,
        log_k2_BF=log_k2_BF,
        log_k2_ER=log_k2_ER,
        log_k2_LH=log_k2_LH,
    )
