from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from mkm.constants import H_J_S, K_B_EV_K, K_B_J_K
from mkm.model_inputs import ModelPointInputs
from mkm.mechanisms.base import MechanismResult


@dataclass(frozen=True)
class AgPdPointState:
    E_V_SHE: np.ndarray

    ln_a_OH: np.ndarray
    ln_a_CO: np.ndarray

    Ag_fraction: np.ndarray
    Pd_fraction: np.ndarray

    theta_CO_max: np.ndarray | None = None


def thermal_energy_eV(temperature_K):
    temperature_K = float(temperature_K)

    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("Temperature must be finite and positive.")

    return K_B_EV_K * temperature_K


def log_eyring_prefactor_s_inv(temperature_K):
    temperature_K = float(temperature_K)

    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("Temperature must be finite and positive.")

    return np.log(K_B_J_K * temperature_K / H_J_S)


def log_equilibrium_constant(delta_G_eV, temperature_K):
    return -pt.as_tensor_variable(delta_G_eV) / thermal_energy_eV(temperature_K)


def log_tst_rate_constant(activation_G_eV, temperature_K):
    return log_eyring_prefactor_s_inv(temperature_K) - pt.as_tensor_variable(activation_G_eV) / thermal_energy_eV(
        temperature_K
    )


def electrochemical_free_energy(delta_G_0_eV, electron_transfer, E_V_SHE):
    return (
        pt.as_tensor_variable(delta_G_0_eV)
        - pt.as_tensor_variable(electron_transfer) * pt.as_tensor_variable(E_V_SHE)
    )


def electrochemical_activation_energy(activation_G_0_eV, beta, electron_transfer, E_V_SHE):
    return (
        pt.as_tensor_variable(activation_G_0_eV)
        - pt.as_tensor_variable(beta)
        * pt.as_tensor_variable(electron_transfer)
        * pt.as_tensor_variable(E_V_SHE)
    )


def build_agpd_point_state(inputs: ModelPointInputs, config):
    gas = config["gas"]
    standard_pressure_bar = float(gas["standard_state_pressure_bar"])
    total_pressure_bar = float(gas["total_pressure_bar"])

    if not np.isfinite(standard_pressure_bar) or standard_pressure_bar <= 0:
        raise ValueError("Gas standard-state pressure must be finite and positive.")

    if not np.isfinite(total_pressure_bar) or total_pressure_bar <= 0:
        raise ValueError("Total gas pressure must be finite and positive.")

    electrolyte = config["electrolyte"]
    activity_model = electrolyte["activity_model"]

    if activity_model != "ideal_molarity":
        raise ValueError(f"Unsupported electrolyte activity model '{activity_model}'.")

    standard_concentration_M = float(electrolyte["standard_state_concentration_M"])

    if not np.isfinite(standard_concentration_M) or standard_concentration_M <= 0:
        raise ValueError("Electrolyte standard-state concentration must be finite and positive.")

    ln_a_OH = np.asarray(inputs.ln_electrolyte_concentration, dtype=float) - np.log(standard_concentration_M)
    ln_a_CO = np.asarray(inputs.ln_CO_mole_fraction, dtype=float) + np.log(total_pressure_bar / standard_pressure_bar)

    composition_config = config["surface_composition"]

    Ag_fraction_by_material = []
    Pd_fraction_by_material = []

    for material in inputs.materials:
        if material not in composition_config:
            raise ValueError(f"Missing surface composition for material '{material}'.")

        composition = composition_config[material]

        Ag_fraction = float(composition["Ag_fraction"])
        Pd_fraction = float(composition["Pd_fraction"])

        if not np.isfinite(Ag_fraction) or not np.isfinite(Pd_fraction):
            raise ValueError(f"Non-finite surface composition for material '{material}'.")

        if not (0.0 <= Ag_fraction <= 1.0):
            raise ValueError(f"Ag fraction for '{material}' must lie between 0 and 1.")

        if not (0.0 <= Pd_fraction <= 1.0):
            raise ValueError(f"Pd fraction for '{material}' must lie between 0 and 1.")

        if not np.isclose(Ag_fraction + Pd_fraction, 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"Ag and Pd fractions for '{material}' must sum to 1.")

        Ag_fraction_by_material.append(Ag_fraction)
        Pd_fraction_by_material.append(Pd_fraction)

    Ag_fraction_by_material = np.asarray(Ag_fraction_by_material, dtype=float)
    Pd_fraction_by_material = np.asarray(Pd_fraction_by_material, dtype=float)

    material_index = np.asarray(inputs.material_index, dtype=np.int64)

    Ag_fraction = Ag_fraction_by_material[material_index]
    Pd_fraction = Pd_fraction_by_material[material_index]

    theta_CO_max = None
    coverage_cap_config = config.get("co_coverage_cap")

    if coverage_cap_config is not None:
        missing_caps = [
            material
            for material in inputs.materials
            if material not in coverage_cap_config
        ]
        if missing_caps:
            raise ValueError(
                f"Missing CO coverage caps for materials: {missing_caps}."
            )

        theta_CO_max_by_material = np.asarray(
            [
                float(coverage_cap_config[material])
                for material in inputs.materials
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(theta_CO_max_by_material)):
            raise ValueError("CO coverage caps must be finite.")

        if np.any(theta_CO_max_by_material <= 0.0) or np.any(
            theta_CO_max_by_material > 1.0
        ):
            raise ValueError("CO coverage caps must lie in (0, 1].")

        theta_CO_max = theta_CO_max_by_material[material_index]

    return AgPdPointState(
        E_V_SHE=np.asarray(inputs.E_V_SHE, dtype=float),
        ln_a_OH=ln_a_OH,
        ln_a_CO=ln_a_CO,
        Ag_fraction=Ag_fraction,
        Pd_fraction=Pd_fraction,
        theta_CO_max=theta_CO_max,
    )


@dataclass(frozen=True)
class PdQEACoverages:
    log_theta_empty_Pd: object
    log_theta_CO: object
    log_theta_OH_Pd: object


@dataclass(frozen=True)
class AgQEACoverages:
    log_theta_empty_Ag: object
    log_theta_OH_Ag: object


def calculate_pd_qea_coverages(log_K_CO, log_K_OH_Pd, state: AgPdPointState):
    log_K_CO = pt.as_tensor_variable(log_K_CO)
    log_K_OH_Pd = pt.as_tensor_variable(log_K_OH_Pd)

    ln_a_CO = pt.as_tensor_variable(state.ln_a_CO)
    ln_a_OH = pt.as_tensor_variable(state.ln_a_OH)

    term_CO = log_K_CO + ln_a_CO
    term_OH_Pd = log_K_OH_Pd + ln_a_OH

    zeros = pt.zeros_like(term_CO)

    log_theta_empty_Pd = -pt.logsumexp(pt.stack([zeros, term_CO, term_OH_Pd]), axis=0)
    log_theta_CO = term_CO + log_theta_empty_Pd
    log_theta_OH_Pd = term_OH_Pd + log_theta_empty_Pd

    return PdQEACoverages(
        log_theta_empty_Pd=log_theta_empty_Pd,
        log_theta_CO=log_theta_CO,
        log_theta_OH_Pd=log_theta_OH_Pd,
    )


def calculate_ag_qea_coverages(log_K_OH_Ag, state: AgPdPointState):
    log_K_OH_Ag = pt.as_tensor_variable(log_K_OH_Ag)
    ln_a_OH = pt.as_tensor_variable(state.ln_a_OH)

    term_OH_Ag = log_K_OH_Ag + ln_a_OH
    zeros = pt.zeros_like(term_OH_Ag)

    log_theta_empty_Ag = -pt.logsumexp(pt.stack([zeros, term_OH_Ag]), axis=0)
    log_theta_OH_Ag = term_OH_Ag + log_theta_empty_Ag

    return AgQEACoverages(log_theta_empty_Ag=log_theta_empty_Ag, log_theta_OH_Ag=log_theta_OH_Ag)


@dataclass(frozen=True)
class AgPdBFParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2: object
    q: object

    Gact2_0: object


@dataclass(frozen=True)
class AgPdBFResult:
    mechanism_result: object

    log_rate_BF: object

    log_K1: object
    log_K4: object
    log_K5: object

    log_k2_BF: object


def evaluate_agpd_bf(state: AgPdPointState, parameters: AgPdBFParameters, temperature_K):
    if np.any(state.Ag_fraction <= 0):
        raise ValueError("The BF-only mechanism requires a positive Ag surface fraction at every model point.")

    E = pt.as_tensor_variable(state.E_V_SHE)

    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(delta_G_0_eV=parameters.deltaG4_0, electron_transfer=1.0, E_V_SHE=E)
    deltaG5 = electrochemical_free_energy(
        delta_G_0_eV=parameters.deltaG5_0, electron_transfer=parameters.q, E_V_SHE=E
    )

    Gact2_BF = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_0,
        beta=parameters.beta_2,
        electron_transfer=1.0 - parameters.q,
        E_V_SHE=E,
    )

    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)
    log_K5 = log_equilibrium_constant(delta_G_eV=deltaG5, temperature_K=temperature_K)

    log_k2_BF = log_tst_rate_constant(activation_G_eV=Gact2_BF, temperature_K=temperature_K)

    pd_coverages = calculate_pd_qea_coverages(log_K_CO=log_K1, log_K_OH_Pd=log_K4, state=state)
    ag_coverages = calculate_ag_qea_coverages(log_K_OH_Ag=log_K5, state=state)

    log_Ag_fraction = log_surface_fraction(state.Ag_fraction)

    log_rate_BF = log_k2_BF + pd_coverages.log_theta_CO + ag_coverages.log_theta_OH_Ag + log_Ag_fraction

    mechanism_result = MechanismResult(
        ln_rate=log_rate_BF,
        pointwise={
            "theta_CO": pt.exp(pd_coverages.log_theta_CO),
            "theta_OH_Pd": pt.exp(pd_coverages.log_theta_OH_Pd),
            "theta_empty_Pd": pt.exp(pd_coverages.log_theta_empty_Pd),
            "theta_OH_Ag": pt.exp(ag_coverages.log_theta_OH_Ag),
            "theta_empty_Ag": pt.exp(ag_coverages.log_theta_empty_Ag),
            "ln_rate_BF": log_rate_BF,
        },
    )

    return AgPdBFResult(
        mechanism_result=mechanism_result,
        log_rate_BF=log_rate_BF,
        log_K1=log_K1,
        log_K4=log_K4,
        log_K5=log_K5,
        log_k2_BF=log_k2_BF,
    )


def log_surface_fraction(fraction):
    fraction = np.asarray(fraction, dtype=float)

    if not np.all(np.isfinite(fraction)):
        raise ValueError("Surface fractions contain non-finite values.")

    if np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("Surface fractions must lie between 0 and 1.")

    log_fraction = np.full(fraction.shape, -np.inf, dtype=float)
    positive = fraction > 0

    log_fraction[positive] = np.log(fraction[positive])

    return pt.as_tensor_variable(log_fraction)


def logsumexp_pathways(*log_rates):
    if len(log_rates) < 1:
        raise ValueError("At least one pathway is required.")

    tensors = [pt.as_tensor_variable(log_rate) for log_rate in log_rates]

    for tensor in tensors:
        if tensor.ndim != 1:
            raise ValueError("Pathway log rates must be one-dimensional.")

    return pt.logsumexp(pt.stack(tensors, axis=0), axis=0)


@dataclass(frozen=True)
class AgPdBFLHParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object

    beta_2: object
    q: object

    Gact2_BF_0: object
    Gact2_LH_0: object


@dataclass(frozen=True)
class AgPdBFLHResult:
    mechanism_result: MechanismResult

    log_rate_BF: object
    log_rate_LH: object
    log_rate_total: object

    log_K1: object
    log_K4: object
    log_K5: object

    log_k2_BF: object
    log_k2_LH: object


def evaluate_agpd_bf_lh(state: AgPdPointState, parameters: AgPdBFLHParameters, temperature_K):
    E = pt.as_tensor_variable(state.E_V_SHE)

    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(delta_G_0_eV=parameters.deltaG4_0, electron_transfer=1.0, E_V_SHE=E)
    deltaG5 = electrochemical_free_energy(
        delta_G_0_eV=parameters.deltaG5_0, electron_transfer=parameters.q, E_V_SHE=E
    )

    Gact2_BF = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_BF_0,
        beta=parameters.beta_2,
        electron_transfer=1.0 - parameters.q,
        E_V_SHE=E,
    )
    Gact2_LH = pt.as_tensor_variable(parameters.Gact2_LH_0)

    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)
    log_K5 = log_equilibrium_constant(delta_G_eV=deltaG5, temperature_K=temperature_K)

    log_k2_BF = log_tst_rate_constant(activation_G_eV=Gact2_BF, temperature_K=temperature_K)
    log_k2_LH = log_tst_rate_constant(activation_G_eV=Gact2_LH, temperature_K=temperature_K)

    pd_coverages = calculate_pd_qea_coverages(log_K_CO=log_K1, log_K_OH_Pd=log_K4, state=state)
    ag_coverages = calculate_ag_qea_coverages(log_K_OH_Ag=log_K5, state=state)

    log_Ag_fraction = log_surface_fraction(state.Ag_fraction)
    log_Pd_fraction = log_surface_fraction(state.Pd_fraction)

    log_rate_BF = log_k2_BF + pd_coverages.log_theta_CO + ag_coverages.log_theta_OH_Ag + log_Ag_fraction
    log_rate_LH = log_k2_LH + pd_coverages.log_theta_CO + pd_coverages.log_theta_OH_Pd + log_Pd_fraction

    log_rate_total = logsumexp_pathways(log_rate_BF, log_rate_LH)

    rate_fraction_BF = pt.exp(log_rate_BF - log_rate_total)
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
            "ln_rate_LH": log_rate_LH,
            "rate_fraction_BF": rate_fraction_BF,
            "rate_fraction_LH": rate_fraction_LH,
        },
    )

    return AgPdBFLHResult(
        mechanism_result=mechanism_result,
        log_rate_BF=log_rate_BF,
        log_rate_LH=log_rate_LH,
        log_rate_total=log_rate_total,
        log_K1=log_K1,
        log_K4=log_K4,
        log_K5=log_K5,
        log_k2_BF=log_k2_BF,
        log_k2_LH=log_k2_LH,
    )


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
        activation_G_0_eV=parameters.Gact2_ER_0, beta=parameters.beta_2_ER, electron_transfer=1.0, E_V_SHE=E
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
        activation_G_0_eV=parameters.Gact2_ER_0, beta=parameters.beta_2_ER, electron_transfer=1.0, E_V_SHE=E
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


@dataclass(frozen=True)
class PdCOSSACoverages:
    log_theta_empty_Pd: object
    log_theta_CO: object
    log_theta_OH_Pd: object


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
    log_k_BF_app,
    log_k_ER_app,
    log_k_LH_app,
    state: AgPdPointState,
    theta_CO_max=1.0,
):
    log_K_OH_Pd = pt.as_tensor_variable(log_K_OH_Pd)
    term_OH_Pd = log_K_OH_Pd + pt.as_tensor_variable(state.ln_a_OH)

    point_template = pt.zeros_like(term_OH_Pd)

    log_k1_a_CO = pt.as_tensor_variable(log_k1_a_CO) + point_template
    log_k_minus_1 = pt.as_tensor_variable(log_k_minus_1) + point_template
    log_k_BF_app = pt.as_tensor_variable(log_k_BF_app) + point_template
    log_k_ER_app = pt.as_tensor_variable(log_k_ER_app) + point_template
    log_k_LH_app = pt.as_tensor_variable(log_k_LH_app) + point_template

    log_A_Pd = pt.logsumexp(pt.stack([point_template, term_OH_Pd]), axis=0)
    log_k_cons = pt.logsumexp(pt.stack([log_k_minus_1, log_k_BF_app, log_k_ER_app]), axis=0)

    log_rho = log_k1_a_CO - log_k_cons
    log_lambda = log_k_LH_app - log_k_cons

    A_Pd = pt.exp(log_A_Pd)
    rho = pt.exp(log_rho)
    lambda_LH = pt.exp(log_lambda)

    quadratic_a = A_Pd * lambda_LH
    quadratic_b = rho / pt.as_tensor_variable(theta_CO_max) + A_Pd - lambda_LH

    discriminant = quadratic_b**2 + 4.0 * quadratic_a
    sqrt_discriminant = pt.sqrt(discriminant)

    root_positive_b = 2.0 / (quadratic_b + sqrt_discriminant)
    root_negative_b = (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a)

    theta_empty_Pd = pt.where(quadratic_b >= 0, root_positive_b, root_negative_b)

    log_theta_empty_Pd = pt.log(theta_empty_Pd)
    log_theta_CO = log_rho + log_theta_empty_Pd - pt.log1p(lambda_LH * theta_empty_Pd)
    log_theta_OH_Pd = term_OH_Pd + log_theta_empty_Pd

    return PdCOSSACoverages(
        log_theta_empty_Pd=log_theta_empty_Pd,
        log_theta_CO=log_theta_CO,
        log_theta_OH_Pd=log_theta_OH_Pd,
    )


def evaluate_agpd_co_bf_er_lh(
    state: AgPdPointState,
    parameters: AgPdCOBFERRLHParameters,
    temperature_K,
    theta_CO_max=None,
):
    if np.all(state.Ag_fraction <= 0):
        raise ValueError(
            "CO-BF-ER-LH requires a positive Ag fraction in at least one model point because "
            "BF parameters would otherwise be unidentified."
        )

    if np.any(state.Pd_fraction <= 0):
        raise ValueError("CO-BF-ER-LH requires a positive Pd fraction.")

    E = pt.as_tensor_variable(state.E_V_SHE)

    deltaG1 = pt.as_tensor_variable(parameters.deltaG1_0)
    deltaG4 = electrochemical_free_energy(delta_G_0_eV=parameters.deltaG4_0, electron_transfer=1.0, E_V_SHE=E)
    deltaG5 = electrochemical_free_energy(
        delta_G_0_eV=parameters.deltaG5_0, electron_transfer=parameters.q, E_V_SHE=E
    )

    Gact1 = pt.as_tensor_variable(parameters.Gact1_0)

    Gact2_BF = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_BF_0,
        beta=parameters.beta_2_BF,
        electron_transfer=1.0 - parameters.q,
        E_V_SHE=E,
    )
    Gact2_ER = electrochemical_activation_energy(
        activation_G_0_eV=parameters.Gact2_ER_0, beta=parameters.beta_2_ER, electron_transfer=1.0, E_V_SHE=E
    )
    Gact2_LH = pt.as_tensor_variable(parameters.Gact2_LH_0)

    log_K1 = log_equilibrium_constant(delta_G_eV=deltaG1, temperature_K=temperature_K)
    log_K4 = log_equilibrium_constant(delta_G_eV=deltaG4, temperature_K=temperature_K)
    log_K5 = log_equilibrium_constant(delta_G_eV=deltaG5, temperature_K=temperature_K)

    log_k1 = log_tst_rate_constant(activation_G_eV=Gact1, temperature_K=temperature_K)
    log_k_minus_1 = log_k1 - log_K1

    log_k2_BF = log_tst_rate_constant(activation_G_eV=Gact2_BF, temperature_K=temperature_K)
    log_k2_ER = log_tst_rate_constant(activation_G_eV=Gact2_ER, temperature_K=temperature_K)
    log_k2_LH = log_tst_rate_constant(activation_G_eV=Gact2_LH, temperature_K=temperature_K)

    ag_coverages = calculate_ag_qea_coverages(log_K_OH_Ag=log_K5, state=state)

    log_k1_a_CO = log_k1 + pt.as_tensor_variable(state.ln_a_CO)

    log_k_BF_app = log_k2_BF + ag_coverages.log_theta_OH_Ag + log_surface_fraction(state.Ag_fraction)
    log_k_ER_app = log_k2_ER + pt.as_tensor_variable(state.ln_a_OH)

    term_OH_Pd = log_K4 + pt.as_tensor_variable(state.ln_a_OH)
    log_k_LH_app = log_k2_LH + term_OH_Pd + log_surface_fraction(state.Pd_fraction)

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

    log_rate_BF = log_k_BF_app + pd_coverages.log_theta_CO
    log_rate_ER = log_k_ER_app + pd_coverages.log_theta_CO

    # LH contains an additional theta_empty_Pd
    # because OH_Pd = K4 a_OH theta_empty_Pd.
    log_rate_LH = log_k_LH_app + pd_coverages.log_theta_empty_Pd + pd_coverages.log_theta_CO

    log_rate_total = logsumexp_pathways(log_rate_BF, log_rate_ER, log_rate_LH)

    rate_fraction_BF = pt.exp(log_rate_BF - log_rate_total)
    rate_fraction_ER = pt.exp(log_rate_ER - log_rate_total)
    rate_fraction_LH = pt.exp(log_rate_LH - log_rate_total)

    theta_CO = pt.exp(pd_coverages.log_theta_CO)

    pointwise = {
        "theta_CO": theta_CO,
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
    }

    if theta_CO_max is not None:
        pointwise["theta_CO_site_occupation"] = (
            theta_CO / pt.as_tensor_variable(theta_CO_max)
        )

    mechanism_result = MechanismResult(
        ln_rate=log_rate_total,
        pointwise=pointwise,
    )

    return AgPdCOBFERRLHResult(
        mechanism_result=mechanism_result,
        log_rate_BF=log_rate_BF,
        log_rate_ER=log_rate_ER,
        log_rate_LH=log_rate_LH,
        log_rate_total=log_rate_total,
        log_K1=log_K1,
        log_K4=log_K4,
        log_K5=log_K5,
        log_k1=log_k1,
        log_k_minus_1=log_k_minus_1,
        log_k2_BF=log_k2_BF,
        log_k2_ER=log_k2_ER,
        log_k2_LH=log_k2_LH,
    )


def evaluate_agpd_co_bf_er_lh_capped(
    state: AgPdPointState,
    parameters: AgPdCOBFERRLHParameters,
    temperature_K,
):
    if state.theta_CO_max is None:
        raise ValueError(
            "CO_BF_ER_LH_capped requires material-resolved "
            "CO coverage caps in the model configuration."
        )

    return evaluate_agpd_co_bf_er_lh(
        state=state,
        parameters=parameters,
        temperature_K=temperature_K,
        theta_CO_max=state.theta_CO_max,
    )