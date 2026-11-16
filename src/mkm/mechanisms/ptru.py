"""Pt and PtRu finite-CO-SSA / quasi-equilibrium OH kinetics in acidic or basic media.

All rates are normalized per Pt CO adsorption site. Ru OH adsorption transfers one
full electron and BF coupling is chemical: neither q nor beta_2_BF is fitted.
"""

from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from mkm.mechanisms.base import MechanismResult
from mkm.mechanisms.co_ssa import solve_co_ssa_qea_oh
from mkm.mechanisms.electrochem import (
    electrochemical_activation_energy, electrochemical_free_energy,
    log_equilibrium_constant, log_tst_rate_constant,
)


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


@dataclass(frozen=True)
class PtRuCOBFParameters:
    deltaG1_0: object
    deltaG4_0: object
    deltaG5_0: object
    Gact1_0: object
    Gact2_BF_0: object


@dataclass(frozen=True)
class PtRuCOBFLHParameters(PtRuCOBFParameters):
    Gact2_LH_0: object


@dataclass(frozen=True)
class PtRuCOBFERParameters(PtRuCOBFParameters):
    Gact2_ER_0: object
    beta_2_ER: object


@dataclass(frozen=True)
class PtRuCOBFERLHParameters(PtRuCOBFERParameters):
    Gact2_LH_0: object


PTRU_PARAMETER_CLASSES = {
    "CO_LH": PtCOLHParameters,
    "CO_ER": PtCOERParameters,
    "CO_BF": PtRuCOBFParameters,
    "CO_ER_LH": PtCOERLHParameters,
    "CO_BF_LH": PtRuCOBFLHParameters,
    "CO_BF_ER": PtRuCOBFERParameters,
    "CO_BF_ER_LH": PtRuCOBFERLHParameters,
}
PURE_PT_MODELS = ("CO_LH", "CO_ER", "CO_ER_LH")


def evaluate_ptru_co(state, parameters, temperature_K, medium, model_name, ru_fraction):
    """Evaluate a single or pooled material vector using its pointwise Ru availability."""
    if medium not in ("acidic", "basic"):
        raise ValueError(f"Unknown PtRu medium: {medium}")
    if model_name not in PTRU_PARAMETER_CLASSES:
        raise ValueError(f"Unknown PtRu mechanism: {model_name}")
    ru_fraction = np.asarray(ru_fraction, dtype=float)
    if not np.all(np.isfinite(ru_fraction)) or np.any((ru_fraction < 0) | (ru_fraction >= 1)):
        raise ValueError("Ru fractions must be finite and in [0, 1).")
    if np.any(ru_fraction == 0) and model_name == "CO_BF":
        raise ValueError("BF-only mechanism predicts zero rate on Pt100; exclude Pt100 or select another model.")

    E = pt.as_tensor_variable(state.E_V_SHE)
    ln_ion = pt.as_tensor_variable(state.ln_electrolyte_concentration)
    ln_co = pt.as_tensor_variable(state.ln_CO_mole_fraction)
    source = -ln_ion if medium == "acidic" else ln_ion
    zeros = pt.zeros_like(E)
    ru = pt.as_tensor_variable(ru_fraction) + zeros
    pathways = tuple(model_name.removeprefix("CO_").split("_"))

    log_k1 = log_tst_rate_constant(parameters.Gact1_0, temperature_K)
    log_k1_co = log_k1 + ln_co
    log_k_minus_1 = log_k1 - log_equilibrium_constant(parameters.deltaG1_0, temperature_K)
    deltaG4 = electrochemical_free_energy(parameters.deltaG4_0, 1.0, E)
    term_oh_pt = log_equilibrium_constant(deltaG4, temperature_K) + source

    log_other = []
    log_k_bf_app = log_k_er_app = log_k_lh = None
    log_theta_empty_ru = log_theta_oh_ru = None
    if "BF" in pathways:
        deltaG5 = electrochemical_free_energy(parameters.deltaG5_0, 1.0, E)
        term_oh_ru = log_equilibrium_constant(deltaG5, temperature_K) + source
        log_theta_empty_ru = -pt.logsumexp(pt.stack((zeros, term_oh_ru)), axis=0)
        log_theta_oh_ru = term_oh_ru + log_theta_empty_ru
        # Never take log(0) on Pt100; enforce exactly zero BF rate there with -inf.
        log_ru = pt.switch(pt.gt(ru, 0.0), pt.log(pt.maximum(ru, 1e-300)), -1e6)
        log_k_bf_app = log_tst_rate_constant(parameters.Gact2_BF_0, temperature_K)
        log_k_bf_app = log_k_bf_app + log_theta_oh_ru + log_ru
        log_other.append(log_k_bf_app)
    if "ER" in pathways:
        barrier = electrochemical_activation_energy(parameters.Gact2_ER_0, parameters.beta_2_ER, 1.0, E)
        log_k_er_app = log_tst_rate_constant(barrier, temperature_K)
        log_k_er_app = log_k_er_app + (zeros if medium == "acidic" else ln_ion)
        log_other.append(log_k_er_app)
    if "LH" in pathways:
        log_k_lh = log_tst_rate_constant(parameters.Gact2_LH_0, temperature_K)

    coverages = solve_co_ssa_qea_oh(
        log_term_oh=term_oh_pt, log_k1_a_co=log_k1_co, log_k_minus_1=log_k_minus_1,
        log_k_other=tuple(log_other),
        log_k_lh_app=None if log_k_lh is None else log_k_lh + term_oh_pt,
    )
    log_rates = {}
    if log_k_bf_app is not None:
        log_rates["BF"] = log_k_bf_app + coverages.log_theta_CO
    if log_k_er_app is not None:
        log_rates["ER"] = log_k_er_app + coverages.log_theta_CO
    if log_k_lh is not None:
        log_rates["LH"] = log_k_lh + coverages.log_theta_CO + coverages.log_theta_OH
    log_total = pt.logsumexp(pt.stack([log_rates[name] for name in pathways]), axis=0)
    theta_empty = pt.exp(coverages.log_theta_empty)
    theta_oh = pt.exp(coverages.log_theta_OH)
    pointwise = {
        "theta_empty": theta_empty, "theta_CO": pt.exp(coverages.log_theta_CO), "theta_OH": theta_oh,
        "theta_empty_Pt": theta_empty, "theta_OH_Pt": theta_oh,
    }
    if log_theta_oh_ru is not None:
        pointwise["theta_empty_Ru"] = pt.switch(pt.gt(ru, 0.0), pt.exp(log_theta_empty_ru), 1.0)
        pointwise["theta_OH_Ru"] = pt.switch(pt.gt(ru, 0.0), pt.exp(log_theta_oh_ru), 0.0)
    for name in pathways:
        pointwise[f"ln_rate_{name}"] = log_rates[name]
        pointwise[f"rate_fraction_{name}"] = pt.exp(log_rates[name] - log_total)
    return MechanismResult(ln_rate=log_total, pointwise=pointwise)

# Legacy evaluator signatures retained in this unified module for reference comparisons.
PT100_PARAMETER_CLASSES = {name: PTRU_PARAMETER_CLASSES[name] for name in PURE_PT_MODELS}
PTRU_ALLOY_PARAMETER_CLASSES = PTRU_PARAMETER_CLASSES


def evaluate_pt100_co(state, parameters, temperature_K, medium, model_name):
    if tuple(state.materials) != ("Pt100",) or model_name not in PT100_PARAMETER_CLASSES:
        raise ValueError("Pure Pt evaluation requires Pt100 and a non-BF mechanism.")
    evaluated = evaluate_ptru_co(state, parameters, temperature_K, medium, model_name, 0.0)
    pointwise = {name: value for name, value in evaluated.pointwise.items()
                 if name not in ("theta_empty_Pt", "theta_OH_Pt")}
    return MechanismResult(evaluated.ln_rate, pointwise)


def evaluate_ptru_alloy_co(state, parameters, temperature_K, medium, model_name, ru_fraction):
    if len(state.materials) != 1 or state.materials[0] == "Pt100":
        raise ValueError("Alloy evaluation requires exactly one alloy material.")
    return evaluate_ptru_co(state, parameters, temperature_K, medium, model_name, ru_fraction)
