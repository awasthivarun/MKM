from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from .state import AgPdPointState

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

