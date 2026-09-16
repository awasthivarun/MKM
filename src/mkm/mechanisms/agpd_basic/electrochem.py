import numpy as np
import pytensor.tensor as pt

from mkm.constants import H_J_S, K_B_EV_K, K_B_J_K

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
    return pt.as_tensor_variable(delta_G_0_eV) - pt.as_tensor_variable(electron_transfer) * pt.as_tensor_variable(
        E_V_SHE
    )


def electrochemical_activation_energy(activation_G_0_eV, beta, electron_transfer, E_V_SHE):
    return (
        pt.as_tensor_variable(activation_G_0_eV)
        - pt.as_tensor_variable(beta) * pt.as_tensor_variable(electron_transfer) * pt.as_tensor_variable(E_V_SHE)
    )

