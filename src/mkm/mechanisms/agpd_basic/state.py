from dataclasses import dataclass

import numpy as np

from mkm.model_inputs import ModelPointInputs

@dataclass(frozen=True)
class AgPdPointState:
    E_V_SHE: np.ndarray

    ln_a_OH: np.ndarray
    ln_a_CO: np.ndarray

    Ag_fraction: np.ndarray
    Pd_fraction: np.ndarray

    theta_CO_max: np.ndarray | None = None
    materials: tuple[str, ...] | None = None
    material_index: np.ndarray | None = None

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
        missing_caps = [material for material in inputs.materials if material not in coverage_cap_config]
        if missing_caps:
            raise ValueError(f"Missing CO coverage caps for materials: {missing_caps}.")
        theta_CO_max_by_material = np.asarray(
            [float(coverage_cap_config[material]) for material in inputs.materials], dtype=float
        )
        if not np.all(np.isfinite(theta_CO_max_by_material)):
            raise ValueError("CO coverage caps must be finite.")
        if np.any(theta_CO_max_by_material <= 0.0) or np.any(theta_CO_max_by_material > 1.0):
            raise ValueError("CO coverage caps must lie in (0, 1].")
        theta_CO_max = theta_CO_max_by_material[material_index]

    return AgPdPointState(
        E_V_SHE=np.asarray(inputs.E_V_SHE, dtype=float),
        ln_a_OH=ln_a_OH,
        ln_a_CO=ln_a_CO,
        Ag_fraction=Ag_fraction,
        Pd_fraction=Pd_fraction,
        materials=tuple(inputs.materials),
        material_index=material_index,
        theta_CO_max=theta_CO_max,
    )

