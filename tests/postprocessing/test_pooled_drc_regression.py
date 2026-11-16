"""Pooled TS-DRC regression checks against independent single-material calculations.

These use synthetic posterior draws only; there is no sampling or experimental-data dependency.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.model_inputs import ModelPointInputs
from mkm.models.agpd_basic import get_agpd_material_parameter_names
from mkm.models.ptru import ptru_parameter_names
from mkm.postprocessing.drc import (
    compare_transition_state_drc_steps,
    compute_composition_transition_state_drc,
    compute_ptru_transition_state_drc,
    compute_transition_state_drc,
)


AGPD_BASE = {
    "deltaG1_0": -0.12, "deltaG4_0": 0.11, "deltaG5_0": 0.03,
    "Gact1_0": 0.60, "Gact2_BF_0": 0.68, "Gact2_ER_0": 0.73, "Gact2_LH_0": 0.76,
    "beta_2_BF": 0.10, "beta_2_ER": 0.35, "q": 0.35,
}
PTRU_BASE = {
    "deltaG1_0": -0.35, "deltaG4_0": 0.12, "deltaG5_0": -0.05,
    "Gact1_0": 0.62, "Gact2_BF_0": 0.74, "Gact2_ER_0": 0.82,
    "Gact2_LH_0": 0.76, "beta_2_ER": 0.40,
}


def _idata(values):
    # Scalar values get a second draw shifted by 0.001; pairs specify both draws explicitly.
    posterior = xr.Dataset({
        name: (("chain", "draw"), np.array([(
            (value, value + 0.001) if np.isscalar(value) else value
        )], dtype=float))
        for name, value in values.items()
    }, coords={"chain": [0], "draw": [0, 1]})
    return SimpleNamespace(posterior=posterior)


def _points(materials, potentials):
    count = len(materials)
    inputs = ModelPointInputs(
        materials=tuple(materials), material_index=np.arange(count, dtype=np.int64),
        E_V_SHE=np.asarray(potentials, dtype=float),
        ln_electrolyte_concentration=np.log(np.full(count, 0.5)),
        ln_CO_mole_fraction=np.log(np.full(count, 0.1)),
    )
    metadata = pd.DataFrame({
        "model_point_id": np.arange(count), "condition_id": np.arange(count),
        "material": materials, "electrolyte_concentration_M": np.full(count, 0.5),
        "CO_mole_fraction": np.full(count, 0.1), "analysis_grid_index": np.zeros(count, dtype=int),
        "E_V_SHE": potentials,
    })
    return inputs, metadata


def _single_point(inputs, metadata, index):
    material = inputs.materials[index]
    single = ModelPointInputs(
        materials=(material,), material_index=np.zeros(1, dtype=np.int64),
        E_V_SHE=inputs.E_V_SHE[index:index + 1],
        ln_electrolyte_concentration=inputs.ln_electrolyte_concentration[index:index + 1],
        ln_CO_mole_fraction=inputs.ln_CO_mole_fraction[index:index + 1],
    )
    row = metadata.iloc[[index]].copy().reset_index(drop=True)
    row["model_point_id"] = 0
    row["condition_id"] = 0
    return single, row


def _assert_same_drc(pooled, independent, index):
    for control in independent.draws.coords["control"].values:
        actual = pooled.draws["X_TS"].sel(control=control).isel(model_point=index)
        expected = independent.draws["X_TS"].sel(control=control).isel(model_point=0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)


def _assert_sums_and_half_step(coarse, fine, *, atol=2e-5):
    np.testing.assert_allclose(coarse.draws["X_TS"].sum("control"), 1.0, rtol=0, atol=atol)
    convergence = compare_transition_state_drc_steps(coarse, fine)
    assert np.isfinite(convergence["max_abs_difference"]).all()
    assert (convergence["max_abs_difference"] < 2e-4).all()


@pytest.mark.parametrize("parameterization", ("independent", "linear_xAg"))
def test_agpd_pooled_drc_reconstructs_effective_barriers(parameterization):
    model = "CO_BF_ER_LH"
    materials = ("Ag10Pd90", "Ag90Pd10")
    fractions = (0.1, 0.9)
    config = {
        "temperature_K": 293.15,
        "gas": {"standard_state_pressure_bar": 1.0, "total_pressure_bar": 1.0},
        "electrolyte": {"activity_model": "ideal_molarity", "standard_state_concentration_M": 1.0},
        "surface_composition": {
            material: {"Ag_fraction": fraction, "Pd_fraction": 1 - fraction}
            for material, fraction in zip(materials, fractions)
        },
        "composition_parameterizations": {
            "independent": {"models": {model: {"independent_materials": True}}},
            "linear_xAg": {"x_reference": 0.5, "models": {model: {"slopes": {
                "Gact1_0": {"distribution": "normal", "mu": 0, "sigma": 0.1},
                "Gact2_BF_0": {"distribution": "normal", "mu": 0, "sigma": 0.1},
            }}}},
        },
    }
    inputs, metadata = _points(materials, (0.37, 0.37))
    effective = []
    if parameterization == "independent":
        pooled_values = {}
        for index, material in enumerate(materials):
            values = {name: value + 0.025 * index for name, value in AGPD_BASE.items()}
            effective.append(values)
            for name in get_agpd_material_parameter_names(model, material):
                pooled_values[f"{name}_{material}"] = values[name]
    else:
        slopes = {"Gact1_0": 0.05, "Gact2_BF_0": 0.10}
        pooled_values = {**AGPD_BASE, **{f"{name}_xAg_slope": value for name, value in slopes.items()}}
        for fraction in fractions:
            x_shift = fraction - 0.5
            effective.append({
                name: tuple(
                    value + offset + (slopes.get(name, 0.0) + (offset if name in slopes else 0.0)) * x_shift
                    for offset in (0.0, 0.001)
                )
                for name, value in AGPD_BASE.items()
            })
    kwargs = dict(
        inference_data=_idata(pooled_values), model_name=model, point_inputs=inputs,
        model_points=metadata, config=config, parameterization=parameterization,
    )
    coarse = compute_composition_transition_state_drc(**kwargs, step_eV=1e-4)
    fine = compute_composition_transition_state_drc(**kwargs, step_eV=5e-5)
    _assert_sums_and_half_step(coarse, fine)
    for index in range(len(materials)):
        single_inputs, single_metadata = _single_point(inputs, metadata, index)
        standalone = compute_transition_state_drc(
            _idata(effective[index]), model, single_inputs, single_metadata, config, step_eV=1e-4,
        )
        _assert_same_drc(coarse, standalone, index)
    assert coarse.draws.attrs["parameterization"] == parameterization
    assert not np.allclose(coarse.draws["X_TS"].isel(model_point=0),
                           coarse.draws["X_TS"].isel(model_point=1))


@pytest.mark.parametrize("medium", ("acidic", "basic"))
@pytest.mark.parametrize("parameterization", ("independent", "linear_xRu"))
def test_ptru_pooled_drc_reconstructs_effective_barriers_and_pt100_limits(medium, parameterization):
    model = "CO_BF_ER_LH"
    materials = ("Pt100", "Pt66Ru33", "Pt50Ru50")
    fractions = (0.0, 1 / 3, 0.5)
    config = {
        "temperature_K": 293.15,
        "surface_composition": {material: {"Ru_fraction": fraction}
                                for material, fraction in zip(materials, fractions)},
        "parameterizations": {"linear_xRu": {"x_reference": 0.25}},
    }
    priors = {"prior_profiles": {medium: {"parameters": {
        "beta_2_ER": {"lower": 0.0, "upper": 1.0},
    }}}}
    base = dict(PTRU_BASE)
    if medium == "acidic":
        base.update(deltaG4_0=0.85, deltaG5_0=0.80, Gact2_ER_0=1.12)
    potentials = (0.72, 0.72, 0.72) if medium == "acidic" else (0.32, 0.32, 0.32)
    inputs, metadata = _points(materials, potentials)
    effective = []
    if parameterization == "independent":
        pooled_values = {}
        for index, material in enumerate(materials):
            values = {name: value + 0.012 * index for name, value in base.items()}
            effective.append(values)
            for name in ptru_parameter_names(model, material):
                pooled_values[f"{name}_{material}"] = values[name]
    else:
        slopes = {name: 0.0 for name in base}
        slopes.update(Gact1_0=0.07, Gact2_BF_0=0.12, beta_2_ER=0.4)
        pooled_values = {**base, **{f"{name}_xRu_slope": slope for name, slope in slopes.items()}}
        x_reference = 0.25
        for fraction in fractions:
            values = {}
            for name, value in base.items():
                draws = []
                for offset in (0.0, 0.001):
                    reference = value + offset
                    slope = slopes[name] + offset
                    if name == "beta_2_ER":
                        margin = min(reference, 1.0 - reference)
                        slope *= margin / max(abs(x_reference), abs(0.5 - x_reference))
                    draws.append(reference + slope * (fraction - x_reference))
                values[name] = tuple(draws)
            effective.append(values)
    spec = SimpleNamespace(model_name=model, medium=medium, is_all_materials=True,
                           parameterization=parameterization)
    kwargs = dict(inference_data=_idata(pooled_values), specification=spec, point_inputs=inputs,
                  model_points=metadata, config=config, priors=priors)
    coarse = compute_ptru_transition_state_drc(**kwargs, step_eV=1e-4)
    fine = compute_ptru_transition_state_drc(**kwargs, step_eV=5e-5)
    _assert_sums_and_half_step(coarse, fine)
    assert "q" not in pooled_values and "beta_2_BF" not in pooled_values
    # Pt100 cannot perform BF coupling, even though the pooled model includes BF on alloys.
    np.testing.assert_allclose(coarse.draws["X_TS"].sel(control="BF").isel(model_point=0),
                               0.0, rtol=0, atol=1e-12)
    for index, material in enumerate(materials):
        single_inputs, single_metadata = _single_point(inputs, metadata, index)
        single_model = "CO_ER_LH" if material == "Pt100" else model
        names = ptru_parameter_names(single_model, material)
        single_values = {name: effective[index][name] for name in names}
        standalone = compute_ptru_transition_state_drc(
            _idata(single_values),
            SimpleNamespace(model_name=single_model, medium=medium, is_all_materials=False,
                            parameterization=None),
            single_inputs, single_metadata, config, priors, step_eV=1e-4,
        )
        _assert_same_drc(coarse, standalone, index)
    assert not np.allclose(coarse.draws["X_TS"].isel(model_point=1),
                           coarse.draws["X_TS"].isel(model_point=2))
