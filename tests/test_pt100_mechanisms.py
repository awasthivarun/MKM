
from dataclasses import fields, replace

import numpy as np
import pytest

from mkm.constants import H_J_S, K_B_EV_K, K_B_J_K
from mkm.mechanisms.pt100 import PT100_PARAMETER_CLASSES, evaluate_pt100_co
from mkm.model_inputs import ModelPointInputs


TEMPERATURE_K = 293.15

REFERENCE_PARAMETERS = {
    "deltaG1_0": -0.12,
    "deltaG4_0": 0.05,
    "Gact1_0": 0.55,
    "Gact2_ER_0": 0.72,
    "Gact2_LH_0": 0.75,
    "beta_2_ER": 0.45,
}


def make_inputs():
    return ModelPointInputs(
        materials=("Pt100",),
        material_index=np.zeros(3, dtype=int),
        E_V_SHE=np.array([0.02, 0.07, 0.12]),
        ln_electrolyte_concentration=np.log([0.25, 0.50, 1.00]),
        ln_CO_mole_fraction=np.log([0.01, 0.10, 1.00]),
    )


def make_parameters(model_name):
    parameter_class = PT100_PARAMETER_CLASSES[model_name]
    values = {field.name: REFERENCE_PARAMETERS[field.name] for field in fields(parameter_class)}
    return parameter_class(**values)


def reference_solution(inputs, medium, model_name):
    """Independent NumPy evaluation of the current Pt100 finite-CO-SSA model."""
    p = REFERENCE_PARAMETERS
    E = inputs.E_V_SHE
    concentration = np.exp(inputs.ln_electrolyte_concentration)
    x_co = np.exp(inputs.ln_CO_mole_fraction)

    kbt = K_B_EV_K * TEMPERATURE_K
    prefactor = K_B_J_K * TEMPERATURE_K / H_J_S

    k1 = prefactor * np.exp(-p["Gact1_0"] / kbt)
    K1 = np.exp(-p["deltaG1_0"] / kbt)
    k_minus_1 = k1 / K1
    k1_co = k1 * x_co

    oh_activity = 1.0 / concentration if medium == "acidic" else concentration
    oh_ratio = np.exp(-(p["deltaG4_0"] - E) / kbt) * oh_activity

    has_er = "ER" in model_name.split("_")
    has_lh = "LH" in model_name.split("_")

    k_er = np.zeros_like(E)
    if has_er:
        barrier = p["Gact2_ER_0"] - p["beta_2_ER"] * E
        k_er = prefactor * np.exp(-barrier / kbt)
        if medium == "basic":
            k_er *= concentration

    k_lh = prefactor * np.exp(-p["Gact2_LH_0"] / kbt) if has_lh else 0.0

    # CO SSA combined with OH quasi-equilibrium and the site balance.
    k_consumption = k_minus_1 + k_er
    rho = k1_co / k_consumption
    A = 1.0 + oh_ratio

    if has_lh:
        lam = k_lh * oh_ratio / k_consumption
        a = A * lam
        b = rho + A - lam
        discriminant = np.sqrt(b**2 + 4.0 * a)

        theta_empty = np.where(
            b >= 0.0,
            2.0 / (b + discriminant),
            (discriminant - b) / (2.0 * a),
        )
        theta_co = rho * theta_empty / (1.0 + lam * theta_empty)
    else:
        theta_empty = 1.0 / (A + rho)
        theta_co = rho * theta_empty

    theta_oh = oh_ratio * theta_empty

    pathway_rates = {}
    if has_er:
        pathway_rates["ER"] = k_er * theta_co
    if has_lh:
        pathway_rates["LH"] = k_lh * theta_co * theta_oh

    total_rate = sum(pathway_rates.values())

    expected = {
        "ln_rate": np.log(total_rate),
        "theta_empty": theta_empty,
        "theta_CO": theta_co,
        "theta_OH": theta_oh,
    }
    for name, rate in pathway_rates.items():
        expected[f"ln_rate_{name}"] = np.log(rate)
        expected[f"rate_fraction_{name}"] = rate / total_rate

    return expected


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model_name", ["CO_LH", "CO_ER", "CO_ER_LH"])
def test_pt100_reference_outputs(medium, model_name):
    inputs = make_inputs()
    parameters = make_parameters(model_name)

    result = evaluate_pt100_co(
        state=inputs,
        parameters=parameters,
        temperature_K=TEMPERATURE_K,
        medium=medium,
        model_name=model_name,
    )

    expected = reference_solution(inputs, medium, model_name)
    actual = {"ln_rate": result.ln_rate.eval()}
    actual.update({name: value.eval() for name, value in result.pointwise.items()})

    assert set(actual) == set(expected)

    for name, reference in expected.items():
        np.testing.assert_allclose(
            actual[name], reference, rtol=1e-10, atol=1e-12,
            err_msg=f"{medium}/{model_name}: {name}",
        )

    coverages = np.stack([actual[name] for name in ("theta_empty", "theta_CO", "theta_OH")])
    assert np.all(np.isfinite(coverages))
    assert np.all((coverages >= 0.0) & (coverages <= 1.0))
    np.testing.assert_allclose(coverages.sum(axis=0), 1.0, rtol=1e-10, atol=1e-10)

    fractions = [
        actual[name] for name in ("rate_fraction_ER", "rate_fraction_LH")
        if name in actual
    ]
    np.testing.assert_allclose(np.sum(fractions, axis=0), 1.0, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model_name", ["CO_LH", "CO_ER", "CO_ER_LH"])
def test_pt100_invalid_material_is_rejected(medium, model_name):
    inputs = replace(make_inputs(), materials=("Pt66Ru33",))

    with pytest.raises(ValueError, match="Pt100"):
        evaluate_pt100_co(
            inputs, make_parameters(model_name), TEMPERATURE_K, medium, model_name,
        )


def test_pt100_invalid_medium_is_rejected():
    with pytest.raises(ValueError, match="Unsupported medium"):
        evaluate_pt100_co(
            make_inputs(), make_parameters("CO_LH"), TEMPERATURE_K, "neutral", "CO_LH",
        )


def test_pt100_unknown_mechanism_is_rejected():
    with pytest.raises(ValueError, match="Unknown Pt100 mechanism"):
        evaluate_pt100_co(
            make_inputs(), make_parameters("CO_LH"), TEMPERATURE_K, "acidic", "CO_BF",
        )
