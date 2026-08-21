import numpy as np
import pandas as pd
import pytest
import yaml

from mkm.inference.model import build_pymc_model
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.agpd_basic import available_agpd_models, build_agpd_mechanism, get_agpd_model_definition


def _load_config():
    with open("config/models/agpd_basic.yaml", "r") as file:
        return yaml.safe_load(file)


def _make_inputs(materials=("Ag10Pd90",)):
    records = []

    for material in materials:
        for replicate, offset in zip(["A", "B", "C"], [-0.05, 0.0, 0.05]):
            for grid_index in range(3):
                ln_rate = -2.0 + 3.0 * 0.01 * grid_index + offset

                records.append(
                    {
                        "material": material,
                        "C_KOH_M": 0.50,
                        "CO_mole_fraction": 0.01,
                        "replicate": replicate,
                        "analysis_grid_index": grid_index,
                        "E_V_SHE": 0.01 * grid_index,
                        "rate_s_inv": np.exp(ln_rate),
                        "ln_rate": ln_rate,
                    }
                )

    data = pd.DataFrame(records)

    model_data = build_model_data(selected_replicates=data, electrolyte_concentration_column="C_KOH_M")

    return build_model_input_arrays(model_data)


def test_agpd_registry_contains_models_of_interest():
    assert set(available_agpd_models()) == {"BF", "BF_LH", "CO_BF_ER_LH"}


def test_agpd_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown AgPd model"):
        get_agpd_model_definition("NOT_A_MODEL")


@pytest.mark.parametrize("model_name", ["BF", "BF_LH", "CO_BF_ER_LH"])
def test_ag10pd90_registered_models_build_with_finite_logp(model_name):
    config = _load_config()
    mechanism = build_agpd_mechanism(model_name=model_name, material="Ag10Pd90", config=config)

    built = build_pymc_model(
        inputs=_make_inputs(),
        mechanism=mechanism,
        sigma_prior_median=config["likelihood"]["sigma_prior_median"],
        sigma_prior_log_sd=config["likelihood"]["sigma_prior_log_sd"],
    )

    logp = built.model.compile_logp()(built.model.initial_point())

    assert np.isfinite(logp)


def test_agpd_registered_model_rejects_multiple_materials():
    config = _load_config()
    mechanism = build_agpd_mechanism(model_name="BF", material="Ag10Pd90", config=config)

    with pytest.raises(ValueError, match="exactly that one material"):
        build_pymc_model(inputs=_make_inputs(materials=("Ag10Pd90", "Ag25Pd75")), mechanism=mechanism)