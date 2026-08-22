import numpy as np
import pandas as pd
import pytest
import yaml

from mkm.model_data import build_model_data
from mkm.observable_maps import (
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
    evaluate_linear_observable_map,
)
from mkm.preprocessing.agpd_basic import (
    add_agpd_rates,
    add_agpd_transfer_coefficients,
    build_agpd_analysis_grid,
    calculate_agpd_co_order,
    calculate_agpd_oh_order,
    load_agpd_workbook,
    summarize_agpd_replicates,
    truncate_agpd_analysis,
)


@pytest.fixture(scope="module")
def agpd_contract_data():
    with open("config/preprocessing/agpd_basic.yaml", "r") as file:
        config = yaml.safe_load(file)

    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    analysis = add_agpd_rates(analysis, config)
    selected, _ = truncate_agpd_analysis(analysis, config)
    selected = add_agpd_transfer_coefficients(selected, config)
    summary = summarize_agpd_replicates(selected, config)

    model_data = build_model_data(selected_replicates=selected, electrolyte_concentration_column="C_KOH_M")

    mean_log_rate = (
        model_data.observations.groupby("model_point_id")["ln_rate"]
        .mean()
        .reindex(model_data.model_points["model_point_id"])
        .to_numpy()
    )

    return {
        "config": config,
        "selected": selected,
        "summary": summary,
        "model_data": model_data,
        "mean_log_rate": mean_log_rate,
    }


def test_real_agpd_model_data_contract(agpd_contract_data):
    selected = agpd_contract_data["selected"]
    model_data = agpd_contract_data["model_data"]

    assert len(model_data.observations) == len(selected)
    assert len(model_data.model_points) == (
        selected[["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index"]].drop_duplicates().shape[0]
    )
    assert len(model_data.conditions) == 12


def test_alpha_map_matches_preprocessing_alpha(agpd_contract_data):
    config = agpd_contract_data["config"]
    summary = agpd_contract_data["summary"]
    model_data = agpd_contract_data["model_data"]
    mean_log_rate = agpd_contract_data["mean_log_rate"]

    alpha_map = build_alpha_map(model_points=model_data.model_points, temperature_K=config["temperature_K"])

    mapped_alpha = alpha_map.outputs.merge(
        model_data.conditions,
        on="condition_id",
        how="left",
        validate="many_to_one",
    ).copy()

    mapped_alpha["alpha_mapped"] = evaluate_linear_observable_map(log_rate=mean_log_rate, observable_map=alpha_map)

    expected = summary[["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "alpha_mean"]].rename(
        columns={"C_KOH_M": "electrolyte_concentration_M"}
    )

    comparison = mapped_alpha.merge(
        expected,
        on=["material", "electrolyte_concentration_M", "CO_mole_fraction", "analysis_grid_index"],
        how="inner",
        validate="one_to_one",
    )

    assert len(comparison) == len(expected)
    np.testing.assert_allclose(comparison["alpha_mapped"], comparison["alpha_mean"], rtol=1e-10, atol=1e-10)


def test_oh_order_map_matches_preprocessing(agpd_contract_data):
    config = agpd_contract_data["config"]
    summary = agpd_contract_data["summary"]
    model_data = agpd_contract_data["model_data"]
    mean_log_rate = agpd_contract_data["mean_log_rate"]

    expected = calculate_agpd_oh_order(summary, config)

    order_map = build_log_slope_order_map(
        model_points=model_data.model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=config["KOH_concentrations_M"],
        group_columns=["material", "CO_mole_fraction"],
    )

    mapped = order_map.outputs.copy()
    mapped["delta_OH_mapped"] = evaluate_linear_observable_map(mean_log_rate, order_map)

    comparison = mapped.merge(
        expected,
        on=["material", "CO_mole_fraction", "analysis_grid_index"],
        how="inner",
        validate="one_to_one",
    )

    assert len(comparison) == len(expected)
    np.testing.assert_allclose(comparison["delta_OH_mapped"], comparison["delta_OH"], rtol=1e-12, atol=1e-12)


def test_co_order_map_matches_preprocessing(agpd_contract_data):
    config = agpd_contract_data["config"]
    selected = agpd_contract_data["selected"]
    model_data = agpd_contract_data["model_data"]
    mean_log_rate = agpd_contract_data["mean_log_rate"]

    expected = calculate_agpd_co_order(selected, config)

    order_map = build_adjacent_log_order_map(
        model_points=model_data.model_points,
        varying_column="CO_mole_fraction",
        varying_values=config["CO_mole_fractions"],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="CO_lower_mole_fraction",
        upper_value_column="CO_upper_mole_fraction",
    )

    mapped = order_map.outputs.copy()
    mapped["delta_CO_mapped"] = evaluate_linear_observable_map(mean_log_rate, order_map)

    expected = expected.rename(columns={"C_KOH_M": "electrolyte_concentration_M"})

    comparison = mapped.merge(
        expected,
        on=[
            "material",
            "electrolyte_concentration_M",
            "CO_lower_mole_fraction",
            "CO_upper_mole_fraction",
            "analysis_grid_index",
        ],
        how="inner",
        validate="one_to_one",
    )

    assert len(comparison) == len(expected)
    np.testing.assert_allclose(comparison["delta_CO_mapped"], comparison["delta_CO"], rtol=1e-12, atol=1e-12)