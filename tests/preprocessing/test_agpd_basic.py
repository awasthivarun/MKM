import numpy as np
import pandas as pd
import pytest
import yaml

from mkm.preprocessing.agpd_basic import (
    _parse_co_mole_fraction,
    _parse_current_column,
    add_agpd_rates,
    add_agpd_transfer_coefficients,
    build_agpd_analysis_grid,
    calculate_agpd_co_order,
    calculate_agpd_oh_order,
    calculate_agpd_co_order_replicates,
    load_agpd_workbook,
    summarize_agpd_replicates,
    truncate_agpd_analysis,
)


@pytest.fixture
def config():
    with open("config/preprocessing/agpd_basic.yaml", "r") as file:
        return yaml.safe_load(file)


def test_parse_co_mole_fraction(config):
    expected = config["CO_mole_fractions"]

    assert _parse_co_mole_fraction("0.1% CO", expected) == pytest.approx(0.001)
    assert _parse_co_mole_fraction("1% CO", expected) == pytest.approx(0.01)
    assert _parse_co_mole_fraction("10% CO", expected) == pytest.approx(0.10)
    assert _parse_co_mole_fraction("100% CO", expected) == pytest.approx(1.00)


def test_parse_current_column(config):
    assert _parse_current_column("1M lnj uA/cm2 (A)", config) == (1.0, "A")
    assert _parse_current_column("0.5M lnj uA/cm2 (B)", config) == (0.5, "B")
    assert _parse_current_column("0.25M lnj uA/cm2 (C)", config) == (0.25, "C")


def test_load_agpd_workbook(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    df = load_agpd_workbook(path, config)

    assert set(df["material"]) == {"Ag10Pd90"}
    assert set(df["C_KOH_M"]) == {0.25, 0.50, 1.00}
    assert set(df["CO_mole_fraction"]) == {0.001, 0.01, 0.10, 1.00}
    assert set(df["replicate"]) == {"A", "B", "C"}


def test_workbook_has_expected_number_of_curves(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    df = load_agpd_workbook(path, config)

    curve_columns = ["material", "C_KOH_M", "CO_mole_fraction", "replicate"]
    curves = df.groupby(curve_columns)

    expected_curves = len(config["KOH_concentrations_M"]) * len(config["CO_mole_fractions"]) * len(config["replicates"])

    assert curves.ngroups == expected_curves

    curve_sizes = curves.size()

    assert np.all(curve_sizes.to_numpy() == config["potential"]["expected_points"])


def test_current_reconstruction(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    df = load_agpd_workbook(path, config)
    reconstructed = np.exp(df["ln_j_uA_cm2"].to_numpy())

    np.testing.assert_allclose(df["j_uA_cm2"].to_numpy(), reconstructed)


def test_standardized_rows_are_unique(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    df = load_agpd_workbook(path, config)

    key_columns = ["material", "C_KOH_M", "CO_mole_fraction", "replicate", "point_index"]

    assert not df.duplicated(key_columns).any()


def test_build_agpd_analysis_grid(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    assert not analysis.empty
    assert set(analysis["material"]) == {"Ag10Pd90"}
    assert set(analysis["C_KOH_M"]) == {0.25, 0.50, 1.00}
    assert set(analysis["CO_mole_fraction"]) == {0.001, 0.01, 0.10, 1.00}
    assert set(analysis["replicate"]) == {"A", "B", "C"}


def test_analysis_grid_spacing(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    spacing = config["analysis_grid"]["spacing_V"]
    condition_columns = ["material", "C_KOH_M", "CO_mole_fraction", "replicate"]

    for _, curve in analysis.groupby(condition_columns, sort=False):
        potential = curve["E_V_SHE"].to_numpy()
        np.testing.assert_allclose(np.diff(potential), spacing, rtol=0, atol=1e-12)


def test_analysis_grid_is_shared_across_replicates(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    condition_columns = ["material", "C_KOH_M", "CO_mole_fraction"]

    for _, condition in analysis.groupby(condition_columns, sort=False):
        replicate_grids = []

        for replicate in config["replicates"]:
            replicate_df = condition[condition["replicate"] == replicate]
            replicate_grids.append(replicate_df["E_V_SHE"].to_numpy())

        for grid in replicate_grids[1:]:
            np.testing.assert_array_equal(grid, replicate_grids[0])


def test_analysis_grid_stays_inside_raw_replicate_domains(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    condition_columns = ["material", "C_KOH_M", "CO_mole_fraction", "replicate"]

    raw_groups = standardized.groupby(condition_columns, sort=False)
    analysis_groups = analysis.groupby(condition_columns, sort=False)

    for key, analysis_curve in analysis_groups:
        raw_curve = raw_groups.get_group(key)

        raw_min = raw_curve["E_V_SHE"].min()
        raw_max = raw_curve["E_V_SHE"].max()

        analysis_min = analysis_curve["E_V_SHE"].min()
        analysis_max = analysis_curve["E_V_SHE"].max()

        assert analysis_min >= raw_min - 1e-12
        assert analysis_max <= raw_max + 1e-12


def test_analysis_grid_matches_global_lattice(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    spacing = config["analysis_grid"]["spacing_V"]
    origin = config["analysis_grid"]["origin_V"]

    expected_potential = origin + analysis["analysis_grid_index"].to_numpy() * spacing

    np.testing.assert_allclose(analysis["E_V_SHE"].to_numpy(), expected_potential, rtol=0, atol=1e-12)


def test_analysis_grid_current_reconstruction(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    reconstructed = np.exp(analysis["ln_j_uA_cm2"].to_numpy())

    np.testing.assert_allclose(analysis["j_uA_cm2"].to_numpy(), reconstructed)
    assert np.all(analysis["j_uA_cm2"].to_numpy() > 0)


def test_analysis_grid_has_unique_rows(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)

    key_columns = ["material", "C_KOH_M", "CO_mole_fraction", "replicate", "analysis_grid_index"]

    assert not analysis.duplicated(key_columns).any()


def test_agpd_rate_normalization(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)

    denominator = (
        config["rate_normalization"]["electrons_per_CO_oxidation"]
        * config["rate_normalization"]["site_charge_density_uC_cm2_Pd"]
    )

    expected_rate = rates["j_uA_cm2"].to_numpy() / denominator

    np.testing.assert_allclose(rates["rate_s_inv"].to_numpy(), expected_rate)


def test_agpd_log_rate_normalization(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)

    denominator = (
        config["rate_normalization"]["electrons_per_CO_oxidation"]
        * config["rate_normalization"]["site_charge_density_uC_cm2_Pd"]
    )

    expected_log_rate = rates["ln_j_uA_cm2"].to_numpy() - np.log(denominator)

    np.testing.assert_allclose(rates["ln_rate"].to_numpy(), expected_log_rate)


def test_rate_and_log_rate_are_consistent(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)

    np.testing.assert_allclose(np.log(rates["rate_s_inv"].to_numpy()), rates["ln_rate"].to_numpy())


def test_agpd_rates_are_positive_and_finite(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)

    rate = rates["rate_s_inv"].to_numpy()
    log_rate = rates["ln_rate"].to_numpy()

    assert np.all(np.isfinite(rate))
    assert np.all(rate > 0)
    assert np.all(np.isfinite(log_rate))


def test_agpd_replicate_summary(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    summary = summarize_agpd_replicates(rates, config)

    assert not summary.empty
    assert np.all(summary["n_replicates"].to_numpy() == len(config["replicates"]))
    assert len(summary) == len(rates) // len(config["replicates"])


def test_agpd_replicate_means(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    summary = summarize_agpd_replicates(rates, config)

    group_columns = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE"]

    expected = (
        rates.groupby(group_columns, sort=False)
        .agg(rate_mean=("rate_s_inv", "mean"), ln_rate_mean=("ln_rate", "mean"))
        .reset_index()
    )

    np.testing.assert_allclose(summary["rate_mean_s_inv"], expected["rate_mean"])
    np.testing.assert_allclose(summary["ln_rate_mean"], expected["ln_rate_mean"])


def test_agpd_replicate_standard_deviations(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    summary = summarize_agpd_replicates(rates, config)

    group_columns = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE"]

    expected = (
        rates.groupby(group_columns, sort=False)
        .agg(rate_sd=("rate_s_inv", "std"), ln_rate_sd=("ln_rate", "std"))
        .reset_index()
    )

    np.testing.assert_allclose(summary["rate_sd_s_inv"], expected["rate_sd"])
    np.testing.assert_allclose(summary["ln_rate_sd"], expected["ln_rate_sd"])


def test_rate_sd_scales_with_current_sd(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    summary = summarize_agpd_replicates(rates, config)

    denominator = (
        config["rate_normalization"]["electrons_per_CO_oxidation"]
        * config["rate_normalization"]["site_charge_density_uC_cm2_Pd"]
    )

    np.testing.assert_allclose(summary["rate_sd_s_inv"], summary["j_sd_uA_cm2"] / denominator)


def test_log_rate_sd_equals_log_current_sd(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    summary = summarize_agpd_replicates(rates, config)

    np.testing.assert_allclose(summary["ln_rate_sd"], summary["ln_j_sd"])


def test_agpd_summary_rows_are_unique(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    summary = summarize_agpd_replicates(rates, config)

    key_columns = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index"]

    assert not summary.duplicated(key_columns).any()


def test_agpd_transfer_coefficients(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    rates_with_alpha = add_agpd_transfer_coefficients(rates, config)

    assert "alpha" in rates_with_alpha.columns
    assert np.all(np.isfinite(rates_with_alpha["alpha"].to_numpy()))
    assert len(rates_with_alpha) == len(rates)


def test_agpd_transfer_coefficients_do_not_change_rate_data(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    rates_with_alpha = add_agpd_transfer_coefficients(rates, config)

    columns = ["j_uA_cm2", "ln_j_uA_cm2", "rate_s_inv", "ln_rate"]

    for column in columns:
        np.testing.assert_array_equal(rates_with_alpha[column].to_numpy(), rates[column].to_numpy())


def test_agpd_summary_contains_alpha_statistics(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    rates = add_agpd_transfer_coefficients(rates, config)
    summary = summarize_agpd_replicates(rates, config)

    assert "alpha_mean" in summary.columns
    assert "alpha_sd" in summary.columns
    assert np.all(np.isfinite(summary["alpha_mean"].to_numpy()))
    assert np.all(np.isfinite(summary["alpha_sd"].to_numpy()))


def test_agpd_alpha_summary_matches_replicate_data(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    rates = add_agpd_transfer_coefficients(rates, config)
    summary = summarize_agpd_replicates(rates, config)

    group_columns = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE"]

    expected = (
        rates.groupby(group_columns, sort=False)
        .agg(alpha_mean_expected=("alpha", "mean"), alpha_sd_expected=("alpha", "std"))
        .reset_index()
    )

    np.testing.assert_allclose(summary["alpha_mean"].to_numpy(), expected["alpha_mean_expected"].to_numpy())
    np.testing.assert_allclose(summary["alpha_sd"].to_numpy(), expected["alpha_sd_expected"].to_numpy())


def test_agpd_alpha_is_unchanged_by_tof_normalization(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    rates_with_alpha = add_agpd_transfer_coefficients(rates, config)

    from mkm.preprocessing.observables import add_transfer_coefficients

    alpha_from_log_current = add_transfer_coefficients(
        data=rates,
        group_columns=["material", "C_KOH_M", "CO_mole_fraction", "replicate"],
        temperature_K=config["temperature_K"],
        potential_column="E_V_SHE",
        log_rate_column="ln_j_uA_cm2",
        output_column="alpha_from_ln_j",
    )

    np.testing.assert_allclose(
        rates_with_alpha["alpha"].to_numpy(),
        alpha_from_log_current["alpha_from_ln_j"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def _make_truncation_test_data():
    potential = np.array([-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04])
    grid_index = np.arange(-3, 5)

    rates = {
        "A": [5.0e-4, 9.0e-4, 1.3e-3, 1.0e-3, 1.5e-3, 2.2e-3, 8.2e-3, 8.0e-4],
        "B": [4.0e-4, 8.0e-4, 1.2e-3, 9.0e-4, 1.4e-3, 2.0e-3, 8.0e-3, 7.0e-4],
        "C": [6.0e-4, 1.0e-3, 1.4e-3, 1.1e-3, 1.6e-3, 2.4e-3, 8.4e-3, 9.0e-4],
    }

    frames = []

    for replicate, replicate_rates in rates.items():
        frames.append(
            pd.DataFrame(
                {
                    "material": "test",
                    "C_KOH_M": 0.25,
                    "CO_mole_fraction": 0.001,
                    "replicate": replicate,
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": potential,
                    "rate_s_inv": replicate_rates,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def test_low_rate_truncation_uses_sustained_pre_peak_crossing(config):
    test_config = config.copy()
    test_config["truncation"] = {
        "low_potential": {"method": "rate_threshold", "threshold_s_inv": 1.0e-3, "require_all_replicates": True},
        "high_potential": {"method": "none"},
    }

    data = _make_truncation_test_data()
    truncated, cutoffs = truncate_agpd_analysis(data, test_config)

    assert len(cutoffs) == 1
    np.testing.assert_allclose(cutoffs.loc[0, "retained_min_E_V_SHE"], 0.01)
    assert cutoffs.loc[0, "low_cut_analysis_grid_index"] == 1
    np.testing.assert_allclose(truncated["E_V_SHE"].min(), 0.01)


def test_post_peak_rate_below_threshold_is_retained(config):
    test_config = config.copy()
    test_config["truncation"] = {
        "low_potential": {"method": "rate_threshold", "threshold_s_inv": 1.0e-3, "require_all_replicates": True},
        "high_potential": {"method": "none"},
    }

    data = _make_truncation_test_data()
    truncated, cutoffs = truncate_agpd_analysis(data, test_config)

    np.testing.assert_allclose(cutoffs.loc[0, "peak_E_V_SHE"], 0.03)

    post_peak = truncated[truncated["E_V_SHE"] == 0.04]

    assert len(post_peak) == 3
    assert np.all(post_peak["rate_s_inv"].to_numpy() < 1.0e-3)


def test_truncation_produces_contiguous_potential_window(config):
    test_config = config.copy()
    test_config["truncation"] = {
        "low_potential": {"method": "rate_threshold", "threshold_s_inv": 1.0e-3, "require_all_replicates": True},
        "high_potential": {"method": "none"},
    }

    data = _make_truncation_test_data()
    truncated, _ = truncate_agpd_analysis(data, test_config)

    expected_potential = np.array([0.01, 0.02, 0.03, 0.04])

    for replicate in ["A", "B", "C"]:
        curve = truncated[truncated["replicate"] == replicate].sort_values("E_V_SHE")
        np.testing.assert_array_equal(curve["E_V_SHE"].to_numpy(), expected_potential)


def test_low_rate_truncation_preserves_clean_low_potential_data(config):
    test_config = config.copy()
    test_config["truncation"] = {
        "low_potential": {"method": "rate_threshold", "threshold_s_inv": 1.0e-3, "require_all_replicates": True},
        "high_potential": {"method": "none"},
    }

    data = _make_truncation_test_data()
    data["rate_s_inv"] = data["rate_s_inv"] + 1.0e-2

    truncated, cutoffs = truncate_agpd_analysis(data, test_config)

    assert len(truncated) == len(data)
    np.testing.assert_allclose(cutoffs.loc[0, "retained_min_E_V_SHE"], -0.03)


def test_truncation_records_condition_provenance(config):
    test_config = config.copy()
    test_config["truncation"] = {
        "low_potential": {"method": "rate_threshold", "threshold_s_inv": 1.0e-3, "require_all_replicates": True},
        "high_potential": {"method": "none"},
    }

    data = _make_truncation_test_data()
    _, cutoffs = truncate_agpd_analysis(data, test_config)

    assert cutoffs.loc[0, "low_potential_method"] == "rate_threshold"
    assert cutoffs.loc[0, "high_potential_method"] == "none"
    np.testing.assert_allclose(cutoffs.loc[0, "threshold_s_inv"], 1.0e-3)
    assert cutoffs.loc[0, "n_points_original"] == 8
    assert cutoffs.loc[0, "n_points_retained"] == 4


def test_real_agpd_truncation_satisfies_low_potential_rule(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    rates = add_agpd_rates(analysis, config)
    truncated, cutoffs = truncate_agpd_analysis(rates, config)

    threshold = config["truncation"]["low_potential"]["threshold_s_inv"]
    condition_columns = ["material", "C_KOH_M", "CO_mole_fraction"]

    for condition, cutoff in cutoffs.groupby(condition_columns, sort=False):
        cutoff = cutoff.iloc[0]

        condition_data = truncated[
            (truncated["material"] == condition[0])
            & (truncated["C_KOH_M"] == condition[1])
            & (truncated["CO_mole_fraction"] == condition[2])
        ]

        pre_peak = condition_data[condition_data["E_V_SHE"] <= cutoff["peak_E_V_SHE"]]
        minimum_rate = pre_peak.groupby("analysis_grid_index")["rate_s_inv"].min()

        assert np.all(minimum_rate.to_numpy() >= threshold)


def _build_selected_agpd_summary(path, config):
    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    analysis = add_agpd_rates(analysis, config)
    selected, _ = truncate_agpd_analysis(analysis, config)
    selected = add_agpd_transfer_coefficients(selected, config)

    return summarize_agpd_replicates(selected, config)


def _build_selected_agpd_replicates(path, config):
    standardized = load_agpd_workbook(path, config)
    analysis = build_agpd_analysis_grid(standardized, config)
    analysis = add_agpd_rates(analysis, config)
    selected, _ = truncate_agpd_analysis(analysis, config)

    return selected


def test_agpd_oh_order_structure(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    summary = _build_selected_agpd_summary(path, config)
    delta_OH = calculate_agpd_oh_order(summary, config)

    assert not delta_OH.empty
    assert set(delta_OH["CO_mole_fraction"]) == set(config["CO_mole_fractions"])
    assert set(delta_OH["n_order_conditions"]) == {len(config["KOH_concentrations_M"])}
    assert np.all(np.isfinite(delta_OH["delta_OH"]))
    assert np.all(np.isfinite(delta_OH["delta_OH_sd"]))


def test_agpd_oh_order_uses_common_selected_grid(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    summary = _build_selected_agpd_summary(path, config)
    delta_OH = calculate_agpd_oh_order(summary, config)

    for CO_fraction in config["CO_mole_fractions"]:
        condition = summary[summary["CO_mole_fraction"] == CO_fraction]

        expected_indices = None

        for C_KOH_M in config["KOH_concentrations_M"]:
            indices = set(condition[condition["C_KOH_M"] == C_KOH_M]["analysis_grid_index"])

            if expected_indices is None:
                expected_indices = indices
            else:
                expected_indices &= indices

        observed_indices = set(delta_OH[delta_OH["CO_mole_fraction"] == CO_fraction]["analysis_grid_index"])

        assert observed_indices == expected_indices


def test_agpd_co_order_structure(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    selected = _build_selected_agpd_replicates(path, config)
    delta_CO = calculate_agpd_co_order(selected, config)

    assert not delta_CO.empty

    expected_pairs = {(0.001, 0.01), (0.01, 0.10), (0.10, 1.00)}
    observed_pairs = set(zip(delta_CO["CO_lower_mole_fraction"], delta_CO["CO_upper_mole_fraction"]))

    assert observed_pairs == expected_pairs
    assert np.all(np.isfinite(delta_CO["delta_CO"]))
    assert np.all(np.isfinite(delta_CO["delta_CO_sd"]))


def test_agpd_co_order_uses_pair_specific_selected_grid(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    selected = _build_selected_agpd_replicates(path, config)
    delta_CO = calculate_agpd_co_order(selected, config)

    CO_values = config["CO_mole_fractions"]

    for C_KOH_M in config["KOH_concentrations_M"]:
        concentration_data = selected[selected["C_KOH_M"] == C_KOH_M]

        for i in range(len(CO_values) - 1):
            lower = CO_values[i]
            upper = CO_values[i + 1]

            lower_indices = set(
                concentration_data[concentration_data["CO_mole_fraction"] == lower]["analysis_grid_index"]
            )
            upper_indices = set(
                concentration_data[concentration_data["CO_mole_fraction"] == upper]["analysis_grid_index"]
            )

            expected_indices = lower_indices & upper_indices

            observed_indices = set(
                delta_CO[
                    (delta_CO["C_KOH_M"] == C_KOH_M)
                    & (delta_CO["CO_lower_mole_fraction"] == lower)
                    & (delta_CO["CO_upper_mole_fraction"] == upper)
                ]["analysis_grid_index"]
            )

            assert observed_indices == expected_indices


def test_agpd_co_order_replicates_preserve_pairing(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    selected = _build_selected_agpd_replicates(path, config)
    delta_CO_replicates = calculate_agpd_co_order_replicates(selected, config)

    assert set(delta_CO_replicates["replicate"]) == set(config["replicates"])

    key_columns = [
        "material",
        "C_KOH_M",
        "CO_lower_mole_fraction",
        "CO_upper_mole_fraction",
        "analysis_grid_index",
    ]

    counts = delta_CO_replicates.groupby(key_columns)["replicate"].nunique()

    assert np.all(counts.to_numpy() == len(config["replicates"]))


def test_agpd_co_order_replicate_matches_direct_log_rate_difference(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    selected = _build_selected_agpd_replicates(path, config)
    delta_CO_replicates = calculate_agpd_co_order_replicates(selected, config)

    row = delta_CO_replicates.iloc[0]

    lower = selected[
        (selected["material"] == row["material"])
        & (selected["C_KOH_M"] == row["C_KOH_M"])
        & (selected["replicate"] == row["replicate"])
        & (selected["CO_mole_fraction"] == row["CO_lower_mole_fraction"])
        & (selected["analysis_grid_index"] == row["analysis_grid_index"])
    ].iloc[0]

    upper = selected[
        (selected["material"] == row["material"])
        & (selected["C_KOH_M"] == row["C_KOH_M"])
        & (selected["replicate"] == row["replicate"])
        & (selected["CO_mole_fraction"] == row["CO_upper_mole_fraction"])
        & (selected["analysis_grid_index"] == row["analysis_grid_index"])
    ].iloc[0]

    expected = (upper["ln_rate"] - lower["ln_rate"]) / np.log(
        row["CO_upper_mole_fraction"] / row["CO_lower_mole_fraction"]
    )

    np.testing.assert_allclose(row["delta_CO"], expected, rtol=1e-12, atol=1e-12)


def test_agpd_co_order_summary_matches_paired_replicates(config):
    path = "data/raw/AgPd_COOx_basic/Ag10Pd90_current_densities.xlsx"

    selected = _build_selected_agpd_replicates(path, config)
    delta_CO_replicates = calculate_agpd_co_order_replicates(selected, config)
    delta_CO = calculate_agpd_co_order(selected, config)

    group_columns = [
        "material",
        "C_KOH_M",
        "CO_lower_mole_fraction",
        "CO_upper_mole_fraction",
        "analysis_grid_index",
        "E_V_SHE",
    ]

    expected = (
        delta_CO_replicates.groupby(group_columns, sort=False)
        .agg(delta_CO_expected=("delta_CO", "mean"), delta_CO_sd_expected=("delta_CO", "std"))
        .reset_index()
    )

    np.testing.assert_allclose(delta_CO["delta_CO"], expected["delta_CO_expected"])
    np.testing.assert_allclose(delta_CO["delta_CO_sd"], expected["delta_CO_sd_expected"])