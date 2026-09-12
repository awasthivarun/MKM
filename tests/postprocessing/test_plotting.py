from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

import mkm.postprocessing.plotting as plotting
from mkm.postprocessing.plotting import (
    PATHWAY_COLORS,
    plot_alpha_comparison,
    plot_alpha_overlay_koh,
    plot_alpha_overlay_pco,
    plot_delta_co_comparison,
    plot_delta_co_overlay_koh,
    plot_delta_co_overlay_pco,
    plot_delta_oh_comparison,
    plot_delta_oh_overlay_pco,
    plot_rate_overlay_koh,
    plot_rate_overlay_pco,
    plot_loo_diagnostics,
    plot_loo_pit_summary,
    plot_pointwise_loo_pit,
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_loo,
    plot_pointwise_variable,
    plot_pointwise_variables,
    plot_sampling_correlations,
    plot_sampling_pairs,
)


def test_plot_observation_grid_saves_model_and_predictive_rate_figures(tmp_path):
    frame = pd.DataFrame(
        {
            "model_point_id": [0, 1],
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "rate": [0.4, 0.7],
            "rate_model_hdi80_lower": [0.35, 0.55],
            "rate_model_hdi80_upper": [0.50, 0.80],
            "rate_model_hdi95_lower": [0.30, 0.50],
            "rate_model_median": [0.40, 0.70],
            "rate_model_hdi95_upper": [0.60, 0.90],
            "rate_predictive_hdi80_lower": [0.30, 0.50],
            "rate_predictive_hdi80_upper": [0.70, 1.00],
            "rate_predictive_hdi95_lower": [0.20, 0.40],
            "rate_predictive_median": [0.40, 0.70],
            "rate_predictive_hdi95_upper": [0.80, 1.20],
            "residual": [0.0, 0.0],
        }
    )

    for distribution in ("model", "predictive"):
        output_path = tmp_path / f"{distribution}.png"
        plot_observation_grid(
            frame,
            output_path,
            distribution=distribution,
            y_scale="linear",
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_observation_grid_saves_single_residual_definition(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "replicate": ["A", "A"],
            "E_V_SHE": [0.2, 0.3],
            "residual": [0.1, -0.1],
        }
    )
    output_path = tmp_path / "residual.png"

    plot_observation_grid(frame, output_path, residual=True, y_scale="linear")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_pointwise_variable_saves_figure(tmp_path):
    summary = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "E_V_SHE": [0.2, 0.3],
            "hdi80_lower": [0.30, 0.40],
            "hdi80_upper": [0.50, 0.60],
            "hdi95_lower": [0.20, 0.30],
            "median": [0.40, 0.50],
            "hdi95_upper": [0.60, 0.70],
        }
    )
    output_path = tmp_path / "theta_CO.png"

    plot_pointwise_variable(summary, "theta_CO", output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_parameter_posteriors_combines_physical_and_named_error_components(tmp_path):
    posterior = xr.Dataset(
        {
            "x": (("chain", "draw"), np.array([[0.8, 0.9, 1.0], [1.0, 1.1, 1.2]])),
            "sigma_rate_rel": (
                ("chain", "draw", "material"),
                np.array(
                    [
                        [[0.15, 0.20], [0.16, 0.21], [0.17, 0.22]],
                        [[0.14, 0.19], [0.15, 0.20], [0.16, 0.21]],
                    ]
                ),
            ),
        },
        coords={"material": ["Ag10Pd90", "Pd100"]},
    )
    specs = {
        "x": {"distribution": "normal", "mu": 0.0, "sigma": 2.0},
        "sigma_rate_rel": {
            "distribution": "lognormal",
            "median": 0.18,
            "log_sd": 0.75,
        },
    }
    output_path = tmp_path / "parameters.png"

    plot_parameter_posteriors(posterior, specs, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_pointwise_variables_combines_multiple_variables(tmp_path):
    summary = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1],
            "E_V_SHE": [0.2, 0.3],

            "rate_fraction_BF_median": [0.3, 0.4],
            "rate_fraction_BF_hdi80_lower": [0.25, 0.35],
            "rate_fraction_BF_hdi80_upper": [0.35, 0.45],
            "rate_fraction_BF_hdi95_lower": [0.2, 0.3],
            "rate_fraction_BF_hdi95_upper": [0.4, 0.5],

            "rate_fraction_ER_median": [0.6, 0.5],
            "rate_fraction_ER_hdi80_lower": [0.55, 0.45],
            "rate_fraction_ER_hdi80_upper": [0.65, 0.55],
            "rate_fraction_ER_hdi95_lower": [0.5, 0.4],
            "rate_fraction_ER_hdi95_upper": [0.7, 0.6],

            "rate_fraction_LH_median": [0.1, 0.1],
            "rate_fraction_LH_hdi80_lower": [0.075, 0.075],
            "rate_fraction_LH_hdi80_upper": [0.125, 0.125],
            "rate_fraction_LH_hdi95_lower": [0.05, 0.05],
            "rate_fraction_LH_hdi95_upper": [0.15, 0.15],
        }
    )
    output_path = tmp_path / "rate_fractions.png"

    created = plot_pointwise_variables(
        summary,
        ("rate_fraction_BF", "rate_fraction_ER", "rate_fraction_LH"),
        output_path,
        title="Pathway rate fractions",
        ylabel="rate fraction",
        colors={
            "rate_fraction_BF": PATHWAY_COLORS["BF"],
            "rate_fraction_ER": PATHWAY_COLORS["ER"],
            "rate_fraction_LH": PATHWAY_COLORS["LH"],
        },
    )

    assert created is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_pointwise_variables_overlays_scaled_mean_observed_rate(tmp_path):
    summary = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1, 0.1],
            "E_V_SHE": [0.2, 0.3, 0.4],
            "rate_fraction_BF_median": [0.2, 0.4, 0.3],
            "rate_fraction_BF_hdi80_lower": [0.15, 0.35, 0.25],
            "rate_fraction_BF_hdi80_upper": [0.25, 0.45, 0.35],
            "rate_fraction_BF_hdi95_lower": [0.1, 0.3, 0.2],
            "rate_fraction_BF_hdi95_upper": [0.3, 0.5, 0.4],
        }
    )
    observed = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0] * 6,
            "CO_mole_fraction": [0.1] * 6,
            "replicate": ["A", "B"] * 3,
            "E_V_SHE": [0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
            "rate": [1.0, 3.0, 4.0, 6.0, 2.0, 4.0],
        }
    )
    output_path = tmp_path / "rate_fractions_with_tof.png"

    created = plot_pointwise_variables(
        summary,
        ("rate_fraction_BF",),
        output_path,
        title="Pathway rate fractions",
        ylabel="rate fraction",
        observed_rates=observed,
    )

    assert created is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_loo_pit_summary_writes_one_two_row_file(tmp_path):
    loo_pit = np.linspace(0.01, 0.20, 100)
    output_path = tmp_path / "loo_pit.png"

    plot_loo_pit_summary(loo_pit, "CO_BF_ER_LH", output_path, context_label="Ag50Pd50")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_loo_diagnostics_writes_one_three_row_file(tmp_path):
    loo_result = SimpleNamespace(
        pareto_k=xr.DataArray(np.array([0.1, 0.2, 0.4, 0.6]), dims=("observation",)),
        good_k=0.7,
    )
    loo_pit = np.linspace(0.01, 0.20, 100)
    output_path = tmp_path / "loo_diagnostics.png"

    plot_loo_diagnostics(loo_result, loo_pit, "CO_BF_ER_LH", output_path, context_label="all materials")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_sampling_pairs_passes_list_var_names_to_arviz(monkeypatch, tmp_path):
    inference_data = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "x": (("chain", "draw"), np.array([[0.0, 0.1], [0.2, 0.3]])),
                    "y": (("chain", "draw"), np.array([[1.0, 1.1], [1.2, 1.3]])),
                }
            )
        }
    )
    captured = {}

    class DummyPlot:
        def add_title(self, title):
            captured["title"] = title

        def savefig(self, path, **kwargs):
            captured["path"] = path

    def fake_plot_pair(data, *, var_names, **kwargs):
        captured["var_names"] = var_names
        captured["aes"] = kwargs.get("aes")
        captured["aes_by_visuals"] = kwargs.get("aes_by_visuals")
        captured["divergence"] = kwargs.get("visuals", {}).get("divergence")
        return DummyPlot()

    monkeypatch.setattr(plotting.azp, "plot_pair", fake_plot_pair)

    plot_sampling_pairs(inference_data, ("x", "y"), tmp_path / "pairs.png")

    assert captured["var_names"] == ["x", "y"]


def test_plot_sampling_correlations_saves_numeric_heatmap(tmp_path):
    inference_data = xr.DataTree.from_dict(
        {
            "/posterior": xr.Dataset(
                {
                    "x": (("chain", "draw"), np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])),
                    "y": (("chain", "draw"), np.array([[1.0, 1.2, 1.4], [1.6, 1.8, 2.0]])),
                }
            )
        }
    )
    output_path = tmp_path / "correlations.png"
    plot_sampling_correlations(inference_data, ("x", "y"), output_path)
    assert output_path.exists()
    assert output_path.with_suffix(".svg").exists()


def test_pointwise_loo_pit_combines_metrics_with_twin_axis(tmp_path):
    loo = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0, 1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1, 0.1, 0.1],
            "replicate": ["A", "A", "B", "B"],
            "E_V_SHE": [0.0, 0.1, 0.0, 0.1],
            "elpd_loo": [1.0, 0.8, 0.9, 0.7],
        }
    )
    pit = loo.drop(columns="elpd_loo").copy()
    pit["loo_pit"] = [0.2, 0.4, 0.3, 0.5]
    output_path = tmp_path / "loo_pointwise_pit.png"

    plot_pointwise_loo_pit(loo, pit, output_path, context_label="Ag50Pd50")

    assert output_path.exists()
    assert output_path.with_suffix(".svg").exists()


def test_observable_plots_write_one_file_per_observable(tmp_path):
    potentials = [0.0, 0.1]

    alpha_rows = []
    for koh in (0.25, 0.5, 1.0):
        for co in (0.001, 0.01, 0.1, 1.0):
            for potential in potentials:
                alpha_rows.append(
                    {
                        "C_KOH_M": koh,
                        "CO_mole_fraction": co,
                        "E_V_SHE": potential,
                        "hdi80_lower": 0.25,
                        "hdi80_upper": 0.35,
                        "hdi95_lower": 0.2,
                        "median": 0.3,
                        "hdi95_upper": 0.4,
                        "alpha_mean": 0.31,
                        "alpha_sd": 0.02,
                    }
                )
    alpha_path = tmp_path / "alpha.png"
    plot_alpha_comparison(pd.DataFrame(alpha_rows), alpha_path, "Ag50Pd50")
    assert alpha_path.exists()

    oh_rows = []
    for co in (0.001, 0.01, 0.1, 1.0):
        for potential in potentials:
            oh_rows.append(
                {
                    "CO_mole_fraction": co,
                    "E_V_SHE": potential,
                    "hdi80_lower": 0.45,
                    "hdi80_upper": 0.55,
                    "hdi95_lower": 0.4,
                    "median": 0.5,
                    "hdi95_upper": 0.6,
                    "delta_OH": 0.52,
                    "delta_OH_sd": 0.03,
                }
            )
    oh_path = tmp_path / "delta_OH.png"
    plot_delta_oh_comparison(pd.DataFrame(oh_rows), oh_path, "Ag50Pd50")
    assert oh_path.exists()

    co_rows = []
    intervals = ((0.001, 0.01), (0.01, 0.1), (0.1, 1.0))
    for koh in (0.25, 0.5, 1.0):
        for lower, upper in intervals:
            for potential in potentials:
                co_rows.append(
                    {
                        "C_KOH_M": koh,
                        "CO_lower_mole_fraction": lower,
                        "CO_upper_mole_fraction": upper,
                        "E_V_SHE": potential,
                        "hdi80_lower": -0.15,
                        "hdi80_upper": 0.15,
                        "hdi95_lower": -0.2,
                        "median": 0.0,
                        "hdi95_upper": 0.2,
                        "delta_CO": 0.01,
                        "delta_CO_sd": 0.04,
                    }
                )
    co_path = tmp_path / "delta_CO.png"
    plot_delta_co_comparison(pd.DataFrame(co_rows), co_path, "Ag50Pd50")
    assert co_path.exists()



def test_all_material_observable_overlay_plots_write_files(tmp_path):
    materials = ("Pd100", "Ag10Pd90")
    potentials = (0.0, 0.1)

    alpha_rows = []
    for material in materials:
        for koh in (0.25, 1.0):
            for co in (0.001, 0.1):
                for potential in potentials:
                    alpha_rows.append(
                        {
                            "material": material,
                            "C_KOH_M": koh,
                            "CO_mole_fraction": co,
                            "E_V_SHE": potential,
                            "hdi80_lower": 0.25,
                            "hdi80_upper": 0.35,
                            "hdi95_lower": 0.2,
                            "median": 0.3,
                            "hdi95_upper": 0.4,
                            "alpha_mean": 0.31,
                            "alpha_sd": 0.02,
                        }
                    )
    alpha = pd.DataFrame(alpha_rows)
    for function, name in ((plot_alpha_overlay_pco, "alpha_pco"), (plot_alpha_overlay_koh, "alpha_koh")):
        output_path = tmp_path / f"{name}.png"
        function(alpha, output_path)
        assert output_path.exists()
        assert output_path.with_suffix(".svg").exists()

    oh_rows = []
    for material in materials:
        for co in (0.001, 0.1):
            for potential in potentials:
                oh_rows.append(
                    {
                        "material": material,
                        "CO_mole_fraction": co,
                        "E_V_SHE": potential,
                        "hdi80_lower": 0.45,
                        "hdi80_upper": 0.55,
                        "hdi95_lower": 0.4,
                        "median": 0.5,
                        "hdi95_upper": 0.6,
                        "delta_OH": 0.52,
                        "delta_OH_sd": 0.03,
                    }
                )
    oh_path = tmp_path / "delta_oh_pco.png"
    plot_delta_oh_overlay_pco(pd.DataFrame(oh_rows), oh_path)
    assert oh_path.exists()
    assert oh_path.with_suffix(".svg").exists()

    co_rows = []
    for material in materials:
        for koh in (0.25, 1.0):
            for lower, upper in ((0.001, 0.01), (0.01, 0.1)):
                for potential in potentials:
                    co_rows.append(
                        {
                            "material": material,
                            "C_KOH_M": koh,
                            "CO_lower_mole_fraction": lower,
                            "CO_upper_mole_fraction": upper,
                            "E_V_SHE": potential,
                            "hdi80_lower": -0.15,
                            "hdi80_upper": 0.15,
                            "hdi95_lower": -0.2,
                            "median": 0.0,
                            "hdi95_upper": 0.2,
                            "delta_CO": 0.01,
                            "delta_CO_sd": 0.04,
                        }
                    )
    delta_co = pd.DataFrame(co_rows)
    for function, name in ((plot_delta_co_overlay_pco, "delta_co_pco"), (plot_delta_co_overlay_koh, "delta_co_koh")):
        output_path = tmp_path / f"{name}.png"
        function(delta_co, output_path)
        assert output_path.exists()
        assert output_path.with_suffix(".svg").exists()

def test_rate_overlays_and_single_material_overlays_write_files(tmp_path):
    potentials = (0.0, 0.1)
    rows = []
    for replicate, offset in (("A", 0.0), ("B", 0.05)):
        for koh in (0.25, 1.0):
            for co in (0.001, 0.1):
                for potential in potentials:
                    median = 0.5 + potential + 0.2 * koh + 0.1 * co
                    rows.append(
                        {
                            "material": "Ag10Pd90",
                            "electrolyte_concentration_M": koh,
                            "CO_mole_fraction": co,
                            "replicate": replicate,
                            "E_V_SHE": potential,
                            "rate": median + offset,
                            "rate_model_median": median,
                            "rate_model_hdi80_lower": 0.9 * median,
                            "rate_model_hdi80_upper": 1.1 * median,
                            "rate_model_hdi95_lower": 0.8 * median,
                            "rate_model_hdi95_upper": 1.2 * median,
                        }
                    )
    frame = pd.DataFrame(rows)
    for function, name in ((plot_rate_overlay_pco, "rate_pco"), (plot_rate_overlay_koh, "rate_koh")):
        output_path = tmp_path / f"{name}.png"
        function(frame, output_path)
        assert output_path.exists()
        assert output_path.with_suffix(".svg").exists()

    alpha = pd.DataFrame(
        [
            {
                "material": "Ag10Pd90",
                "C_KOH_M": koh,
                "CO_mole_fraction": co,
                "E_V_SHE": potential,
                "hdi80_lower": 0.25,
                "hdi80_upper": 0.35,
                "hdi95_lower": 0.2,
                "median": 0.3,
                "hdi95_upper": 0.4,
                "alpha_mean": 0.31,
                "alpha_sd": 0.02,
            }
            for koh in (0.25, 1.0)
            for co in (0.001, 0.1)
            for potential in potentials
        ]
    )
    output_path = tmp_path / "single_material_alpha_overlay.png"
    plot_alpha_overlay_pco(alpha, output_path)
    assert output_path.exists()
    assert output_path.with_suffix(".svg").exists()


def test_pointwise_loo_plot_accepts_one_material_without_cross_material_stitching(tmp_path):
    frame = pd.DataFrame(
        {
            "electrolyte_concentration_M": [1.0, 1.0, 1.0, 1.0],
            "CO_mole_fraction": [0.1, 0.1, 0.1, 0.1],
            "replicate": ["A", "A", "B", "B"],
            "E_V_SHE": [0.0, 0.1, 0.0, 0.1],
            "elpd_loo": [1.0, 0.8, 0.9, 0.7],
        }
    )
    output_path = tmp_path / "loo_pointwise.png"
    plot_pointwise_loo(frame, "CO_BF_ER_LH", output_path, context_label="Ag50Pd50")
    assert output_path.exists()
