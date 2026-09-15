import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkm.preprocessing.plotting import (
    plot_agpd_alpha_overlay_koh,
    plot_agpd_alpha_overlay_pco,
    plot_agpd_co_order,
    plot_agpd_co_order_overlay_koh,
    plot_agpd_co_order_overlay_pco,
    plot_agpd_experimental_summary,
    plot_agpd_oh_order,
    plot_agpd_oh_order_overlay_pco,
    plot_agpd_rate_overlay_koh,
    plot_agpd_rate_overlay_pco,
    plot_agpd_second_order_composition,
    plot_agpd_second_order_difference,
    plot_agpd_second_order_overlay_koh,
    plot_agpd_second_order_overlay_pco,
    save_agpd_figure,
)


def _plot_data():
    materials = ["Pd100", "Ag10Pd90"]
    koh_values = [0.25, 0.5, 1.0]
    co_values = [0.001, 0.01, 0.1, 1.0]
    replicates = ["A", "B", "C"]
    potentials = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1]
    config = {
        "materials": materials,
        "KOH_concentrations_M": koh_values,
        "CO_mole_fractions": co_values,
        "replicates": replicates,
    }

    full_rows = []
    selected_rows = []
    summary_rows = []
    truncation_rows = []
    for material_index, material in enumerate(materials):
        for c_koh in koh_values:
            for co_fraction in co_values:
                truncation_rows.append(
                    {
                        "material": material,
                        "C_KOH_M": c_koh,
                        "CO_mole_fraction": co_fraction,
                        "retained_min_E_V_SHE": -0.1,
                        "original_min_E_V_SHE": -0.2,
                    }
                )
                for potential in potentials:
                    rates = []
                    alphas = []
                    for replicate_index, replicate in enumerate(replicates):
                        rate = 0.15 + 0.2 * (potential + 0.2) + 0.03 * c_koh + 0.01 * material_index
                        rate *= 1.0 + 0.03 * (replicate_index - 1)
                        alpha = 0.55 - 0.4 * potential + 0.01 * replicate_index - 0.02 * material_index
                        row = {
                            "material": material,
                            "C_KOH_M": c_koh,
                            "CO_mole_fraction": co_fraction,
                            "replicate": replicate,
                            "E_V_SHE": potential,
                            "rate_s_inv": rate,
                            "alpha": alpha,
                        }
                        full_rows.append(row)
                        if potential >= -0.1:
                            selected_rows.append(row)
                            rates.append(rate)
                            alphas.append(alpha)
                    if potential >= -0.1:
                        summary_rows.append(
                            {
                                "material": material,
                                "C_KOH_M": c_koh,
                                "CO_mole_fraction": co_fraction,
                                "E_V_SHE": potential,
                                "rate_mean_s_inv": float(np.mean(rates)),
                                "rate_sd_s_inv": float(np.std(rates, ddof=1)),
                                "alpha_mean": float(np.mean(alphas)),
                                "alpha_sd": float(np.std(alphas, ddof=1)),
                            }
                        )

    oh_rows = []
    for material_index, material in enumerate(materials):
        for co_fraction in co_values:
            for potential in potentials[1:]:
                oh_rows.append(
                    {
                        "material": material,
                        "CO_mole_fraction": co_fraction,
                        "E_V_SHE": potential,
                        "delta_OH": 0.8 - 0.2 * potential - 0.02 * material_index,
                        "delta_OH_sd": 0.05,
                    }
                )

    co_replicate_rows = []
    co_summary_rows = []
    for material_index, material in enumerate(materials):
        for c_koh in koh_values:
            for lower_co, upper_co in zip(co_values[:-1], co_values[1:], strict=True):
                for potential in potentials[1:]:
                    values = []
                    for replicate_index, replicate in enumerate(replicates):
                        value = 0.25 + 0.2 * potential + 0.02 * material_index + 0.01 * replicate_index
                        values.append(value)
                        co_replicate_rows.append(
                            {
                                "material": material,
                                "C_KOH_M": c_koh,
                                "CO_lower_mole_fraction": lower_co,
                                "CO_upper_mole_fraction": upper_co,
                                "replicate": replicate,
                                "E_V_SHE": potential,
                                "delta_CO": value,
                            }
                        )
                    co_summary_rows.append(
                        {
                            "material": material,
                            "C_KOH_M": c_koh,
                            "CO_lower_mole_fraction": lower_co,
                            "CO_upper_mole_fraction": upper_co,
                            "E_V_SHE": potential,
                            "delta_CO": float(np.mean(values)),
                            "delta_CO_sd": float(np.std(values, ddof=1)),
                        }
                    )

    return (
        config,
        pd.DataFrame(full_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(truncation_rows),
        pd.DataFrame(oh_rows),
        pd.DataFrame(co_replicate_rows),
        pd.DataFrame(co_summary_rows),
    )


def test_material_preprocessing_plots_render_and_save_png_svg(tmp_path):
    config, full, selected, summary, truncation, delta_oh, delta_co_replicates, delta_co = _plot_data()
    figures = plot_agpd_experimental_summary(full, selected, summary, truncation, "Ag10Pd90", config)
    figures["delta_OH"] = plot_agpd_oh_order(delta_oh, "Ag10Pd90", config)
    figures["delta_CO"] = plot_agpd_co_order(delta_co_replicates, delta_co, "Ag10Pd90", config)
    figures["second_order_difference"] = plot_agpd_second_order_difference(
        summary, delta_oh, "Ag10Pd90", config
    )

    for name, figure in figures.items():
        assert all(ax.get_legend() is None for ax in figure.axes)
        output_path = tmp_path / f"{name}.png"
        save_agpd_figure(figure, output_path)
        plt.close(figure)
        assert output_path.exists()
        assert output_path.with_suffix(".svg").exists()


def test_preprocessing_overlay_plots_render():
    config, _, _, summary, _, delta_oh, _, delta_co = _plot_data()
    figures = [
        plot_agpd_alpha_overlay_pco(summary, config),
        plot_agpd_alpha_overlay_koh(summary, config),
        plot_agpd_co_order_overlay_pco(delta_co, config),
        plot_agpd_co_order_overlay_koh(delta_co, config),
        plot_agpd_oh_order_overlay_pco(delta_oh, config),
        plot_agpd_rate_overlay_pco(summary, config),
        plot_agpd_rate_overlay_koh(summary, config),
        plot_agpd_second_order_overlay_pco(summary, delta_oh, config),
        plot_agpd_second_order_overlay_koh(summary, delta_oh, config),
    ]
    for figure in figures:
        assert figure.axes
        assert figure.legends
        data_lines = [line for ax in figure.axes for line in ax.lines if line.get_marker() == "o"]
        assert data_lines
        assert all(line.get_linestyle() in {"None", "none", ""} for line in data_lines)
        plt.close(figure)




def test_second_order_composition_uses_common_support():
    config, _, _, summary, _, delta_oh, _, _ = _plot_data()
    figure = plot_agpd_second_order_composition(summary, delta_oh, config)
    assert len(figure.axes) == 1
    assert not figure.legends
    assert len(figure.axes[0].containers) == len(config["materials"])
    plt.close(figure)
