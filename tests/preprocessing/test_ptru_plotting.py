
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkm.preprocessing import plotting as style
from mkm.preprocessing.ptru_plotting import (
    _condition_panels,
    plot_alpha_grid,
    plot_rate_grid,
)


def test_ptru_condition_panel_layouts():
    is_em, nrows, ncols, panels = _condition_panels([0.1, 0.25], [0.01, 0.1])
    assert not is_em
    assert (nrows, ncols) == (2, 2)
    assert panels == [
        (0, 0, 0.01, 0.1),
        (0, 1, 0.01, 0.25),
        (1, 0, 0.1, 0.1),
        (1, 1, 0.1, 0.25),
    ]

    is_em, nrows, ncols, panels = _condition_panels([0.1], [0.001, 0.01, 0.1])
    assert is_em
    assert (nrows, ncols) == (2, 2)
    assert panels == [
        (0, 0, 0.001, 0.1),
        (0, 1, 0.01, 0.1),
        (1, 0, 0.1, 0.1),
    ]


def test_ptru_rate_and_alpha_plots_save_png_svg(tmp_path):
    rows = []
    for concentration in (0.1, 0.25):
        for co_fraction in (0.01, 0.1):
            for replicate, scale in (("1", 1.0), ("2", 1.1)):
                for index, potential in enumerate((0.70, 0.71, 0.72, 0.73)):
                    rows.append({
                        "medium": "acidic",
                        "material": "Pt100",
                        "C_electrolyte_M": concentration,
                        "CO_mole_fraction": co_fraction,
                        "replicate": replicate,
                        "analysis_grid_index": index,
                        "E_V_SHE": potential,
                        "rate_s_inv": scale * np.exp(10 * (potential - 0.70)),
                        "alpha": 0.5 + 0.1 * index + 0.02 * (scale - 1),
                    })

    replicates = pd.DataFrame(rows)
    keys = ["material", "C_electrolyte_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE"]

    summary = replicates.groupby(keys, as_index=False).agg(
        alpha_mean=("alpha", "mean"),
        alpha_sd=("alpha", "std"),
        n_replicates=("replicate", "nunique"),
    )

    config = {
        "electrolyte_concentrations_M": [0.1, 0.25],
        "CO_mole_fractions": [0.01, 0.1],
    }
    cutoffs = {(0.1, 0.01): 0.71}

    figures = {
        "rate": plot_rate_grid(
            replicates, "Pt100", "acidic", False, config, cutoffs=cutoffs,
        ),
        "rate_logscale": plot_rate_grid(
            replicates, "Pt100", "acidic", True, config, cutoffs=cutoffs,
        ),
        "alpha": plot_alpha_grid(
            replicates, summary, "Pt100", "acidic", config, cutoffs=cutoffs,
        ),
    }

    for name, fig in figures.items():
        output_path = tmp_path / f"{name}.png"
        style.save_agpd_figure(fig, output_path)
        plt.close(fig)

        assert output_path.is_file()
        assert output_path.with_suffix(".svg").is_file()
