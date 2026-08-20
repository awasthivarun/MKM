from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mkm.likelihood_diagnostics import (
    calculate_centered_log_residuals,
    calculate_pooled_log_rate_sd,
    summarize_material_dispersion,
    summarize_point_dispersion,
)
from mkm.model_data import (
    build_model_data,
)


REPO_ROOT = Path(
    __file__
).resolve().parents[1]

SELECTED_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "AgPd_COOx_basic"
    / "analysis"
    / "AgPd_COOx_basic_selected.parquet"
)

FIGURE_DIR = (
    REPO_ROOT
    / "figures"
    / "likelihood_diagnostics"
    / "AgPd_COOx_basic"
)


def main():
    selected = pd.read_parquet(
        SELECTED_PATH
    )

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column=(
            "C_KOH_M"
        ),
    )

    point_summary = (
        summarize_point_dispersion(
            model_data
        )
    )

    residuals = (
        calculate_centered_log_residuals(
            model_data
        )
    )

    material_summary = (
        summarize_material_dispersion(
            model_data
        )
    )

    pooled_sd = (
        calculate_pooled_log_rate_sd(
            model_data
        )
    )

    print(
        "\nGlobal pooled log-rate SD:"
    )
    print(
        f"{pooled_sd:.6f}"
    )

    print(
        "\nMaterial-level pooled "
        "log-rate SD:"
    )
    print(
        material_summary.to_string(
            index=False
        )
    )

    print(
        "\nPointwise log-rate SD "
        "quantiles:"
    )

    print(
        point_summary[
            "ln_rate_sd"
        ]
        .quantile(
            [
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.scatter(
        point_summary[
            "ln_rate_mean"
        ],
        point_summary[
            "ln_rate_sd"
        ],
        s=10,
        alpha=0.35,
    )

    ax.set_xlabel(
        "Mean ln(TOF / s$^{-1}$)"
    )

    ax.set_ylabel(
        "Replicate SD of ln(TOF)"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ln_rate_sd_vs_mean.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.scatter(
        point_summary[
            "E_V_SHE"
        ],
        point_summary[
            "ln_rate_sd"
        ],
        s=10,
        alpha=0.35,
    )

    ax.set_xlabel(
        "Potential (V vs SHE)"
    )

    ax.set_ylabel(
        "Replicate SD of ln(TOF)"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "ln_rate_sd_vs_potential.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.hist(
        residuals[
            "ln_rate_centered_residual"
        ],
        bins=50,
    )

    ax.set_xlabel(
        "Within-point centered "
        "ln(TOF) residual"
    )

    ax.set_ylabel(
        "Count"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "centered_log_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    for material in model_data.conditions["material"].unique():
        material_residuals = residuals[
            residuals["material"] == material
        ]

        KOH_values = sorted(
            material_residuals[
                "electrolyte_concentration_M"
            ].unique()
        )

        CO_values = sorted(
            material_residuals[
                "CO_mole_fraction"
            ].unique()
        )

        fig, axes = plt.subplots(
            nrows=len(CO_values),
            ncols=len(KOH_values),
            figsize=(
                3.2 * len(KOH_values),
                2.5 * len(CO_values),
            ),
            sharex=True,
            sharey=False,
            squeeze=False,
        )

        for row, CO_fraction in enumerate(CO_values):
            for col, C_KOH_M in enumerate(KOH_values):
                ax = axes[row, col]

                condition = material_residuals[
                    (
                        material_residuals[
                            "electrolyte_concentration_M"
                        ]
                        == C_KOH_M
                    )
                    & (
                        material_residuals[
                            "CO_mole_fraction"
                        ]
                        == CO_fraction
                    )
                ]

                for replicate in sorted(
                    condition["replicate"].unique()
                ):
                    curve = (
                        condition[
                            condition["replicate"]
                            == replicate
                        ]
                        .sort_values("E_V_SHE")
                    )

                    ax.plot(
                        curve["E_V_SHE"],
                        curve[
                            "ln_rate_centered_residual"
                        ],
                        linewidth=1.2,
                        label=replicate,
                    )

                ax.axhline(
                    0.0,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )

                ax.grid(
                    alpha=0.2,
                )

                if row == 0:
                    ax.set_title(
                        f"{C_KOH_M:g} M KOH"
                    )

                if col == len(KOH_values) - 1:
                    ax.text(
                        1.04,
                        0.5,
                        f"{100 * CO_fraction:g}% CO",
                        transform=ax.transAxes,
                        rotation=-90,
                        va="center",
                    )

        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper right",
        )

        fig.suptitle(
            f"{material} — Within-point replicate residuals"
        )

        fig.supxlabel(
            "Potential (V vs SHE)"
        )

        fig.supylabel(
            "ln(TOF) - point mean ln(TOF)"
        )

        fig.tight_layout(
            rect=(0.03, 0.03, 0.96, 0.96)
        )

        fig.savefig(
            FIGURE_DIR
            / f"{material}_replicate_residuals.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

if __name__ == "__main__":
    main()