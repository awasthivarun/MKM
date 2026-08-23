from pathlib import Path

import matplotlib.pyplot as plt


def plot_observation_grid(
    observations,
    output_path: str | Path,
    residual=False,
):
    KOH_values = sorted(
        observations[
            "electrolyte_concentration_M"
        ].unique()
    )

    CO_values = sorted(
        observations["CO_mole_fraction"].unique()
    )

    fig, axes = plt.subplots(
        len(CO_values),
        len(KOH_values),
        figsize=(
            3.4 * len(KOH_values),
            2.5 * len(CO_values),
        ),
        sharex=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]

            condition = observations[
                (
                    observations[
                        "electrolyte_concentration_M"
                    ]
                    == c_koh
                )
                & (
                    observations["CO_mole_fraction"]
                    == co_fraction
                )
            ]

            for replicate, curve in condition.groupby(
                "replicate",
                sort=True,
            ):
                curve = curve.sort_values("E_V_SHE")

                if residual:
                    ax.plot(
                        curve["E_V_SHE"],
                        curve["residual_conditional"],
                        linewidth=1.2,
                        alpha=0.65,
                        label=replicate,
                    )
                else:
                    observed_line, = ax.plot(
                        curve["E_V_SHE"],
                        curve["ln_rate"],
                        linewidth=1.0,
                        alpha=0.45,
                        label=f"{replicate} observed",
                    )

                    ax.fill_between(
                        curve["E_V_SHE"],
                        curve["ln_rate_predictive_q025"],
                        curve["ln_rate_predictive_q975"],
                        color=observed_line.get_color(),
                        alpha=0.08,
                        linewidth=0,
                    )

                    ax.plot(
                        curve["E_V_SHE"],
                        curve["ln_rate_predictive_q50"],
                        color=observed_line.get_color(),
                        linewidth=1.5,
                    )

            if residual:
                ax.axhline(
                    0.0,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")

            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )

            ax.grid(alpha=0.20)

    if residual:
        fig.suptitle(
            "Conditional log-rate residuals"
        )
        fig.supylabel(
            "ln(rate) observed - posterior "
            "conditional median"
        )
    else:
        fig.suptitle(
            "Posterior predictive log rates"
        )
        fig.supylabel(
            "ln(rate / s$^{-1}$)"
        )

    fig.supxlabel("Potential (V vs SHE)")
    fig.tight_layout(
        rect=(0.04, 0.04, 0.96, 0.97)
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pointwise_variable(
    summary,
    variable_name,
    output_path: str | Path,
):
    KOH_values = sorted(
        summary[
            "electrolyte_concentration_M"
        ].unique()
    )

    CO_values = sorted(
        summary["CO_mole_fraction"].unique()
    )

    fig, axes = plt.subplots(
        len(CO_values),
        len(KOH_values),
        figsize=(
            3.4 * len(KOH_values),
            2.5 * len(CO_values),
        ),
        sharex=True,
        squeeze=False,
    )

    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]

            condition = summary[
                (
                    summary[
                        "electrolyte_concentration_M"
                    ]
                    == c_koh
                )
                & (
                    summary["CO_mole_fraction"]
                    == co_fraction
                )
            ].sort_values("E_V_SHE")

            ax.fill_between(
                condition["E_V_SHE"],
                condition["q025"],
                condition["q975"],
                alpha=0.20,
                linewidth=0,
            )

            ax.plot(
                condition["E_V_SHE"],
                condition["q50"],
                linewidth=2.0,
            )

            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")

            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )

            ax.grid(alpha=0.20)

    fig.suptitle(variable_name)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(variable_name)

    fig.tight_layout(
        rect=(0.04, 0.04, 0.96, 0.97)
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)