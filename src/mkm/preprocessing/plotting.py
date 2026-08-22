import matplotlib.pyplot as plt
import numpy as np


def _get_condition_data(data, material, C_KOH_M, CO_fraction):
    return data[
        (data["material"] == material)
        & (data["C_KOH_M"] == C_KOH_M)
        & (data["CO_mole_fraction"] == CO_fraction)
    ]


def _get_truncation_row(truncation, material, C_KOH_M, CO_fraction):
    rows = truncation[
        (truncation["material"] == material)
        & (truncation["C_KOH_M"] == C_KOH_M)
        & (truncation["CO_mole_fraction"] == CO_fraction)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one truncation record for {material}, {C_KOH_M} M KOH, "
            f"{CO_fraction} CO mole fraction; found {len(rows)}."
        )

    return rows.iloc[0]


def _format_condition_grid(axes, config):
    KOH_values = config["KOH_concentrations_M"]
    CO_values = config["CO_mole_fractions"]

    for row, CO_fraction in enumerate(CO_values):
        for col, C_KOH_M in enumerate(KOH_values):
            ax = axes[row, col]

            if row == 0:
                ax.set_title(f"{C_KOH_M:g} M KOH")

            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * CO_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                )

            ax.grid(alpha=0.20)


def _plot_rate_like_grid(
    full_replicates,
    selected_replicates,
    selected_summary,
    truncation,
    material,
    config,
    replicate_column,
    mean_column,
    sd_column,
    ylabel,
    title,
    log_y=False,
):
    KOH_values = config["KOH_concentrations_M"]
    CO_values = config["CO_mole_fractions"]

    fig, axes = plt.subplots(
        nrows=len(CO_values),
        ncols=len(KOH_values),
        figsize=(3.2 * len(KOH_values), 2.5 * len(CO_values)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, CO_fraction in enumerate(CO_values):
        for col, C_KOH_M in enumerate(KOH_values):
            ax = axes[row, col]

            full_condition = _get_condition_data(full_replicates, material, C_KOH_M, CO_fraction)
            selected_condition = _get_condition_data(selected_replicates, material, C_KOH_M, CO_fraction)
            summary_condition = _get_condition_data(selected_summary, material, C_KOH_M, CO_fraction).sort_values(
                "E_V_SHE"
            )

            truncation_row = _get_truncation_row(truncation, material, C_KOH_M, CO_fraction)

            cutoff_E = truncation_row["retained_min_E_V_SHE"]
            original_min_E = truncation_row["original_min_E_V_SHE"]

            if cutoff_E > original_min_E:
                ax.axvspan(original_min_E, cutoff_E, alpha=0.08)

            ax.axvline(cutoff_E, linestyle="--", linewidth=1.0, alpha=0.6)

            for replicate in config["replicates"]:
                full_curve = full_condition[full_condition["replicate"] == replicate].sort_values("E_V_SHE")
                selected_curve = selected_condition[selected_condition["replicate"] == replicate].sort_values("E_V_SHE")

                if not full_curve.empty:
                    ax.plot(full_curve["E_V_SHE"], full_curve[replicate_column], linewidth=1.0, alpha=0.18)

                if not selected_curve.empty:
                    ax.plot(selected_curve["E_V_SHE"], selected_curve[replicate_column], linewidth=1.2, alpha=0.45)

            potential = summary_condition["E_V_SHE"].to_numpy()
            mean = summary_condition[mean_column].to_numpy()
            sd = summary_condition[sd_column].to_numpy()

            lower = mean - sd
            upper = mean + sd

            if log_y:
                lower = np.where(lower > 0, lower, np.nan)

            ax.fill_between(potential, lower, upper, alpha=0.20, linewidth=0)
            ax.plot(potential, mean, linewidth=2.2)

            if log_y:
                ax.set_yscale("log")

    _format_condition_grid(axes, config)

    fig.suptitle(f"{material} — {title}", fontweight="bold")
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(ylabel)
    fig.tight_layout(rect=(0.03, 0.03, 0.96, 0.97))

    return fig


def _plot_alpha_grid(selected_replicates, selected_summary, truncation, material, config):
    KOH_values = config["KOH_concentrations_M"]
    CO_values = config["CO_mole_fractions"]

    fig, axes = plt.subplots(
        nrows=len(CO_values),
        ncols=len(KOH_values),
        figsize=(3.2 * len(KOH_values), 2.5 * len(CO_values)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, CO_fraction in enumerate(CO_values):
        for col, C_KOH_M in enumerate(KOH_values):
            ax = axes[row, col]

            replicate_condition = _get_condition_data(selected_replicates, material, C_KOH_M, CO_fraction)
            summary_condition = _get_condition_data(selected_summary, material, C_KOH_M, CO_fraction).sort_values(
                "E_V_SHE"
            )

            truncation_row = _get_truncation_row(truncation, material, C_KOH_M, CO_fraction)
            cutoff_E = truncation_row["retained_min_E_V_SHE"]

            ax.axvline(cutoff_E, linestyle="--", linewidth=1.0, alpha=0.6)

            for replicate in config["replicates"]:
                curve = replicate_condition[replicate_condition["replicate"] == replicate].sort_values("E_V_SHE")

                if curve.empty:
                    continue

                ax.plot(curve["E_V_SHE"], curve["alpha"], linewidth=1.0, alpha=0.35)

            potential = summary_condition["E_V_SHE"].to_numpy()
            mean = summary_condition["alpha_mean"].to_numpy()
            sd = summary_condition["alpha_sd"].to_numpy()

            ax.fill_between(potential, mean - sd, mean + sd, alpha=0.20, linewidth=0)
            ax.plot(potential, mean, linewidth=2.2)

    _format_condition_grid(axes, config)

    fig.suptitle(f"{material} — Transfer Coefficient", fontweight="bold")
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$\alpha$")
    fig.tight_layout(rect=(0.03, 0.03, 0.96, 0.97))

    return fig


def plot_agpd_experimental_summary(full_replicates, selected_replicates, selected_summary, truncation, material, config):
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")

    rate_figure = _plot_rate_like_grid(
        full_replicates=full_replicates,
        selected_replicates=selected_replicates,
        selected_summary=selected_summary,
        truncation=truncation,
        material=material,
        config=config,
        replicate_column="rate_s_inv",
        mean_column="rate_mean_s_inv",
        sd_column="rate_sd_s_inv",
        ylabel="TOF (s$^{-1}$)",
        title="Rate",
        log_y=False,
    )

    rate_logscale_figure = _plot_rate_like_grid(
        full_replicates=full_replicates,
        selected_replicates=selected_replicates,
        selected_summary=selected_summary,
        truncation=truncation,
        material=material,
        config=config,
        replicate_column="rate_s_inv",
        mean_column="rate_mean_s_inv",
        sd_column="rate_sd_s_inv",
        ylabel="TOF (s$^{-1}$)",
        title="Rate — Log Scale",
        log_y=True,
    )

    alpha_figure = _plot_alpha_grid(
        selected_replicates=selected_replicates,
        selected_summary=selected_summary,
        truncation=truncation,
        material=material,
        config=config,
    )

    return {"rate": rate_figure, "rate_logscale": rate_logscale_figure, "alpha": alpha_figure}


def plot_agpd_oh_order(delta_OH, material, config):
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")

    CO_values = config["CO_mole_fractions"]

    fig, axes = plt.subplots(
        nrows=len(CO_values),
        ncols=1,
        figsize=(6.0, 2.3 * len(CO_values)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, CO_fraction in enumerate(CO_values):
        ax = axes[row, 0]

        condition = delta_OH[
            (delta_OH["material"] == material) & (delta_OH["CO_mole_fraction"] == CO_fraction)
        ].sort_values("E_V_SHE")

        if condition.empty:
            raise ValueError(f"No OH-order data found for {material}, {100 * CO_fraction:g}% CO.")

        potential = condition["E_V_SHE"].to_numpy()
        mean = condition["delta_OH"].to_numpy()
        sd = condition["delta_OH_sd"].to_numpy()

        ax.fill_between(potential, mean - sd, mean + sd, alpha=0.20, linewidth=0)
        ax.plot(potential, mean, linewidth=2.0)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)

        ax.set_title(f"{100 * CO_fraction:g}% CO")
        ax.grid(alpha=0.20)

    fig.suptitle(f"{material} — OH Reaction Order", fontweight="bold")
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$\delta_{\mathrm{OH}}$")
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.97))

    return fig


def plot_agpd_co_order(delta_CO_replicates, delta_CO, material, config):
    if material not in config["materials"]:
        raise ValueError(f"Unknown material '{material}'.")

    KOH_values = config["KOH_concentrations_M"]
    CO_values = config["CO_mole_fractions"]

    CO_pairs = list(zip(CO_values[:-1], CO_values[1:]))

    fig, axes = plt.subplots(
        nrows=len(CO_pairs),
        ncols=len(KOH_values),
        figsize=(3.2 * len(KOH_values), 2.6 * len(CO_pairs)),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row, (lower_CO, upper_CO) in enumerate(CO_pairs):
        for col, C_KOH_M in enumerate(KOH_values):
            ax = axes[row, col]

            replicate_condition = delta_CO_replicates[
                (delta_CO_replicates["material"] == material)
                & (delta_CO_replicates["C_KOH_M"] == C_KOH_M)
                & (delta_CO_replicates["CO_lower_mole_fraction"] == lower_CO)
                & (delta_CO_replicates["CO_upper_mole_fraction"] == upper_CO)
            ]
            
            condition = delta_CO[
                (delta_CO["material"] == material)
                & (delta_CO["C_KOH_M"] == C_KOH_M)
                & (delta_CO["CO_lower_mole_fraction"] == lower_CO)
                & (delta_CO["CO_upper_mole_fraction"] == upper_CO)
            ].sort_values("E_V_SHE")

            if condition.empty:
                raise ValueError(
                    f"No CO-order data found for {material}, {C_KOH_M:g} M KOH, "
                    f"{100 * lower_CO:g}% to {100 * upper_CO:g}% CO."
                )

            potential = condition["E_V_SHE"].to_numpy()
            mean = condition["delta_CO"].to_numpy()
            sd = condition["delta_CO_sd"].to_numpy()

            for replicate in config["replicates"]:
                curve = replicate_condition[replicate_condition["replicate"] == replicate].sort_values("E_V_SHE")

                if curve.empty:
                    raise ValueError(
                        f"No paired CO-order data found for replicate {replicate}, {material}, "
                        f"{C_KOH_M:g} M KOH, {100 * lower_CO:g}% to {100 * upper_CO:g}% CO."
                    )

                ax.plot(curve["E_V_SHE"], curve["delta_CO"], linewidth=1.0, alpha=0.35)

            ax.fill_between(potential, mean - sd, mean + sd, alpha=0.20, linewidth=0)
            ax.plot(potential, mean, linewidth=2.0)
            ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.5)
            ax.grid(alpha=0.20)

            if row == 0:
                ax.set_title(f"{C_KOH_M:g} M KOH")

            if col == len(KOH_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * lower_CO:g}%"
                    r"$\rightarrow$"
                    f"{100 * upper_CO:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                )

    fig.suptitle(f"{material} — CO Reaction Order", fontweight="bold")
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel(r"$\delta_{\mathrm{CO}}$")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))

    return fig