from ._shared import *
def plot_alpha_comparison(comparison, output_path: str | Path, material):
    """Plot alpha for all KOH/CO conditions in one 4 x 3-style grid."""
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            data = comparison[
                (comparison["C_KOH_M"] == c_koh) & (comparison["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            if not data.empty:
                _plot_summary_curve(ax, data, "E_V_SHE", color=posterior_color, linewidth=2.0)
                ax.errorbar(
                    data["E_V_SHE"], data["alpha_mean"], yerr=data["alpha_sd"], fmt="o",
                    color=experiment_color, markersize=3.4, linewidth=0.9, capsize=1.7, zorder=2.5,
                )
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: transfer coefficient", xlabel="Potential (V vs SHE)",
        ylabel="Transfer coefficient",
        legend=False, left=0.075, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_delta_oh_comparison(comparison, output_path: str | Path, material):
    """Plot OH reaction order as one vertical row per CO fraction."""
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), 1,
        figsize=(OH_ORDER_FIGURE_WIDTH_IN, MATERIAL_PANEL_HEIGHT_IN * len(co_values) + OH_ORDER_EXTRA_HEIGHT_IN),
        sharex=True, sharey=True, squeeze=False
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, co_fraction in enumerate(co_values):
        ax = axes[row, 0]
        data = comparison.loc[comparison["CO_mole_fraction"] == co_fraction].sort_values("E_V_SHE")
        _plot_summary_curve(ax, data, "E_V_SHE", color=posterior_color, linewidth=2.0)
        ax.errorbar(
            data["E_V_SHE"], data["delta_OH"], yerr=data["delta_OH_sd"], fmt="o",
            color=experiment_color, markersize=3.4, linewidth=0.9, capsize=1.7, zorder=2.5,
        )
        ax.text(
            1.02, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
            rotation=-90, va="center", fontweight=FONT_WEIGHT,
        )
        _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: OH reaction order", xlabel="Potential (V vs SHE)", ylabel="OH order",
        legend=False, left=0.20, right=0.84, bottom=0.06, top=0.925, hspace=0.08,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_delta_co_comparison(comparison, output_path: str | Path, material):
    """Plot adjacent CO reaction orders as CO-interval rows by KOH columns."""
    koh_values = sorted(comparison["C_KOH_M"].unique())
    intervals = (
        comparison[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]]
        .drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"])
    )
    interval_values = list(intervals.itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        len(interval_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(interval_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, (lower_co, upper_co) in enumerate(interval_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            data = comparison[
                (comparison["C_KOH_M"] == c_koh)
                & (comparison["CO_lower_mole_fraction"] == lower_co)
                & (comparison["CO_upper_mole_fraction"] == upper_co)
            ].sort_values("E_V_SHE")
            if not data.empty:
                _plot_summary_curve(ax, data, "E_V_SHE", color=posterior_color, linewidth=2.0)
                ax.errorbar(
                    data["E_V_SHE"], data["delta_CO"], yerr=data["delta_CO_sd"], fmt="o",
                    color=experiment_color, markersize=3.4, linewidth=0.9, capsize=1.7, zorder=2.5,
                )
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * lower_co:g}% → {100 * upper_co:g}% CO",
                    transform=ax.transAxes, rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: adjacent CO reaction order", xlabel="Potential (V vs SHE)",
        ylabel="CO order", legend=False, left=0.075, right=0.91, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_second_order_difference(comparison, output_path: str | Path, material):
    """Plot d(alpha)/dE - d(delta_OH)/dE using posterior draws and experimental point estimates."""
    data = comparison.loc[comparison["observable"] == "delta2"].copy()
    if data.empty:
        raise ValueError("No delta2 second-order points were provided.")
    koh_values = sorted(data["C_KOH_M"].dropna().unique())
    co_values = sorted(data["CO_mole_fraction"].dropna().unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    posterior_color, experiment_color = _material_plot_colors(material)
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = data[
                (data["C_KOH_M"] == c_koh) & (data["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            if not condition.empty:
                _plot_summary_curve(ax, condition, "E_V_SHE", color=posterior_color, linewidth=2.0)
                ax.scatter(
                    condition["E_V_SHE"], condition["experimental"], color=experiment_color,
                    s=18, zorder=2.5,
                )
            ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, alpha=0.65)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _finish_condition_grid(
        fig, axes, title=f"{material}: second-order kinetic difference", xlabel="Potential (V vs SHE)",
        ylabel=r"$d\alpha/dE-d\delta_{\mathbf{OH}}/dE$ / V$^{-1}$", legend=False,
        left=0.08, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def _overlay_title(base_title, materials):
    return f"{base_title} across materials" if len(materials) > 1 else base_title


def _summarize_rate_overlay_condition(data):
    if data.empty:
        return data
    required = {
        "E_V_SHE", "rate", "rate_model_median", "rate_model_hdi80_lower", "rate_model_hdi80_upper",
        "rate_model_hdi95_lower", "rate_model_hdi95_upper",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Rate overlay is missing required columns: {missing}.")
    grouped = data.groupby("E_V_SHE", as_index=False).agg(
        experimental_mean=("rate", "mean"),
        experimental_sd=("rate", "std"),
        rate_model_median=("rate_model_median", "first"),
        rate_model_hdi80_lower=("rate_model_hdi80_lower", "first"),
        rate_model_hdi80_upper=("rate_model_hdi80_upper", "first"),
        rate_model_hdi95_lower=("rate_model_hdi95_lower", "first"),
        rate_model_hdi95_upper=("rate_model_hdi95_upper", "first"),
    )
    grouped["experimental_sd"] = grouped["experimental_sd"].fillna(0.0)
    return grouped.sort_values("E_V_SHE")


def _plot_rate_overlay_panel(ax, data, *, color):
    summary = _summarize_rate_overlay_condition(data)
    if summary.empty:
        return
    ax.errorbar(
        summary["E_V_SHE"], summary["experimental_mean"], yerr=summary["experimental_sd"], fmt="o",
        color=color, markersize=2.8, linewidth=0.65, capsize=1.2, alpha=0.72, zorder=2.5,
    )
    _plot_summary_curve(ax, summary, "E_V_SHE", prefix="rate_model", color=color, linewidth=1.65)


def _overlay_condition_legend(fig, condition_labels, *, n_conditions, y=0.952):
    shades = [_mix_with_white("#4D4D4D", amount) for amount in np.linspace(0.58, 0.0, n_conditions)]
    handles = [Line2D([0], [0], color=color, linewidth=2.0) for color in shades]
    legend = fig.legend(
        handles, condition_labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(handles),
        columnspacing=0.8, handlelength=1.5, prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
    )
    _style_legend(legend)
    return legend

def _plot_overlay_panel(ax, data, *, color, observed_column, observed_sd_column=None):
    if data.empty:
        return
    _plot_summary_curve(ax, data, "E_V_SHE", color=color, linewidth=1.65)
    if observed_sd_column is None:
        ax.scatter(data["E_V_SHE"], data[observed_column], color=color, s=12, zorder=2.5)
    else:
        ax.errorbar(
            data["E_V_SHE"], data[observed_column], yerr=data[observed_sd_column], fmt="o",
            color=color, markersize=2.8, linewidth=0.65, capsize=1.2, zorder=2.5,
        )

def _finish_overlay_grid(fig, axes, *, title, ylabel, condition_labels, row_labels=None, wspace=0.09):
    _set_presentation_axes(axes)
    for ax in np.asarray(axes, dtype=object).flat:
        _style_axis(ax, potential_x=True)
    if row_labels is not None:
        for row, label in enumerate(row_labels):
            axes[row, -1].text(
                1.0, 0.5, label, transform=axes[row, -1].transAxes, rotation=-90,
                va="center", fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE,
            )
    legend = _overlay_condition_legend(fig, condition_labels, n_conditions=len(condition_labels), y=0.955)
    fig.subplots_adjust(left=0.05, right=0.965, bottom=0.07, top=0.875, hspace=0.09, wspace=wspace)
    _add_fixed_gap_global_labels(fig, axes, xlabel="Potential (V vs SHE)", ylabel=ylabel)
    _position_row_labels(fig, axes)
    _position_legend_above_column_titles(fig, legend, axes)
    _add_fixed_gap_global_title(fig, axes, title, legend=legend)

def plot_alpha_overlay_pco(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(koh_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(koh_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, c_koh in enumerate(koh_values):
            for color, co_fraction in zip(colors, co_values, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["C_KOH_M"] == c_koh)
                    & (comparison["CO_mole_fraction"] == co_fraction)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="alpha_mean", observed_sd_column="alpha_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Transfer coefficient", materials), ylabel="Transfer coefficient",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_alpha_overlay_koh(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(co_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, co_fraction in enumerate(co_values):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["CO_mole_fraction"] == co_fraction)
                    & (comparison["C_KOH_M"] == c_koh)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="alpha_mean", observed_sd_column="alpha_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Transfer coefficient", materials), ylabel="Transfer coefficient",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_delta_co_overlay_pco(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    intervals = list(
        comparison[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]].drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"]).itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(
        len(koh_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(koh_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(intervals))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, c_koh in enumerate(koh_values):
            for color, (lower_co, upper_co) in zip(colors, intervals, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["C_KOH_M"] == c_koh)
                    & (comparison["CO_lower_mole_fraction"] == lower_co)
                    & (comparison["CO_upper_mole_fraction"] == upper_co)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="delta_CO", observed_sd_column="delta_CO_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Adjacent CO reaction order", materials), ylabel="CO order",
        condition_labels=[f"{100 * lo:g}–{100 * hi:g}% CO" for lo, hi in intervals],
        row_labels=[f"{value:g} M KOH" for value in koh_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_delta_co_overlay_koh(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    koh_values = sorted(comparison["C_KOH_M"].unique())
    intervals = list(
        comparison[["CO_lower_mole_fraction", "CO_upper_mole_fraction"]].drop_duplicates()
        .sort_values(["CO_lower_mole_fraction", "CO_upper_mole_fraction"]).itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(
        len(intervals), len(materials),
        figsize=_grid_figsize(
            len(materials), len(intervals), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for row, (lower_co, upper_co) in enumerate(intervals):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = comparison[
                    (comparison["material"] == material) & (comparison["C_KOH_M"] == c_koh)
                    & (comparison["CO_lower_mole_fraction"] == lower_co)
                    & (comparison["CO_upper_mole_fraction"] == upper_co)
                ].sort_values("E_V_SHE")
                _plot_overlay_panel(
                    axes[row, col], data, color=color, observed_column="delta_CO", observed_sd_column="delta_CO_sd"
                )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Adjacent CO reaction order", materials), ylabel="CO order",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * lo:g}–{100 * hi:g}% CO" for lo, hi in intervals],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_delta_oh_overlay_pco(comparison, output_path: str | Path):
    materials = _ordered_materials(comparison["material"].unique())
    co_values = sorted(comparison["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        1, len(materials),
        figsize=(OVERLAY_PANEL_WIDTH_IN * len(materials) + OVERLAY_EXTRA_WIDTH_IN, OVERLAY_SINGLE_ROW_HEIGHT_IN),
        sharex=True, sharey=True, squeeze=False
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight="bold")
        for color, co_fraction in zip(colors, co_values, strict=True):
            data = comparison[
                (comparison["material"] == material) & (comparison["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            _plot_overlay_panel(
                axes[0, col], data, color=color, observed_column="delta_OH", observed_sd_column="delta_OH_sd"
            )
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("OH reaction order", materials), ylabel="OH order",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_rate_overlay_pco(observations, output_path: str | Path):
    materials = _ordered_materials(observations["material"].unique())
    koh_values = sorted(observations["electrolyte_concentration_M"].unique())
    co_values = sorted(observations["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(koh_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(koh_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(co_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight=FONT_WEIGHT)
        for row, c_koh in enumerate(koh_values):
            for color, co_fraction in zip(colors, co_values, strict=True):
                data = observations[
                    (observations["material"] == material)
                    & (observations["electrolyte_concentration_M"] == c_koh)
                    & (observations["CO_mole_fraction"] == co_fraction)
                ]
                _plot_rate_overlay_panel(axes[row, col], data, color=color)
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Model rate", materials), ylabel=r"rate / s$^{-1}$",
        condition_labels=[f"{100 * value:g}% CO" for value in co_values],
        row_labels=[f"{value:g} M KOH" for value in koh_values], wspace=0.24,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_rate_overlay_koh(observations, output_path: str | Path):
    materials = _ordered_materials(observations["material"].unique())
    koh_values = sorted(observations["electrolyte_concentration_M"].unique())
    co_values = sorted(observations["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(materials),
        figsize=_grid_figsize(
            len(materials), len(co_values), panel_width=OVERLAY_PANEL_WIDTH_IN,
            panel_height=OVERLAY_PANEL_HEIGHT_IN, extra_width=OVERLAY_EXTRA_WIDTH_IN,
            extra_height=OVERLAY_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    for col, material in enumerate(materials):
        colors = _condition_colors(material, len(koh_values))
        axes[0, col].set_title(material, pad=TEXT_GAP_PT, fontweight=FONT_WEIGHT)
        for row, co_fraction in enumerate(co_values):
            for color, c_koh in zip(colors, koh_values, strict=True):
                data = observations[
                    (observations["material"] == material)
                    & (observations["CO_mole_fraction"] == co_fraction)
                    & (observations["electrolyte_concentration_M"] == c_koh)
                ]
                _plot_rate_overlay_panel(axes[row, col], data, color=color)
    _finish_overlay_grid(
        fig, axes, title=_overlay_title("Model rate", materials), ylabel=r"rate / s$^{-1}$",
        condition_labels=[f"{value:g} M KOH" for value in koh_values],
        row_labels=[f"{100 * value:g}% CO" for value in co_values], wspace=0.24,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


