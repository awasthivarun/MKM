from ._shared import *
def plot_transition_state_drc(summary, model_name, output_path: str | Path):
    koh_values = sorted(summary["electrolyte_concentration_M"].unique())
    co_values = sorted(summary["CO_mole_fraction"].unique())
    controls = summary[["control", "label"]].drop_duplicates().itertuples(index=False)
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for control, label in controls:
        control_data = summary.loc[summary["control"] == control]
        for row, co_fraction in enumerate(co_values):
            for col, c_koh in enumerate(koh_values):
                ax = axes[row, col]
                condition = control_data[
                    (control_data["electrolyte_concentration_M"] == c_koh)
                    & (control_data["CO_mole_fraction"] == co_fraction)
                ].sort_values("E_V_SHE")
                _plot_summary_curve(
                    ax, condition, "E_V_SHE", color=PATHWAY_COLORS.get(control), label=label, linewidth=1.8
                )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8, alpha=0.55)
            ax.axhline(1.0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8, alpha=0.45)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    _add_axis_legend(axes[0, 0], loc="best")
    _finish_condition_grid(
        fig, axes, title="Transition-state degree of rate control", xlabel="Potential (V vs SHE)",
        ylabel=r"$X_{\mathbf{TS}}$", legend=False, left=0.075, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)
