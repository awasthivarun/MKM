from ._shared import *
def plot_loo_comparison(compare_table, output_path: str | Path):
    frame = compare_table.set_index("model")

    pc = azp.plot_compare(
        frame,
        relative_scale=True,
        rotated=True,
        backend="matplotlib",
        visuals={"similar_line": True},
    )
    pc.add_title("PSIS-LOO model comparison")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_pareto_k(loo_result, model_name, output_path: str | Path):
    pc = azp.plot_khat(
        loo_result,
        threshold=float(loo_result.good_k),
        backend="matplotlib",
        visuals={"hlines": True, "bin_text": True},
    )
    pc.add_title(f"Pareto-k diagnostics: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_pointwise_elpd_difference(frame, numerator_model, denominator_model, output_path: str | Path):
    comparison = f"{numerator_model}_minus_{denominator_model}"
    data = frame.loc[frame["comparison"] == comparison].copy()
    if data.empty:
        raise ValueError(f"No pointwise ELPD comparison found for '{comparison}'.")
    koh_values = sorted(data["electrolyte_concentration_M"].unique())
    co_values = sorted(data["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_PANEL_WIDTH_IN,
            panel_height=LOO_PANEL_HEIGHT_IN, extra_height=LOO_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    max_abs = float(np.max(np.abs(data["elpd_difference"])))
    y_limit = 1.05 * max_abs if max_abs > 0 else 1.0
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = data[
                (data["electrolyte_concentration_M"] == c_koh)
                & (data["CO_mole_fraction"] == co_fraction)
            ]
            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["elpd_difference"], marker="o", markersize=2.5,
                    linewidth=1.0, alpha=0.8, label=replicate,
                )
            ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.55, color=REFERENCE_COLOR)
            ax.set_ylim(-y_limit, y_limit)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    legend = axes[0, 0].legend(
        title="replicate", prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND}
    )
    _style_legend(legend)
    _finish_condition_grid(
        fig, axes, title="Pointwise PSIS-LOO difference", xlabel="Potential (V vs SHE)",
        ylabel=f"Pointwise ELPD: {numerator_model} - {denominator_model}", legend=False,
        left=0.075, right=0.93, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_pointwise_loo(pointwise, model_name, output_path: str | Path, context_label=None):
    """Plot observation-level PSIS-LOO contributions for one material."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_PANEL_WIDTH_IN,
            panel_height=LOO_PANEL_HEIGHT_IN, extra_height=LOO_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = pointwise[
                (pointwise["electrolyte_concentration_M"] == c_koh)
                & (pointwise["CO_mole_fraction"] == co_fraction)
            ]
            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["elpd_loo"], marker="o", markersize=2.5,
                    linewidth=1.0, alpha=0.8, label=replicate,
                )
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    legend = axes[0, 0].legend(
        title="replicate", prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND}
    )
    _style_legend(legend)
    prefix = f"{context_label}: " if context_label else ""
    _finish_condition_grid(
        fig, axes, title=f"{prefix}Pointwise PSIS-LOO", xlabel="Potential (V vs SHE)",
        ylabel="Pointwise PSIS-LOO ELPD", legend=False, left=0.075, right=0.93, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_pointwise_loo_pit(pointwise_loo, pointwise_pit, output_path: str | Path, context_label=None):
    """Plot pointwise PSIS-LOO ELPD and LOO-PIT together, pairing replicates by color."""
    required_loo = {"electrolyte_concentration_M", "CO_mole_fraction", "replicate", "E_V_SHE", "elpd_loo"}
    required_pit = {"electrolyte_concentration_M", "CO_mole_fraction", "replicate", "E_V_SHE", "loo_pit"}
    missing_loo = sorted(required_loo - set(pointwise_loo.columns))
    missing_pit = sorted(required_pit - set(pointwise_pit.columns))
    if missing_loo or missing_pit:
        raise ValueError(f"Combined LOO/PIT plot is missing columns: LOO={missing_loo}, PIT={missing_pit}.")

    koh_values = sorted(pointwise_loo["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise_loo["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_POINTWISE_PANEL_WIDTH_IN,
            panel_height=LOO_POINTWISE_PANEL_HEIGHT_IN, extra_width=LOO_POINTWISE_EXTRA_WIDTH_IN,
            extra_height=LOO_POINTWISE_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    twin_axes = np.empty_like(axes, dtype=object)
    replicates = sorted(set(pointwise_loo["replicate"].astype(str)) | set(pointwise_pit["replicate"].astype(str)))
    color_map = {replicate: DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index, replicate in enumerate(replicates)}

    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        for row, co_fraction in enumerate(co_values):
            for col, c_koh in enumerate(koh_values):
                ax = axes[row, col]
                ax_pit = ax.twinx()
                twin_axes[row, col] = ax_pit
                loo_condition = pointwise_loo[
                    (pointwise_loo["electrolyte_concentration_M"] == c_koh)
                    & (pointwise_loo["CO_mole_fraction"] == co_fraction)
                ]
                pit_condition = pointwise_pit[
                    (pointwise_pit["electrolyte_concentration_M"] == c_koh)
                    & (pointwise_pit["CO_mole_fraction"] == co_fraction)
                ]
                for replicate in replicates:
                    color = color_map[replicate]
                    loo_curve = loo_condition.loc[
                        loo_condition["replicate"].astype(str) == replicate
                    ].sort_values("E_V_SHE")
                    pit_curve = pit_condition.loc[
                        pit_condition["replicate"].astype(str) == replicate
                    ].sort_values("E_V_SHE")
                    if not loo_curve.empty:
                        ax.plot(
                            loo_curve["E_V_SHE"], loo_curve["elpd_loo"], color=color,
                            linewidth=1.0, marker="o", markersize=2.2, alpha=0.85,
                        )
                    if not pit_curve.empty:
                        ax_pit.plot(
                            pit_curve["E_V_SHE"], pit_curve["loo_pit"], color=color,
                            linewidth=1.0, linestyle="--", alpha=0.80,
                        )
                ax_pit.set_ylim(0.0, 1.0)
                ax_pit.set_yticks([0.0, 0.5, 1.0])
                ax_pit.grid(False)
                if col != len(koh_values) - 1:
                    ax_pit.tick_params(axis="y", right=False, labelright=False)
                else:
                    for label in ax_pit.get_yticklabels():
                        label.set_fontweight(FONT_WEIGHT)
                        label.set_fontsize(FONT_SIZE_TICK)
                if row == 0:
                    ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
                if col == len(koh_values) - 1:
                    ax.text(
                        1.17, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                        rotation=-90, va="center", fontweight=FONT_WEIGHT,
                    )
                _style_axis(ax, potential_x=True)

        legend_handles = [
            Line2D([0], [0], color=DEFAULT_COLORS[0], linewidth=1.5, marker="o", markersize=3, label="PSIS-LOO ELPD"),
            Line2D([0], [0], color=DEFAULT_COLORS[0], linewidth=1.5, linestyle="--", label="LOO-PIT"),
        ]
        legend = axes[0, 0].legend(
            handles=legend_handles, loc="best", ncol=1,
            prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND},
        )
        _style_legend(legend)
        prefix = f"{context_label}: " if context_label else ""
        fig.subplots_adjust(left=0.075, right=0.91, bottom=0.10, top=0.90, hspace=0.14, wspace=0.22)
        _add_fixed_gap_global_labels(
            fig, axes, xlabel="Potential (V vs SHE)", ylabel="Pointwise PSIS-LOO ELPD"
        )
        _position_row_labels(fig, axes)
        _add_fixed_gap_right_ylabel(fig, twin_axes[:, -1], axes[:, -1], ylabel="LOO-PIT")
        _add_fixed_gap_global_title(fig, axes, f"{prefix}Pointwise PSIS-LOO and LOO-PIT")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)


def _plot_loo_pit_axis(ax, loo_pit, *, coverage, title):
    values = np.asarray(loo_pit, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("LOO-PIT values are empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("LOO-PIT values contain non-finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("LOO-PIT values must lie inside [0, 1].")
    if coverage:
        values = 2.0 * np.abs(values - 0.5)
    distribution = xr.Dataset({"rate_observed": ("observation", values)})
    ecdf = distribution.azstats.ecdf(dim=["observation"], pit=True, npoints=len(values))
    uniformity_result = distribution.azstats.uniformity_test(dim=["observation"], method="pot_c")
    if len(uniformity_result) < 2:
        raise ValueError("Uniformity test did not return p-values and pointwise contributions.")
    p_values = uniformity_result[0]
    shapley_values = uniformity_result[1]
    ecdf_values = ecdf["rate_observed"]
    x = np.asarray(ecdf_values.sel(plot_axis="x"), dtype=float)
    y = np.asarray(ecdf_values.sel(plot_axis="y"), dtype=float)
    p_value = float(np.asarray(p_values["rate_observed"], dtype=float).reshape(-1)[0])
    shapley = np.asarray(shapley_values["rate_observed"], dtype=float).reshape(-1)
    alpha = 0.05
    expected_max = np.sqrt(np.log(2.0 / alpha) / (2.0 * len(values))) * 1.3
    actual_max = float(np.max(np.abs(y)))
    epsilon = max(expected_max, actual_max)
    suspicious = (shapley > 0.0) & (p_value < alpha)
    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.65)
    ax.step(x, y, where="pre", color=DEFAULT_COLORS[0], linewidth=1.5)
    if np.any(suspicious):
        ax.scatter(x[suspicious], y[suspicious], color=DEFAULT_COLORS[1], s=22, marker="x", linewidths=1.2)
    ax.text(
        0.01, 0.92, f"p={p_value:.2f} (α={alpha:.2f})", transform=ax.transAxes,
        va="top", fontweight=FONT_WEIGHT,
    )
    ax.set_ylim(-epsilon, epsilon)
    ax.set_ylabel(r"$\Delta$ ECDF", fontweight="bold")
    ax.set_title(title, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
    if coverage:
        ax.set_xticks([0.0, 0.5, 1.0], labels=["0", "50", "100"])
        ax.set_xlabel("ETI %", fontweight="bold")
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel("PIT", fontweight="bold")
    _nice_linear_ticks(ax.yaxis)
    _bold_axis_text(ax)
    ax.grid(False)


def _plot_pareto_k_axis(ax, loo_result, *, title):
    pareto_k = loo_result.pareto_k
    if isinstance(pareto_k, xr.Dataset):
        if len(pareto_k.data_vars) != 1:
            raise ValueError("Pareto-k dataset must contain exactly one variable.")
        pareto_k = pareto_k[next(iter(pareto_k.data_vars))]
    values = np.asarray(pareto_k, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Pareto-k values contain no finite entries.")
    indices = np.arange(len(values))
    good_k = float(loo_result.good_k)
    acceptable = finite & (values <= good_k)
    problematic = finite & (values > good_k)
    ax.plot(
        indices[acceptable], values[acceptable], color=DEFAULT_COLORS[0],
        linestyle="none", marker=".", markersize=4.0,
    )
    if np.any(problematic):
        ax.plot(
            indices[problematic], values[problematic], color=DEFAULT_COLORS[1],
            linestyle="none", marker="o", markersize=3.5,
        )
    ax.axhline(good_k, color=REFERENCE_COLOR, linestyle="--", linewidth=1.1, label=f"good k = {good_k:g}")
    for level in (0.5, 0.7, 1.0):
        if not np.isclose(level, good_k):
            ax.axhline(level, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Observation index", fontweight="bold")
    ax.set_ylabel("Pareto k", fontweight="bold")
    ax.set_title(title, fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
    legend = ax.legend(prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND})
    _style_legend(legend)
    _style_axis(ax)


def plot_loo_pit_ecdf(loo_pit, model_name, output_path: str | Path, context_label=None):
    from mkm.postprocessing.calibration import build_loo_pit_datatree

    data = build_loo_pit_datatree(loo_pit)
    pc = azp.plot_ecdf_pit(
        data,
        var_names=["rate_observed"],
        group="loo_pit",
        sample_dims=["observation"],
        method="pot_c",
        envelope_prob=0.95,
        coverage=False,
        backend="matplotlib",
    )
    prefix = f"{context_label}: " if context_label else ""
    pc.add_title(f"{prefix}LOO-PIT calibration: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_loo_pit_coverage(loo_pit, model_name, output_path: str | Path, context_label=None):
    from mkm.postprocessing.calibration import build_loo_pit_datatree

    data = build_loo_pit_datatree(loo_pit)
    pc = azp.plot_ecdf_pit(
        data,
        var_names=["rate_observed"],
        group="loo_pit",
        sample_dims=["observation"],
        method="pot_c",
        envelope_prob=0.95,
        coverage=True,
        backend="matplotlib",
    )
    prefix = f"{context_label}: " if context_label else ""
    pc.add_title(f"{prefix}LOO predictive coverage: {model_name}")
    pc.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close("all")


def plot_loo_pit_summary(loo_pit, model_name, output_path: str | Path, context_label=None):
    fig, axes = plt.subplots(2, 1, figsize=LOO_PIT_SUMMARY_SIZE_IN)
    _plot_loo_pit_axis(axes[0], loo_pit, coverage=True, title="Predictive coverage")
    _plot_loo_pit_axis(axes[1], loo_pit, coverage=False, title="LOO-PIT calibration")
    prefix = f"{context_label}: " if context_label else ""
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.90, hspace=0.58)
    _add_fixed_gap_global_title(fig, axes, f"{prefix}LOO-PIT diagnostics")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_loo_diagnostics(loo_result, loo_pit, model_name, output_path: str | Path, context_label=None):
    fig, axes = plt.subplots(3, 1, figsize=LOO_DIAGNOSTICS_SIZE_IN)
    _plot_pareto_k_axis(axes[0], loo_result, title="Pareto-k")
    if loo_pit is None:
        for ax, title in zip(axes[1:], ("Predictive coverage", "LOO-PIT calibration"), strict=True):
            ax.set_title(title, fontweight="bold")
            ax.text(
                0.5, 0.5, "LOO-PIT unavailable", ha="center", va="center",
                transform=ax.transAxes, fontweight=FONT_WEIGHT,
            )
            ax.set_axis_off()
    else:
        _plot_loo_pit_axis(axes[1], loo_pit, coverage=True, title="Predictive coverage")
        _plot_loo_pit_axis(axes[2], loo_pit, coverage=False, title="LOO-PIT calibration")
    prefix = f"{context_label}: " if context_label else ""
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.065, top=0.92, hspace=0.66)
    _add_fixed_gap_global_title(fig, axes, f"{prefix}LOO diagnostics")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_loo_pit_conditions(pointwise, model_name, output_path: str | Path, context_label=None):
    """Plot potential-resolved raw LOO-PIT values for one material."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values), len(koh_values),
        figsize=_grid_figsize(
            len(koh_values), len(co_values), panel_width=LOO_PANEL_WIDTH_IN,
            panel_height=LOO_PANEL_HEIGHT_IN, extra_height=LOO_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, co_fraction in enumerate(co_values):
        for col, c_koh in enumerate(koh_values):
            ax = axes[row, col]
            condition = pointwise[
                (pointwise["electrolyte_concentration_M"] == c_koh)
                & (pointwise["CO_mole_fraction"] == co_fraction)
            ]
            for replicate, curve in condition.groupby("replicate", sort=True):
                curve = curve.sort_values("E_V_SHE")
                ax.plot(
                    curve["E_V_SHE"], curve["loo_pit"], marker="o", markersize=2.5,
                    linewidth=0.8, alpha=0.8, label=replicate,
                )
            ax.axhline(0.50, linestyle="--", linewidth=1.0, alpha=0.60, color=REFERENCE_COLOR)
            ax.axhline(0.05, linestyle=":", linewidth=0.8, alpha=0.40, color=REFERENCE_COLOR)
            ax.axhline(0.95, linestyle=":", linewidth=0.8, alpha=0.40, color=REFERENCE_COLOR)
            ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(koh_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True, bounded01_y=True)
    legend = axes[0, 0].legend(
        title="replicate", prop={"weight": FONT_WEIGHT, "size": FONT_SIZE_LEGEND}
    )
    _style_legend(legend)
    prefix = f"{context_label}: " if context_label else ""
    _finish_condition_grid(
        fig, axes, title=f"{prefix}Condition-resolved LOO-PIT", xlabel="Potential (V vs SHE)", ylabel="LOO-PIT",
        legend=False, left=0.075, right=0.93, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


