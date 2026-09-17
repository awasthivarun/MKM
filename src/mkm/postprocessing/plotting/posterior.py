from ._shared import *
def plot_parameter_posteriors(posterior, parameter_specs, output_path: str | Path):
    components = _posterior_components(posterior, parameter_specs)
    if not components:
        raise ValueError("No parameter specifications were provided.")
    ncols = min(3, len(components))
    nrows = math.ceil(len(components) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=_grid_figsize(
            ncols, nrows, panel_width=POSTERIOR_PANEL_WIDTH_IN,
            panel_height=POSTERIOR_PANEL_HEIGHT_IN, extra_height=POSTERIOR_EXTRA_HEIGHT_IN,
        ),
        squeeze=False,
    )

    for ax, (label, values, spec) in zip(axes.flat, components):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Posterior parameter '{label}' contains non-finite values.")
        hdi80 = np.asarray(azs.hdi(values, prob=0.80), dtype=float).reshape(-1)
        hdi95 = np.asarray(azs.hdi(values, prob=0.95), dtype=float).reshape(-1)
        mean = float(np.mean(values))
        lower, upper = np.quantile(values, [0.001, 0.999])
        if np.isclose(lower, upper):
            width = max(abs(float(lower)) * 0.05, 1e-9)
            lower -= width
            upper += width
        posterior_width = max(float(upper - lower), 1e-12)
        support_lower, support_upper = _prior_support(spec)
        pad = 0.08 * posterior_width
        if np.isfinite(support_lower) and support_lower >= lower - 2.0 * posterior_width:
            lower = min(lower, support_lower - pad)
        if np.isfinite(support_upper) and support_upper <= upper + 2.0 * posterior_width:
            upper = max(upper, support_upper + pad)
        x = np.linspace(lower, upper, 600)

        if len(np.unique(values)) > 1:
            posterior_density = gaussian_kde(values)(x)
            posterior_density = np.where((x >= support_lower) & (x <= support_upper), posterior_density, 0.0)
            density_max = float(np.max(posterior_density))
            ax.fill_between(
                x, 0.0, posterior_density, where=(x >= hdi95[0]) & (x <= hdi95[1]),
                color=POSTERIOR_COLOR, alpha=HDI95_ALPHA, interpolate=True,
            )
            ax.fill_between(
                x, 0.0, posterior_density, where=(x >= hdi80[0]) & (x <= hdi80[1]),
                color=POSTERIOR_COLOR, alpha=HDI80_ALPHA, interpolate=True,
            )
            ax.plot(x, posterior_density, color=POSTERIOR_COLOR, linewidth=1.7, label="posterior")
        else:
            density_max = 1.0
            ax.axvline(values[0], color=POSTERIOR_COLOR, linewidth=1.7, label="posterior")

        prior = _prior_pdf(x, spec)
        if np.any(np.isfinite(prior)) and np.nanmax(prior) > 0:
            prior_scaled = prior / np.nanmax(prior) * 0.35 * density_max
            ax.plot(x, prior_scaled, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, alpha=0.75, label="prior")

        ax.axvline(hdi95[0], color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, alpha=0.85)
        ax.axvline(hdi95[1], color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, alpha=0.85)
        ax.axvline(mean, color="#303030", linewidth=1.0, alpha=0.9, zorder=2)
        hdi_width = float(hdi95[1] - hdi95[0])
        text_y = 0.96 * max(density_max, 1e-12)
        for value, horizontal_alignment in ((hdi95[0], "right"), (hdi95[1], "left")):
            ax.annotate(
                _parameter_value_text(float(value), hdi_width),
                xy=(float(value), text_y),
                rotation=90,
                ha=horizontal_alignment,
                va="top",
                fontsize=FONT_SIZE_POSTERIOR_ANNOTATION,
                fontweight=FONT_WEIGHT,
                color="#303030",
            )
        ax.annotate(
            _parameter_value_text(mean, hdi_width),
            xy=(mean, 0.04 * max(density_max, 1e-12)),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_POSTERIOR_ANNOTATION,
            fontweight=FONT_WEIGHT,
            color="#202020",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.0},
            zorder=5,
        )
        ax.set_title(label, pad=TEXT_GAP_PT, fontweight="bold")
        ax.set_yticks([])
        ax.set_box_aspect(POSTERIOR_BOX_ASPECT)
        _nice_linear_ticks(ax.xaxis)
        _bold_axis_text(ax)
        ax.grid(False)

    for ax in axes.flat[len(components):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.025, top=0.938, hspace=0.25, wspace=POSTERIOR_WSPACE)
    _add_fixed_gap_global_title(fig, axes, "Posterior parameters")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def _posterior_interval_columns(frame, prefix=None):
    base = "" if prefix is None else f"{prefix}_"
    columns = (
        f"{base}median",
        f"{base}hdi95_lower",
        f"{base}hdi95_upper",
    )
    if all(column in frame.columns for column in columns):
        return columns
    raise ValueError(
        f"Could not find posterior median/95% interval columns for prefix '{prefix}'."
    )


def plot_observation_grid(
    observations,
    output_path: str | Path,
    residual=False,
    y_scale="log",
    distribution="predictive",
    context_label=None,
):
    if y_scale not in {"linear", "log"}:
        raise ValueError("y_scale must be 'linear' or 'log'.")
    if distribution not in {"model", "predictive"}:
        raise ValueError("distribution must be 'model' or 'predictive'.")
    KOH_values = sorted(observations["electrolyte_concentration_M"].unique())
    CO_values = sorted(observations["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(CO_values), len(KOH_values),
        figsize=_grid_figsize(
            len(KOH_values), len(CO_values), panel_width=OBSERVATION_PANEL_WIDTH_IN,
            panel_height=OBSERVATION_PANEL_HEIGHT_IN, extra_width=OBSERVATION_EXTRA_WIDTH_IN,
            extra_height=OBSERVATION_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    material = _material_from_context(context_label)
    posterior_color, experiment_color = _material_plot_colors(material)
    replicate_names = (
        sorted(str(value) for value in observations["replicate"].unique())
        if "replicate" in observations
        else []
    )
    grey_levels = np.linspace(*RESIDUAL_REPLICATE_GREY_RANGE, max(len(replicate_names), 1))
    replicate_color_map = {
        name: str(float(level)) for name, level in zip(replicate_names, grey_levels, strict=False)
    }

    if not residual:
        prefix = f"rate_{distribution}"
        _posterior_interval_columns(observations, prefix)
    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = observations[
                (observations["electrolyte_concentration_M"] == c_koh)
                & (observations["CO_mole_fraction"] == co_fraction)
            ]
            if residual:
                for replicate, curve in condition.groupby("replicate", sort=True):
                    curve = curve.sort_values("E_V_SHE")
                    ax.plot(
                        curve["E_V_SHE"], curve["residual"], linewidth=REPLICATE_LINEWIDTH, alpha=0.95,
                        color=replicate_color_map.get(str(replicate), "0.45"),
                    )
                ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, alpha=0.7)
            else:
                for _, curve in condition.groupby("replicate", sort=True):
                    curve = curve.sort_values("E_V_SHE")
                    ax.plot(
                        curve["E_V_SHE"], curve["rate"], color=REPLICATE_LINE_COLOR,
                        linewidth=REPLICATE_LINEWIDTH, alpha=REPLICATE_LINE_ALPHA, zorder=2.4,
                    )
                posterior_curve = condition.sort_values("E_V_SHE").drop_duplicates("model_point_id")
                _plot_summary_curve(
                    ax, posterior_curve, "E_V_SHE", prefix=f"rate_{distribution}", color=posterior_color,
                    linewidth=2.0,
                )
                if distribution == "predictive":
                    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8, alpha=0.45)
            if not residual and y_scale == "log":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True, log_y=(not residual and y_scale == "log"))
    _set_presentation_axes(axes)
    display_context = _material_from_context(context_label)
    prefix = f"{display_context}: " if display_context else ""
    if residual:
        title = f"{prefix}Rate residuals"
        ylabel = r"observed rate - posterior model median / s$^{-1}$"
    else:
        rate_title = "Model rate" if distribution == "model" else "Posterior predictive rate"
        title = f"{prefix}{rate_title}"
        ylabel = r"rate / s$^{-1}$"
    _finish_condition_grid(
        fig, axes, title=title, xlabel="Potential (V vs SHE)", ylabel=ylabel, legend=False,
        left=0.085, right=0.92, bottom=0.065, top=0.925, wspace=0.30,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)

def plot_pointwise_variable(summary, variable_name, output_path: str | Path, context_label=None):
    KOH_values = sorted(summary["electrolyte_concentration_M"].unique())
    CO_values = sorted(summary["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(CO_values), len(KOH_values),
        figsize=_grid_figsize(
            len(KOH_values), len(CO_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=POINTWISE_EXTRA_HEIGHT_IN,
        ),
        sharex=True, squeeze=False,
    )
    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = summary[
                (summary["electrolyte_concentration_M"] == c_koh)
                & (summary["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            _plot_summary_curve(ax, condition, "E_V_SHE", color=POSTERIOR_COLOR, linewidth=1.9)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True)
    _set_presentation_axes(axes)
    display_context = _material_from_context(context_label)
    title = f"{display_context}: {variable_name}" if display_context else variable_name
    _finish_condition_grid(
        fig, axes, title=title, xlabel="Potential (V vs SHE)", ylabel=variable_name,
        legend=False, left=0.08, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def plot_pointwise_variables(
    summary,
    variable_names,
    output_path: str | Path,
    *,
    title,
    ylabel,
    colors=None,
    labels=None,
    context_label=None,
    bounded=True,
    observed_rates=None,
):
    variable_names = [variable for variable in variable_names if f"{variable}_median" in summary.columns]
    if not variable_names:
        return False
    colors = {} if colors is None else colors
    labels = {} if labels is None else labels
    KOH_values = sorted(summary["electrolyte_concentration_M"].unique())
    CO_values = sorted(summary["CO_mole_fraction"].unique())
    if observed_rates is not None:
        required = {"electrolyte_concentration_M", "CO_mole_fraction", "E_V_SHE", "rate"}
        missing = sorted(required - set(observed_rates.columns))
        if missing:
            raise ValueError(f"Observed-rate overlay is missing required columns: {missing}.")
    fig, axes = plt.subplots(
        len(CO_values), len(KOH_values),
        figsize=_grid_figsize(
            len(KOH_values), len(CO_values), panel_width=MATERIAL_PANEL_WIDTH_IN,
            panel_height=MATERIAL_PANEL_HEIGHT_IN, extra_width=MATERIAL_EXTRA_WIDTH_IN,
            extra_height=MATERIAL_EXTRA_HEIGHT_IN,
        ),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, co_fraction in enumerate(CO_values):
        for col, c_koh in enumerate(KOH_values):
            ax = axes[row, col]
            condition = summary[
                (summary["electrolyte_concentration_M"] == c_koh)
                & (summary["CO_mole_fraction"] == co_fraction)
            ].sort_values("E_V_SHE")
            for variable in variable_names:
                _plot_summary_curve(
                    ax, condition, "E_V_SHE", prefix=variable, color=colors.get(variable),
                    label=labels.get(variable, variable), linewidth=1.9,
                )
            if observed_rates is not None:
                observed_condition = observed_rates[
                    (observed_rates["electrolyte_concentration_M"] == c_koh)
                    & (observed_rates["CO_mole_fraction"] == co_fraction)
                ]
                if not observed_condition.empty:
                    mean_rate = (
                        observed_condition.groupby("E_V_SHE", as_index=False)["rate"]
                        .mean()
                        .sort_values("E_V_SHE")
                    )
                    rate_values = mean_rate["rate"].to_numpy(dtype=float)
                    if not np.all(np.isfinite(rate_values)):
                        raise ValueError("Observed-rate overlay contains non-finite mean TOF values.")
                    max_rate = float(np.max(rate_values))
                    if max_rate > 0.0:
                        ax.plot(
                            mean_rate["E_V_SHE"], rate_values / max_rate, color=REFERENCE_COLOR,
                            linestyle=":", linewidth=1.5, label="TOF",
                        )
            if bounded:
                ax.set_ylim(-0.02, 1.02)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH", pad=TEXT_GAP_PT, fontweight="bold")
            if col == len(KOH_values) - 1:
                ax.text(
                    1.035, 0.5, f"{100 * co_fraction:g}% CO", transform=ax.transAxes,
                    rotation=-90, va="center", fontweight=FONT_WEIGHT,
                )
            _style_axis(ax, potential_x=True, bounded01_y=bounded)
    _set_presentation_axes(axes)
    _add_axis_legend(axes[0, 0], loc="best")
    display_context = _material_from_context(context_label)
    prefix = f"{display_context}: " if display_context else ""
    _finish_condition_grid(
        fig, axes, title=f"{prefix}{title}", xlabel="Potential (V vs SHE)", ylabel=ylabel,
        legend=False, left=0.08, right=0.915, bottom=0.065, top=0.925,
    )
    _save_presentation_figure(fig, output_path)
    plt.close(fig)
    return True

def plot_sampling_trace(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    nrows = math.ceil(len(plotted_names) / 3)
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        with azb.rc_context({"plot.max_subplots": max(40, len(plotted_names))}):
            pc = azp.plot_trace(
                data,
                var_names=plotted_names,
                group="posterior",
                backend="matplotlib",
                aes_by_visuals={"divergence": ["color"]},
                visuals={"divergence": {"marker": "|", "size": 30}},
                col_wrap=3,
                figure_kwargs={
                    "figsize": (TRACE_FIGURE_WIDTH_IN, TRACE_PANEL_HEIGHT_IN * nrows + TRACE_EXTRA_HEIGHT_IN),
                    "layout": "none",
                },
            )
        fig = pc.get_target(plotted_names[0], {}).figure
        for ax in fig.axes:
            ax.grid(False)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
            ax.set_title(ax.get_title(), fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        _apply_trace_chain_alpha(fig, inference_data)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.035, top=0.965, hspace=0.30, wspace=0.20)
        _add_fixed_gap_global_title(fig, fig.axes, "MCMC traces")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)

def _rank_ecdf_outside_envelope(data, variable, *, envelope_prob=0.99, n_simulations=1000):
    """Return rank-ECDF curves and a mask for excursions outside the simultaneous envelope."""
    posterior = data.posterior.to_dataset()[[variable]]
    thinned = posterior.azstats.thin(sample_dims=["draw"])
    sample_size = int(thinned.sizes["draw"])
    n_chains = int(thinned.sizes["chain"])

    ranks = thinned.azstats.compute_ranks(dim=["chain", "draw"])
    ecdf = ranks.azstats.ecdf(dim=["draw"], pit=True, npoints=sample_size)[variable]
    x = np.asarray(ecdf.sel(plot_axis="x"), dtype=float)
    y = np.asarray(ecdf.sel(plot_axis="y"), dtype=float)

    x_ci, _, lower_ci, upper_ci = ecdf_pit(
        np.linspace(0.0, 1.0, sample_size),
        envelope_prob,
        n_simulations=n_simulations,
        n_chains=n_chains,
    )
    lower_ci = np.asarray(lower_ci, dtype=float) - np.asarray(x_ci, dtype=float)
    upper_ci = np.asarray(upper_ci, dtype=float) - np.asarray(x_ci, dtype=float)

    lower_at_x = np.vstack([np.interp(chain_x, x_ci, lower_ci) for chain_x in x])
    upper_at_x = np.vstack([np.interp(chain_x, x_ci, upper_ci) for chain_x in x])
    outside = (y < lower_at_x) | (y > upper_at_x)
    return x, y, outside


def _overlay_rank_envelope_violations(ax, data, variable):
    x, y, outside = _rank_ecdf_outside_envelope(data, variable)
    for chain_index in range(x.shape[0]):
        flagged = np.where(outside[chain_index], y[chain_index], np.nan)
        if np.any(np.isfinite(flagged)):
            ax.step(
                x[chain_index], flagged, where="pre", color="black", linewidth=1.8,
                solid_capstyle="round", zorder=8,
            )


def plot_sampling_rank(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    nrows = math.ceil(len(plotted_names) / 3)
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        with azb.rc_context({"plot.max_subplots": max(40, len(plotted_names))}):
            pc = azp.plot_rank(
                data,
                var_names=plotted_names,
                group="posterior",
                backend="matplotlib",
                envelope_prob=0.99,
                stats={"ecdf_pit": {"n_simulations": 1000}},
                col_wrap=3,
                visuals={
                    "xlabel": False,
                    "credible_interval": {"color": "#D9D9D9", "alpha": 0.42},
                },
                figure_kwargs={
                    "figsize": (RANK_FIGURE_WIDTH_IN, RANK_PANEL_HEIGHT_IN * nrows + RANK_EXTRA_HEIGHT_IN),
                    "layout": "none",
                },
            )
        fig = pc.get_target(plotted_names[0], {}).figure
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        for ax, variable in zip(visible_axes, plotted_names, strict=False):
            _overlay_rank_envelope_violations(ax, data, variable)
        for ax in fig.axes:
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
            ax.set_title(ax.get_title(), fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.03, top=0.965, hspace=0.27, wspace=0.20)
        _add_fixed_gap_global_title(fig, fig.axes, "Chain ranks")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)

def plot_sampling_energy(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        azp.plot_energy(
            data,
            backend="matplotlib",
            show_bfmi=True,
            threshold=0.3,
            figure_kwargs={"figsize": ENERGY_FIGURE_SIZE_IN, "layout": "none"},
        )
        fig = plt.gcf()
        for ax in fig.axes:
            ax.grid(False)
            _nice_linear_ticks(ax.xaxis)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
            ax.set_title(ax.get_title(), fontweight=FONT_WEIGHT, fontsize=FONT_SIZE_SUBPLOT_TITLE, pad=TEXT_GAP_PT)
            ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
            ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
            legend = ax.get_legend()
            if legend is not None:
                _style_legend(legend)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.20, top=0.78, wspace=0.32)
        _add_fixed_gap_global_title(fig, fig.axes, "Sampling energy and BFMI")
        _save_presentation_figure(fig, output_path)
        plt.close(fig)

def _sampling_parameter_matrix(inference_data, parameter_names):
    data = build_sampling_datatree(inference_data, parameter_names)
    names = list(data.posterior.data_vars)
    if not names:
        raise ValueError("No sampling parameters were available for correlation diagnostics.")
    columns = []
    for name in names:
        values = np.asarray(data.posterior[name], dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Sampling parameter '{name}' contains non-finite values.")
        columns.append(values)
    return names, np.column_stack(columns)


def plot_sampling_correlations(inference_data, parameter_names, output_path: str | Path):
    names, samples = _sampling_parameter_matrix(inference_data, parameter_names)
    correlation = np.corrcoef(samples, rowvar=False)
    n_parameters = len(names)
    size = max(CORRELATION_MIN_SIZE_IN, CORRELATION_SIZE_PER_PARAMETER_IN * n_parameters)
    fig, ax = plt.subplots(figsize=(size, size))
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    image = ax.imshow(np.ma.masked_where(mask, correlation), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(n_parameters), labels=names, rotation=90)
    ax.set_yticks(np.arange(n_parameters), labels=names)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    _bold_axis_text(ax)
    for row in range(n_parameters):
        for col in range(row + 1):
            value = float(correlation[row, col])
            text_color = "white" if abs(value) >= 0.65 else "black"
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center", color=text_color,
                fontsize=CORRELATION_ANNOTATION_SIZE, fontweight=FONT_WEIGHT,
            )
    ax.set_xlim(-0.5, n_parameters - 0.5)
    ax.set_ylim(n_parameters - 0.5, -0.5)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Pearson r", fontsize=FONT_SIZE_AXIS_LABEL, fontweight=FONT_WEIGHT)
    for label in colorbar.ax.get_yticklabels():
        label.set_fontsize(FONT_SIZE_TICK)
        label.set_fontweight(FONT_WEIGHT)
    fig.subplots_adjust(left=0.23, right=0.91, bottom=0.23, top=0.94)
    _add_fixed_gap_global_title(fig, [ax], "Posterior Pearson correlations")
    _save_presentation_figure(fig, output_path)
    plt.close(fig)


def _simplify_pair_tick_labels(fig):
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if not axes:
        return
    positions = {ax: ax.get_position() for ax in axes}
    min_left = min(position.x0 for position in positions.values())
    min_bottom = min(position.y0 for position in positions.values())
    tolerance = 0.005
    for ax, position in positions.items():
        is_left = abs(position.x0 - min_left) <= tolerance
        is_bottom = abs(position.y0 - min_bottom) <= tolerance
        ax.tick_params(
            axis="x", which="both", bottom=is_bottom, labelbottom=is_bottom,
            top=False, labeltop=False,
        )
        ax.tick_params(
            axis="y", which="both", left=is_left, labelleft=is_left,
            right=False, labelright=False,
        )
        ax.xaxis.get_offset_text().set_visible(is_bottom)
        ax.yaxis.get_offset_text().set_visible(is_left)


def plot_sampling_pairs(inference_data, parameter_names, output_path: str | Path):
    data = build_sampling_datatree(inference_data, parameter_names)
    plotted_names = list(data.posterior.data_vars)
    n_parameters = len(plotted_names)
    size = max(PAIR_MIN_SIZE_IN, PAIR_SIZE_PER_PARAMETER_IN * n_parameters)
    base_visuals = {
        "scatter": {"alpha": 0.12, "s": 5, "rasterized": True},
        "divergence": {"color": "black", "marker": "x", "alpha": 0.9, "s": 16},
    }
    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        with azb.rc_context({"plot.max_subplots": max(40, n_parameters**2)}):
            try:
                pm = azp.plot_pair(
                    data,
                    var_names=plotted_names,
                    group="posterior",
                    marginal=True,
                    marginal_kind="kde",
                    triangle="lower",
                    levels=[0.5, 0.9],
                    backend="matplotlib",
                    aes={"color": ["chain"]},
                    aes_by_visuals={"scatter": ["color"]},
                    visuals={**base_visuals, "contour": True},
                    figure_kwargs={"figsize": (size, size), "layout": "none"},
                )
            except ValueError as error:
                message = str(error)
                if "contour" not in message and "levels" not in message and "aesthetic" not in message:
                    raise
                plt.close("all")
                pm = azp.plot_pair(
                    data,
                    var_names=plotted_names,
                    group="posterior",
                    marginal=True,
                    marginal_kind="kde",
                    triangle="lower",
                    backend="matplotlib",
                    aes={"color": ["chain"]},
                    aes_by_visuals={"scatter": ["color"]},
                    visuals=base_visuals,
                    figure_kwargs={"figsize": (size, size), "layout": "none"},
                )
        fig = plt.gcf()
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            ax.grid(False)
            _nice_linear_ticks(ax.xaxis)
            _nice_linear_ticks(ax.yaxis)
            _bold_axis_text(ax)
        _simplify_pair_tick_labels(fig)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pm.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        if output_path.suffix.lower() == ".png":
            pm.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
        plt.close("all")

