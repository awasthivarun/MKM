"""Postprocessing plotting API split by diagnostic family."""

from ._shared import (
    COVERAGE_COLORS,
    FONT_SIZE_AXIS_LABEL,
    FONT_WEIGHT,
    HDI80_ALPHA,
    HDI95_ALPHA,
    PATHWAY_COLORS,
    POINTWISE_LABELS,
    TEXT_GAP_PT,
    _add_fixed_gap_global_title,
    _bold_axis_text,
    _nice_linear_ticks,
    _save_presentation_figure,
    _style_legend,
    azp,
)
from .drc import plot_transition_state_drc
from .loo import (
    plot_loo_comparison,
    plot_loo_diagnostics,
    plot_loo_pit_conditions,
    plot_loo_pit_coverage,
    plot_loo_pit_ecdf,
    plot_loo_pit_summary,
    plot_pareto_k,
    plot_pointwise_elpd_difference,
    plot_pointwise_loo,
    plot_pointwise_loo_pit,
)
from .observables import (
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
    plot_second_order_difference,
)
from .posterior import (
    plot_observation_grid,
    plot_parameter_posteriors,
    plot_pointwise_variable,
    plot_pointwise_variables,
    plot_sampling_correlations,
    plot_sampling_energy,
    plot_sampling_pairs,
    plot_sampling_rank,
    plot_sampling_trace,
)

__all__ = [name for name in globals() if not name.startswith("_")]
