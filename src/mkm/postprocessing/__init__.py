"""Posterior-derived calculations, diagnostics, and model comparison."""

from mkm.postprocessing.observables import (
    PosteriorObservableSummary,
    summarize_pointwise_pathway_fractions,
    summarize_posterior_linear_observable,
    summarize_posterior_model_variable,
)

from mkm.postprocessing.diagnostics import (
    COVERAGE_VARIABLES,
    NOISE_VARIABLES,
    PATHWAY_FRACTION_VARIABLES,
    build_balance_summary,
    build_noise_summary,
    build_parameter_contraction,
    build_physical_summary,
    flatten_posterior_samples,
    prior_statistics,
    summarize_samples,
    summarize_scalar_samples,
)

__all__ = [
    "PosteriorObservableSummary",
    "summarize_pointwise_pathway_fractions",
    "summarize_posterior_linear_observable",
    "summarize_posterior_model_variable",
    "COVERAGE_VARIABLES",
    "NOISE_VARIABLES",
    "PATHWAY_FRACTION_VARIABLES",
    "build_balance_summary",
    "build_noise_summary",
    "build_parameter_contraction",
    "build_physical_summary",
    "flatten_posterior_samples",
    "prior_statistics",
    "summarize_samples",
    "summarize_scalar_samples",
]
