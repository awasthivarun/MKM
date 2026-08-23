"""Backward-compatible imports for posterior post-processing utilities.

New code should import these functions from ``mkm.postprocessing.observables``.
"""

from mkm.postprocessing.observables import (
    PosteriorObservableSummary,
    summarize_pointwise_pathway_fractions,
    summarize_posterior_linear_observable,
    summarize_posterior_model_variable,
    summarize_pointwise_posterior_variable
)

__all__ = [
    "PosteriorObservableSummary",
    "summarize_pointwise_pathway_fractions",
    "summarize_posterior_linear_observable",
    "summarize_posterior_model_variable",
    "summarize_pointwise_posterior_variable", 
]