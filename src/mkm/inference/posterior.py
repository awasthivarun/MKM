from dataclasses import dataclass

import numpy as np
import pymc as pm


@dataclass(frozen=True)
class SamplerHealth:
    divergences: int
    mean_tree_depth: float | None
    max_tree_depth: int | None


def _get_group(inference_data, group_name):
    if hasattr(inference_data, group_name):
        return getattr(inference_data, group_name)

    try:
        return inference_data[group_name]
    except Exception as error:
        raise ValueError(f"Inference data does not contain group '{group_name}'.") from error


def sample_posterior(
    built_model,
    draws=1000,
    tune=1000,
    chains=4,
    cores=None,
    target_accept=0.9,
    random_seed=None,
    nuts_sampler="nutpie",
    backend="numba",
    compute_convergence_checks=True,
):
    draws = int(draws)
    tune = int(tune)
    chains = int(chains)
    target_accept = float(target_accept)

    if draws < 1:
        raise ValueError("draws must be positive.")
    if tune < 0:
        raise ValueError("tune must be non-negative.")
    if chains < 1:
        raise ValueError("chains must be positive.")
    if not np.isfinite(target_accept) or not 0 < target_accept < 1:
        raise ValueError("target_accept must lie strictly between 0 and 1.")

    model = built_model.model
    free_var_names = [rv.name for rv in model.free_RVs]

    sample_kwargs = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "target_accept": target_accept,
        "random_seed": random_seed,
        "var_names": free_var_names,
        "return_inferencedata": True,
        "compute_convergence_checks": compute_convergence_checks,
    }

    if nuts_sampler is not None:
        sample_kwargs["nuts_sampler"] = nuts_sampler

    if backend is not None:
        sample_kwargs["backend"] = backend

    with model:
        return pm.sample(**sample_kwargs)


def compute_posterior_deterministics(inference_data, built_model, var_names=None, backend=None, progressbar=True):
    posterior = _get_group(inference_data, "posterior")

    with built_model.model:
        return pm.compute_deterministics(
            posterior,
            var_names=var_names,
            model=built_model.model,
            extend_dataset=True,
            progressbar=progressbar,
            backend=backend,
        )


def add_log_likelihood(inference_data, built_model, backend=None, progressbar=True):
    with built_model.model:
        return pm.compute_log_likelihood(
            inference_data,
            model=built_model.model,
            extend_inferencedata=True,
            progressbar=progressbar,
            backend=backend,
        )


def summarize_sampler_health(inference_data):
    sample_stats = _get_group(inference_data, "sample_stats")

    divergences = int(np.asarray(sample_stats["diverging"]).sum()) if "diverging" in sample_stats else 0

    tree_depth = None
    for name in ("tree_depth", "depth"):
        if name in sample_stats:
            tree_depth = np.asarray(sample_stats[name], dtype=float)
            break

    if tree_depth is None:
        return SamplerHealth(divergences=divergences, mean_tree_depth=None, max_tree_depth=None)

    return SamplerHealth(
        divergences=divergences,
        mean_tree_depth=float(np.mean(tree_depth)),
        max_tree_depth=int(np.max(tree_depth)),
    )