from dataclasses import dataclass
from pathlib import Path

import arviz as az
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


def load_inference_data(path: str | Path):
    """Load a NetCDF checkpoint fully into memory and release its file handles."""
    path = Path(path)
    inference_data = az.from_netcdf(path)

    try:
        inference_data.load()
    finally:
        close = getattr(inference_data, "close", None)
        if close is not None:
            close()

    return inference_data


def write_inference_data(inference_data, output_path: str | Path, *, engine="h5netcdf"):
    """Atomically replace a NetCDF checkpoint without leaving a partial target file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        inference_data.to_netcdf(temporary_path, engine=engine)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def retain_group_variables(inference_data, group_name, variable_names):
    """Retain selected variables in one inference-data group without mutating other groups."""
    variable_names = tuple(variable_names)
    group = _get_group(inference_data, group_name)
    missing = [name for name in variable_names if name not in group]
    if missing:
        raise ValueError(
            f"Inference-data group {group_name!r} is missing variables: {missing}"
        )

    if hasattr(group, "to_dataset"):
        dataset = group.to_dataset()
        inference_data[group_name] = dataset[list(variable_names)]
    else:
        setattr(inference_data, group_name, group[list(variable_names)])

    return inference_data


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


def compute_posterior_deterministics(
    inference_data,
    built_model,
    var_names=None,
    backend=None,
    progressbar=True,
):
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

    divergences = (
        int(np.asarray(sample_stats["diverging"]).sum())
        if "diverging" in sample_stats
        else 0
    )

    tree_depth = None
    for name in ("tree_depth", "depth"):
        if name in sample_stats:
            tree_depth = np.asarray(sample_stats[name], dtype=float)
            break

    if tree_depth is None:
        return SamplerHealth(
            divergences=divergences,
            mean_tree_depth=None,
            max_tree_depth=None,
        )

    return SamplerHealth(
        divergences=divergences,
        mean_tree_depth=float(np.mean(tree_depth)),
        max_tree_depth=int(np.max(tree_depth)),
    )
