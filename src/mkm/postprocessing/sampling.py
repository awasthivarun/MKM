import arviz as az
import arviz_stats as azs
import numpy as np
import xarray as xr

from mkm.postprocessing.diagnostics import ERROR_VARIABLES


def _component_label(values, dim, index):
    if dim in values.coords and values.coords[dim].ndim == 1:
        return str(values.coords[dim].values[index])
    return str(index)


def _expand_variable(values, name):
    values = values.squeeze(drop=True)
    extra_dims = tuple(dim for dim in values.dims if dim not in {"chain", "draw"})
    if not extra_dims:
        return {name: values}

    variables = {}
    shape = tuple(values.sizes[dim] for dim in extra_dims)
    for index in np.ndindex(shape):
        indexers = dict(zip(extra_dims, index, strict=True))
        labels = [
            f"{dim}={_component_label(values, dim, dim_index)}"
            for dim, dim_index in zip(extra_dims, index, strict=True)
        ]
        variables[f"{name}[{','.join(labels)}]"] = values.isel(indexers, drop=True)
    return variables


def sampling_parameter_names(posterior, parameter_specs, exclude=ERROR_VARIABLES):
    """Return configured scalar variable names, excluding pair-plot nuisances by default."""
    excluded = set(exclude)
    names = []
    for name in parameter_specs:
        if name in excluded:
            continue
        if name not in posterior:
            raise ValueError(f"Posterior is missing sampling parameter '{name}'.")

        values = posterior[name].squeeze(drop=True)
        extra_dims = set(values.dims) - {"chain", "draw"}
        if extra_dims:
            raise ValueError(
                f"Pair-plot parameter '{name}' is not scalar; remaining dimensions: "
                f"{sorted(extra_dims)}."
            )
        names.append(name)
    return tuple(names)


def build_sampling_datatree(inference_data, parameter_names):
    posterior = inference_data.posterior
    variables = {}

    for name in parameter_names:
        if name not in posterior:
            raise ValueError(f"Posterior is missing sampling parameter '{name}'.")
        variables.update(_expand_variable(posterior[name], name))

    groups = {"/posterior": xr.Dataset(variables)}
    if hasattr(inference_data, "sample_stats"):
        sample_stats = inference_data.sample_stats
        if not isinstance(sample_stats, xr.Dataset):
            sample_stats = sample_stats.to_dataset()
        else:
            sample_stats = sample_stats.copy()

        aliases = {
            "depth": "tree_depth",
            "maxdepth_reached": "reached_max_treedepth",
            "logp": "lp",
        }
        rename = {
            source: target
            for source, target in aliases.items()
            if source in sample_stats and target not in sample_stats
        }
        if rename:
            sample_stats = sample_stats.rename(rename)
        groups["/sample_stats"] = sample_stats

    return xr.DataTree.from_dict(groups)


def build_sampling_health(inference_data, parameter_names=None):
    """Return run-level sampler health for metadata and console reporting."""
    posterior = inference_data.posterior
    if parameter_names is None:
        parameter_names = tuple(posterior.data_vars)

    diagnostic = azs.summary(
        inference_data,
        var_names=list(parameter_names),
        kind="diagnostics",
        round_to="none",
    )

    sample_stats = getattr(inference_data, "sample_stats", None)
    n_divergent = 0
    n_max_treedepth = 0
    fraction_max_treedepth = np.nan

    if sample_stats is not None:
        if "diverging" in sample_stats:
            n_divergent = int(np.asarray(sample_stats["diverging"], dtype=bool).sum())
        reached_name = None
        for candidate in ("reached_max_treedepth", "maxdepth_reached"):
            if candidate in sample_stats:
                reached_name = candidate
                break
        if reached_name is not None:
            reached = np.asarray(sample_stats[reached_name], dtype=bool)
            n_max_treedepth = int(reached.sum())
            fraction_max_treedepth = float(reached.mean())

    bfmi = np.asarray(az.bfmi(inference_data), dtype=float).reshape(-1)
    return {
        "n_divergent": n_divergent,
        "n_max_treedepth": n_max_treedepth,
        "fraction_max_treedepth": fraction_max_treedepth,
        "min_bfmi": float(np.nanmin(bfmi)) if bfmi.size else np.nan,
        "max_rhat": float(diagnostic["r_hat"].max()) if "r_hat" in diagnostic else np.nan,
        "min_ess_bulk": (
            float(diagnostic["ess_bulk"].min()) if "ess_bulk" in diagnostic else np.nan
        ),
        "min_ess_tail": (
            float(diagnostic["ess_tail"].min()) if "ess_tail" in diagnostic else np.nan
        ),
    }
