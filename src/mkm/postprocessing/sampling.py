from dataclasses import dataclass

import arviz_stats as azs
import numpy as np
import pandas as pd
import xarray as xr

from mkm.postprocessing.diagnostics import NOISE_VARIABLES


@dataclass(frozen=True)
class SamplingDiagnostics:
    parameter_summary: pd.DataFrame
    run_summary: pd.DataFrame
    bfmi_by_chain: pd.DataFrame


def sampling_parameter_names(posterior, parameter_specs):
    names = list(parameter_specs)

    for name in NOISE_VARIABLES:
        if name not in posterior:
            continue

        values = posterior[name].squeeze(drop=True)
        extra_dims = set(values.dims) - {"chain", "draw"}

        if not extra_dims:
            names.append(name)

    return names


def build_sampling_datatree(inference_data, parameter_names):
    posterior = inference_data.posterior
    variables = {}

    for name in parameter_names:
        if name not in posterior:
            raise ValueError(f"Posterior is missing sampling parameter '{name}'.")

        values = posterior[name].squeeze(drop=True)
        extra_dims = set(values.dims) - {"chain", "draw"}

        if extra_dims:
            raise ValueError(
                f"Sampling parameter '{name}' is not scalar; remaining dimensions: {sorted(extra_dims)}"
            )

        variables[name] = values

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
        rename = {source: target for source, target in aliases.items() if source in sample_stats and target not in sample_stats}

        if rename:
            sample_stats = sample_stats.rename(rename)

        groups["/sample_stats"] = sample_stats

    return xr.DataTree.from_dict(groups)

def build_sampling_diagnostics(inference_data, parameter_names):
    data = build_sampling_datatree(inference_data, parameter_names)

    parameter_summary = azs.summary(
        data, var_names=parameter_names, group="posterior", kind="diagnostics", fmt="wide", round_to="none"
    )
    parameter_summary = parameter_summary.reset_index()
    parameter_summary = parameter_summary.rename(columns={parameter_summary.columns[0]: "parameter"})

    has_errors, diagnostics = azs.diagnose(
        data, var_names=parameter_names, show_diagnostics=False, return_diagnostics=True
    )

    bfmi = np.asarray(diagnostics["bfmi"]["bfmi_values"], dtype=float).reshape(-1)
    bfmi_by_chain = pd.DataFrame({"chain": np.arange(len(bfmi), dtype=int), "bfmi": bfmi})

    sample_stats = data["sample_stats"].to_dataset()
    reached_max = np.asarray(sample_stats["reached_max_treedepth"], dtype=bool)
    n_max_treedepth = int(np.sum(reached_max))
    fraction_max_treedepth = float(np.mean(reached_max))

    run_summary = pd.DataFrame(
        [
            {
                "has_diagnostic_errors": bool(has_errors),
                "n_divergent": int(diagnostics["divergent"]["n_divergent"]),
                "fraction_divergent": float(diagnostics["divergent"]["pct"]) / 100.0,
                "n_max_treedepth": n_max_treedepth,
                "fraction_max_treedepth": fraction_max_treedepth,
                "min_bfmi": float(np.min(bfmi)),
                "max_rhat": float(parameter_summary["r_hat"].max()),
                "min_ess_bulk": float(parameter_summary["ess_bulk"].min()),
                "min_ess_tail": float(parameter_summary["ess_tail"].min()),
            }
        ]
    )

    return SamplingDiagnostics(
        parameter_summary=parameter_summary,
        run_summary=run_summary,
        bfmi_by_chain=bfmi_by_chain,
    )