"""Posterior-shift summaries and figures for held-out validation refits."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde

from mkm.postprocessing.diagnostics import summarize_scalar_samples


def _posterior_dataset(inference_data):
    posterior = inference_data.posterior if hasattr(inference_data, "posterior") else inference_data
    if isinstance(posterior, xr.Dataset):
        return posterior
    if hasattr(posterior, "to_dataset"):
        return posterior.to_dataset()
    raise TypeError("Posterior must be an xarray Dataset or DataTree posterior node.")


def _component_label(data, indexers):
    labels = []
    for dim, index in indexers.items():
        if dim in data.coords and data.coords[dim].ndim == 1:
            value = data.coords[dim].values[index]
        else:
            value = index
        labels.append(f"{dim}={value}")
    return ",".join(labels)


def _posterior_components(inference_data, parameter_specs):
    posterior = _posterior_dataset(inference_data)
    components = {}

    for name in parameter_specs:
        if name not in posterior:
            raise ValueError(f"Posterior is missing configured parameter '{name}'.")

        data = posterior[name]
        extra_dims = tuple(dim for dim in data.dims if dim not in {"chain", "draw"})
        if not extra_dims:
            components[name] = (name, np.asarray(data, dtype=float).reshape(-1))
            continue

        shape = tuple(data.sizes[dim] for dim in extra_dims)
        for index in np.ndindex(shape):
            indexers = dict(zip(extra_dims, index, strict=True))
            label = _component_label(data, indexers)
            component_name = f"{name}[{label}]"
            values = np.asarray(data.isel(indexers, drop=True), dtype=float).reshape(-1)
            components[component_name] = (name, values)

    return components


def summarize_validation_posterior_shift(
    full_inference_data,
    validation_inference_data,
    parameter_specs,
):
    """Compare a held-out refit posterior with the corresponding full-data posterior."""
    full = _posterior_components(full_inference_data, parameter_specs)
    validation = _posterior_components(validation_inference_data, parameter_specs)

    if tuple(full) != tuple(validation):
        missing_validation = [name for name in full if name not in validation]
        extra_validation = [name for name in validation if name not in full]
        raise ValueError(
            "Full and validation posteriors do not contain the same parameter components: "
            f"missing_validation={missing_validation}, extra_validation={extra_validation}."
        )

    records = []
    for label, (variable, full_values) in full.items():
        validation_variable, validation_values = validation[label]
        if validation_variable != variable:
            raise RuntimeError(f"Parameter component '{label}' changed variable identity.")
        if not np.all(np.isfinite(full_values)) or not np.all(np.isfinite(validation_values)):
            raise ValueError(f"Parameter component '{label}' contains non-finite values.")

        full_summary = summarize_scalar_samples(full_values)
        validation_summary = summarize_scalar_samples(validation_values)

        full_width = full_summary["hdi95_upper"] - full_summary["hdi95_lower"]
        overlap_lower = max(
            full_summary["hdi95_lower"], validation_summary["hdi95_lower"]
        )
        overlap_upper = min(
            full_summary["hdi95_upper"], validation_summary["hdi95_upper"]
        )
        overlap_width = max(0.0, overlap_upper - overlap_lower)
        overlap_fraction = (
            overlap_width / full_width if np.isfinite(full_width) and full_width > 0 else np.nan
        )
        shift = validation_summary["median"] - full_summary["median"]
        standardized_shift = (
            shift / full_summary["sd"]
            if np.isfinite(full_summary["sd"]) and full_summary["sd"] > 0
            else np.nan
        )

        records.append(
            {
                "parameter": label,
                "variable": variable,
                "parameter_type": "error" if variable.startswith("sigma_rate_") else "physical",
                "full_mean": full_summary["mean"],
                "full_sd": full_summary["sd"],
                "full_median": full_summary["median"],
                "full_hdi95_lower": full_summary["hdi95_lower"],
                "full_hdi95_upper": full_summary["hdi95_upper"],
                "validation_mean": validation_summary["mean"],
                "validation_sd": validation_summary["sd"],
                "validation_median": validation_summary["median"],
                "validation_hdi95_lower": validation_summary["hdi95_lower"],
                "validation_hdi95_upper": validation_summary["hdi95_upper"],
                "median_shift": shift,
                "median_shift_in_full_sd": standardized_shift,
                "hdi95_overlap_fraction_of_full": overlap_fraction,
                "validation_median_inside_full_hdi95": bool(
                    full_summary["hdi95_lower"]
                    <= validation_summary["median"]
                    <= full_summary["hdi95_upper"]
                ),
                "full_median_inside_validation_hdi95": bool(
                    validation_summary["hdi95_lower"]
                    <= full_summary["median"]
                    <= validation_summary["hdi95_upper"]
                ),
            }
        )

    return pd.DataFrame(records)


def _density(values, x):
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(np.unique(values)) <= 1:
        return None
    return gaussian_kde(values)(x)


def plot_validation_parameter_posteriors(
    full_inference_data,
    validation_inference_data,
    parameter_specs,
    output_path: str | Path,
    *,
    context_label: str,
):
    """Overlay full-data and held-out-refit parameter posteriors without the prior."""
    full = _posterior_components(full_inference_data, parameter_specs)
    validation = _posterior_components(validation_inference_data, parameter_specs)
    if tuple(full) != tuple(validation):
        raise ValueError("Full and validation posteriors do not have matching components.")

    labels = tuple(full)
    ncols = min(3, len(labels))
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.1 * nrows),
        squeeze=False,
    )

    for ax, label in zip(axes.flat, labels):
        _, full_values = full[label]
        _, validation_values = validation[label]
        combined = np.concatenate([full_values, validation_values])
        lower, upper = np.quantile(combined, [0.001, 0.999])
        if np.isclose(lower, upper):
            width = max(abs(float(lower)) * 0.05, 1e-9)
            lower -= width
            upper += width
        x = np.linspace(lower, upper, 500)

        full_density = _density(full_values, x)
        validation_density = _density(validation_values, x)

        if full_density is None:
            ax.axvline(full_values[0], color="0.35", linestyle="--", label="full fit")
        else:
            ax.fill_between(x, 0.0, full_density, color="0.65", alpha=0.22)
            ax.plot(x, full_density, color="0.35", linestyle="--", linewidth=1.4, label="full fit")

        if validation_density is None:
            ax.axvline(validation_values[0], color="C0", label="held-out refit")
        else:
            ax.fill_between(x, 0.0, validation_density, color="C0", alpha=0.18)
            ax.plot(x, validation_density, color="C0", linewidth=1.6, label="held-out refit")

        full_summary = summarize_scalar_samples(full_values)
        validation_summary = summarize_scalar_samples(validation_values)
        ax.axvline(full_summary["median"], color="0.35", linestyle=":", linewidth=1.0)
        ax.axvline(validation_summary["median"], color="C0", linestyle=":", linewidth=1.0)
        ax.set_title(label)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.15)

    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)

    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Validation posterior shift: {context_label}")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heldout_pit_conditions(
    pointwise,
    output_path: str | Path,
    *,
    context_label: str,
):
    """Plot predictive PIT values for observations that were genuinely held out."""
    koh_values = sorted(pointwise["electrolyte_concentration_M"].unique())
    co_values = sorted(pointwise["CO_mole_fraction"].unique())
    fig, axes = plt.subplots(
        len(co_values),
        len(koh_values),
        figsize=(3.8 * len(koh_values), 2.7 * len(co_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
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
                    curve["E_V_SHE"],
                    curve["heldout_pit"],
                    marker="o",
                    markersize=2.5,
                    linewidth=0.8,
                    alpha=0.75,
                    label=replicate,
                )
            ax.axhline(0.50, linestyle="--", linewidth=1.0, alpha=0.60)
            ax.axhline(0.05, linestyle=":", linewidth=0.8, alpha=0.40)
            ax.axhline(0.95, linestyle=":", linewidth=0.8, alpha=0.40)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.20)
            if row == 0:
                ax.set_title(f"{c_koh:g} M KOH")
            if col == len(koh_values) - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"{100 * co_fraction:g}% CO",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                )

    axes[0, 0].legend(title="replicate", fontsize=8)
    fig.supxlabel("Potential (V vs SHE)")
    fig.supylabel("held-out PIT")
    fig.suptitle(f"Held-out predictive PIT: {context_label}")
    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
