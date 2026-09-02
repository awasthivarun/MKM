"""Shared posterior sampling/checkpoint/finalization lifecycle."""

from dataclasses import dataclass
from shutil import rmtree
from time import perf_counter
from typing import Callable

import numpy as np

from mkm.inference.posterior import (
    add_log_likelihood,
    compute_posterior_deterministics,
    load_inference_data,
    retain_group_variables,
    sample_posterior,
    write_inference_data,
)
from mkm.postprocessing.diagnostics import build_posterior_parameter_summary
from mkm.postprocessing.sampling import build_sampling_health
from mkm.provenance import read_run_metadata, write_run_metadata


RUN_STATUS_SAMPLED = "sampled_free_variables"
RUN_STATUS_COMPLETE = "complete"


@dataclass(frozen=True)
class PosteriorLifecycleResult:
    inference_data: object
    metadata: dict
    sampling_health: dict
    sampling_seconds: float | None
    free_parameter_names: tuple[str, ...]


def _validate_controls(*, resume, overwrite):
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")


def _sample_or_load(
    *,
    built,
    output_dir,
    sampler,
    resume,
    overwrite,
    metadata_factory: Callable[[], dict],
    checkpoint_validator: Callable[[dict], None],
):
    _validate_controls(resume=resume, overwrite=overwrite)

    posterior_path = output_dir / "posterior.nc"
    metadata_path = output_dir / "run_metadata.yaml"

    if overwrite and output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        if not posterior_path.exists():
            raise FileNotFoundError(f"Posterior checkpoint not found: {posterior_path}")
        metadata = read_run_metadata(metadata_path)
        checkpoint_validator(metadata)
        inference_data = load_inference_data(posterior_path)
        return inference_data, metadata, None, posterior_path, metadata_path

    if posterior_path.exists():
        raise FileExistsError(
            f"Posterior already exists: {posterior_path}. Use --resume or --overwrite."
        )

    start = perf_counter()
    inference_data = sample_posterior(
        built,
        draws=sampler["draws"],
        tune=sampler["tune"],
        chains=sampler["chains"],
        cores=sampler["cores"],
        target_accept=sampler["target_accept"],
        random_seed=sampler["random_seed"],
        nuts_sampler=sampler["nuts_sampler"],
        backend=sampler["backend"],
        compute_convergence_checks=False,
    )
    sampling_seconds = perf_counter() - start

    write_inference_data(inference_data, posterior_path)
    metadata = metadata_factory()
    metadata["status"] = RUN_STATUS_SAMPLED
    write_run_metadata(metadata, metadata_path)
    return inference_data, metadata, sampling_seconds, posterior_path, metadata_path


def run_posterior_lifecycle(
    *,
    built,
    output_dir,
    parameter_specs,
    sampler,
    resume=False,
    overwrite=False,
    metadata_factory: Callable[[], dict],
    checkpoint_validator: Callable[[dict], None],
    progressbar=True,
):
    """Sample or resume a run and finalize one canonical posterior checkpoint."""
    inference_data, metadata, sampling_seconds, posterior_path, metadata_path = _sample_or_load(
        built=built,
        output_dir=output_dir,
        sampler=sampler,
        resume=resume,
        overwrite=overwrite,
        metadata_factory=metadata_factory,
        checkpoint_validator=checkpoint_validator,
    )

    free_parameter_names = tuple(variable.name for variable in built.model.free_RVs)
    missing = [
        name for name in free_parameter_names if name not in inference_data.posterior
    ]
    if missing:
        raise ValueError(f"Posterior checkpoint is missing fitted variables: {missing}")

    sampling_health = build_sampling_health(
        inference_data,
        parameter_names=free_parameter_names,
    )
    metadata["sampling_health"] = sampling_health
    metadata["status"] = RUN_STATUS_SAMPLED
    write_run_metadata(metadata, metadata_path)

    if "ln_rate_model" not in inference_data.posterior:
        posterior_for_check = compute_posterior_deterministics(
            inference_data,
            built,
            var_names=["ln_rate_model"],
            backend=sampler["backend"],
            progressbar=progressbar,
        )
    else:
        posterior_for_check = inference_data.posterior

    if not np.all(
        np.isfinite(np.asarray(posterior_for_check["ln_rate_model"], dtype=float))
    ):
        raise RuntimeError(
            'Posterior deterministic "ln_rate_model" contains non-finite values.'
        )

    inference_data.posterior = posterior_for_check
    if (
        not hasattr(inference_data, "log_likelihood")
        or "rate_observed" not in inference_data.log_likelihood
    ):
        inference_data = add_log_likelihood(
            inference_data,
            built,
            backend=sampler["backend"],
            progressbar=progressbar,
        )

    log_likelihood = np.asarray(
        inference_data.log_likelihood["rate_observed"],
        dtype=float,
    )
    if not np.all(np.isfinite(log_likelihood)):
        raise RuntimeError(
            'Posterior log likelihood "rate_observed" contains non-finite values.'
        )

    inference_data = retain_group_variables(
        inference_data,
        "posterior",
        free_parameter_names,
    )
    write_inference_data(inference_data, posterior_path)

    parameter_summary = build_posterior_parameter_summary(
        inference_data,
        parameter_specs,
    )
    parameter_summary.to_csv(output_dir / "posterior_parameters.csv", index=False)

    metadata["status"] = RUN_STATUS_COMPLETE
    metadata["sampling_health"] = sampling_health
    metadata["stored_posterior_variables"] = list(
        inference_data.posterior.data_vars
    )
    write_run_metadata(metadata, metadata_path)

    return PosteriorLifecycleResult(
        inference_data=inference_data,
        metadata=metadata,
        sampling_health=sampling_health,
        sampling_seconds=sampling_seconds,
        free_parameter_names=free_parameter_names,
    )
