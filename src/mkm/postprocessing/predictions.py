from dataclasses import dataclass

import numpy as np

from mkm.inference.likelihoods import get_observation_material_index
from mkm.postprocessing.diagnostics import flatten_posterior_samples, summarize_samples


@dataclass(frozen=True)
class ObservationDistributionDraws:
    model_rate: np.ndarray
    sigma_rate: np.ndarray
    predictive_rate: np.ndarray | None


def _posterior_dataset(inference_data):
    return inference_data.posterior


def _observation_error_draws(posterior, name, inputs):
    if name not in posterior:
        raise ValueError(f"Posterior is missing error parameter '{name}'.")

    data = posterior[name]
    extra_dims = [dim for dim in data.dims if dim not in {"chain", "draw"}]
    if not extra_dims:
        return np.asarray(data.transpose("chain", "draw"), dtype=float)[..., None]

    if extra_dims != ["material"]:
        raise ValueError(
            f"Error parameter '{name}' has unsupported dimensions: {tuple(data.dims)}."
        )

    coordinate_materials = tuple(str(value) for value in data.coords["material"].values)
    missing = [material for material in inputs.materials if material not in coordinate_materials]
    if missing:
        raise ValueError(f"Posterior error parameter '{name}' is missing materials: {missing}.")

    material_lookup = {
        material: index for index, material in enumerate(coordinate_materials)
    }
    ordered = np.asarray(data.transpose("chain", "draw", "material"), dtype=float)
    input_material_draws = ordered[
        ...,
        [material_lookup[material] for material in inputs.materials],
    ]
    observation_material_index = get_observation_material_index(inputs)
    return input_material_draws[..., observation_material_index]


def build_observation_distribution_draws(
    inference_data,
    inputs,
    *,
    random_seed=20260826,
    sample_predictive=True,
):
    posterior = _posterior_dataset(inference_data)
    if "ln_rate_model" not in posterior:
        raise ValueError("Posterior is missing 'ln_rate_model'.")

    ln_rate_model = np.asarray(posterior["ln_rate_model"], dtype=float)
    observation_model_point_index = np.asarray(
        inputs.observation_model_point_index, dtype=np.int64
    )
    model_rate = np.exp(ln_rate_model[..., observation_model_point_index])

    sigma_abs = _observation_error_draws(posterior, "sigma_rate_abs", inputs)
    sigma_rel = _observation_error_draws(posterior, "sigma_rate_rel", inputs)
    sigma_rate = sigma_abs + sigma_rel * model_rate

    if not np.all(np.isfinite(model_rate)) or np.any(model_rate <= 0):
        raise ValueError("Posterior model rates must be finite and positive.")
    if not np.all(np.isfinite(sigma_rate)) or np.any(sigma_rate <= 0):
        raise ValueError("Posterior rate standard deviations must be finite and positive.")

    predictive_rate = None
    if sample_predictive:
        rng = np.random.default_rng(random_seed)
        predictive_rate = model_rate + sigma_rate * rng.standard_normal(model_rate.shape)

    return ObservationDistributionDraws(
        model_rate=model_rate,
        sigma_rate=sigma_rate,
        predictive_rate=predictive_rate,
    )


def _append_summary(frame, prefix, values):
    summary = summarize_samples(flatten_posterior_samples(values))
    for statistic, statistic_values in summary.items():
        frame[f"{prefix}_{statistic}"] = statistic_values


def build_observation_diagnostics(
    inference_data,
    model_data,
    inputs,
    *,
    random_seed=20260826,
    sample_predictive=True,
):
    draws = build_observation_distribution_draws(
        inference_data,
        inputs,
        random_seed=random_seed,
        sample_predictive=sample_predictive,
    )

    result = model_data.observations.copy()
    result["rate"] = result["rate_s_inv"].to_numpy(dtype=float)
    _append_summary(result, "rate_model", draws.model_rate)
    _append_summary(result, "sigma_rate", draws.sigma_rate)

    if draws.predictive_rate is not None:
        _append_summary(result, "rate_predictive", draws.predictive_rate)

    result["residual"] = result["rate"] - result["rate_model_median"]
    result["standardized_residual"] = result["residual"] / result["sigma_rate_median"]

    if draws.predictive_rate is not None:
        result["observed_inside_predictive_95_hdi"] = (
            (result["rate"] >= result["rate_predictive_hdi95_lower"])
            & (result["rate"] <= result["rate_predictive_hdi95_upper"])
        )

    return result
