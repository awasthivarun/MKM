"""Cross-validation splits and held-out prediction for AgPd all-material models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import logsumexp, ndtr

from mkm.inference.likelihoods import get_observation_material_index
from mkm.mechanisms.base import MechanismResult, validate_mechanism_result
from mkm.model_inputs import build_model_coords, build_model_point_inputs
from mkm.postprocessing.diagnostics import flatten_posterior_samples, summarize_samples


@dataclass(frozen=True)
class HeldoutPrediction:
    pointwise: pd.DataFrame
    summary: pd.DataFrame


def split_agpd_lomo(selected, material):
    """Hold out every observation from one material."""
    heldout_mask = selected["material"].eq(material)
    if not heldout_mask.any():
        raise ValueError(f"No observations found for held-out material '{material}'.")

    train = selected.loc[~heldout_mask].copy()
    heldout = selected.loc[heldout_mask].copy()
    if train.empty:
        raise ValueError("LOMO training data are empty.")
    return train, heldout


def split_agpd_loco(selected, material, koh_M, co_mole_fraction):
    """Hold out one material/KOH/CO condition, including all replicates."""
    heldout_mask = (
        selected["material"].eq(material)
        & np.isclose(
            selected["C_KOH_M"].to_numpy(dtype=float),
            float(koh_M),
            rtol=0.0,
            atol=1e-12,
        )
        & np.isclose(
            selected["CO_mole_fraction"].to_numpy(dtype=float),
            float(co_mole_fraction),
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not heldout_mask.any():
        raise ValueError(
            f"No observations found for held-out condition: material={material}, "
            f"KOH={koh_M}, CO={co_mole_fraction}."
        )

    train = selected.loc[~heldout_mask].copy()
    heldout = selected.loc[heldout_mask].copy()
    if train.empty:
        raise ValueError("LOCO training data are empty.")

    condition_count = heldout[
        ["material", "C_KOH_M", "CO_mole_fraction"]
    ].drop_duplicates().shape[0]
    if condition_count != 1:
        raise RuntimeError("LOCO split did not isolate exactly one experimental condition.")
    return train, heldout


def validate_validation_error_structure(scheme, error_structure):
    if scheme not in {"loco", "lomo"}:
        raise ValueError("Validation scheme must be 'loco' or 'lomo'.")
    if error_structure not in {"shared", "material"}:
        raise ValueError("Error structure must be 'shared' or 'material'.")
    if scheme == "lomo" and error_structure == "material":
        raise ValueError(
            "LOMO cannot use independent material-specific errors: the held-out "
            "material has no fitted sigma_rate_abs or sigma_rate_rel. Use shared error "
            "or introduce a hierarchical error model before attempting this comparison."
        )


def validate_heldout_error_support(
    error_structure,
    training_materials,
    heldout_materials,
):
    """Fail before sampling when independent errors cannot cover held-out data."""
    if error_structure == "shared":
        return
    if error_structure != "material":
        raise ValueError("Error structure must be 'shared' or 'material'.")

    training_materials = set(training_materials)
    missing = [
        material
        for material in heldout_materials
        if material not in training_materials
    ]
    if missing:
        raise ValueError(
            "Independent material-specific errors cannot predict held-out materials "
            f"that are absent from training: {missing}. Use shared error or introduce "
            "a hierarchical error model."
        )


def build_mechanism_prediction_model(inputs, mechanism):
    """Build a prediction-only PyMC model containing physical parameters and ln(rate)."""
    coords = build_model_coords(inputs)
    point_inputs = build_model_point_inputs(inputs)
    n_model_points = len(point_inputs.E_V_SHE)

    with pm.Model(coords=coords) as model:
        result = mechanism(point_inputs)
        if not isinstance(result, MechanismResult):
            raise TypeError("Mechanism must return a MechanismResult.")
        validate_mechanism_result(result)

        ln_rate = pt.as_tensor_variable(result.ln_rate)
        if ln_rate.ndim != 1:
            raise ValueError("Prediction ln_rate must be one-dimensional.")
        ln_rate = pt.specify_shape(ln_rate, (n_model_points,))
        pm.Deterministic("ln_rate_model", ln_rate, dims="model_point")

    return model


def _posterior_dataset(inference_data):
    posterior = inference_data.posterior
    if isinstance(posterior, xr.Dataset):
        return posterior
    return posterior.to_dataset()


def _error_draws_for_observations(posterior, name, heldout_inputs, error_structure):
    if name not in posterior:
        raise ValueError(f"Posterior is missing error parameter '{name}'.")

    data = posterior[name]
    if error_structure == "shared":
        unexpected = set(data.dims) - {"chain", "draw"}
        if unexpected:
            raise ValueError(
                f"Shared error parameter '{name}' has unexpected dimensions: "
                f"{sorted(unexpected)}."
            )
        values = np.asarray(data.transpose("chain", "draw"), dtype=float)
        return values[..., None]

    if "material" not in data.dims:
        raise ValueError(
            f"Material-specific error parameter '{name}' has no material dimension."
        )

    coordinate_materials = tuple(str(value) for value in data.coords["material"].values)
    missing = [
        material
        for material in heldout_inputs.materials
        if material not in coordinate_materials
    ]
    if missing:
        raise ValueError(
            f"Held-out materials have no fitted '{name}' values: {missing}. "
            "Independent material errors cannot be extrapolated to an omitted material."
        )

    ordered = data.transpose("chain", "draw", "material")
    values = np.asarray(ordered, dtype=float)
    material_lookup = {
        material: index for index, material in enumerate(coordinate_materials)
    }
    heldout_material_values = values[
        ...,
        [material_lookup[material] for material in heldout_inputs.materials],
    ]
    observation_material_index = get_observation_material_index(heldout_inputs)
    return heldout_material_values[..., observation_material_index]


def _add_summary(frame, prefix, draws):
    summary = summarize_samples(flatten_posterior_samples(draws))
    for statistic, values in summary.items():
        frame[f"{prefix}_{statistic}"] = values


def compute_heldout_rate_predictions(
    inference_data,
    prediction_model,
    heldout_data,
    heldout_inputs,
    *,
    error_structure,
    random_seed=20260826,
    backend="numba",
    progressbar=True,
):
    """Compute held-out mechanism and posterior-predictive rate distributions."""
    posterior = _posterior_dataset(inference_data)
    free_names = tuple(variable.name for variable in prediction_model.free_RVs)
    missing = [name for name in free_names if name not in posterior]
    if missing:
        raise ValueError(f"Training posterior is missing prediction parameters: {missing}.")

    with prediction_model:
        predicted = pm.compute_deterministics(
            posterior[list(free_names)],
            var_names=["ln_rate_model"],
            model=prediction_model,
            extend_dataset=False,
            progressbar=progressbar,
            backend=backend,
        )

    ln_rate_model = np.asarray(predicted["ln_rate_model"], dtype=float)
    observation_model_point_index = np.asarray(
        heldout_inputs.observation_model_point_index, dtype=np.int64
    )
    model_rate = np.exp(ln_rate_model[..., observation_model_point_index])

    sigma_abs = _error_draws_for_observations(
        posterior, "sigma_rate_abs", heldout_inputs, error_structure
    )
    sigma_rel = _error_draws_for_observations(
        posterior, "sigma_rate_rel", heldout_inputs, error_structure
    )
    sigma_rate = sigma_abs + sigma_rel * model_rate
    if not np.all(np.isfinite(sigma_rate)) or np.any(sigma_rate <= 0):
        raise ValueError("Held-out posterior standard deviations must be finite and positive.")

    observed_rate = heldout_data.observations["rate_s_inv"].to_numpy(dtype=float)
    observed_rate_broadcast = observed_rate.reshape((1, 1, -1))
    standardized_draws = (observed_rate_broadcast - model_rate) / sigma_rate
    log_density_draws = (
        -0.5 * standardized_draws**2
        - np.log(sigma_rate)
        - 0.5 * np.log(2.0 * np.pi)
    )
    flat_log_density = log_density_draws.reshape((-1, log_density_draws.shape[-1]))
    heldout_log_predictive_density = (
        logsumexp(flat_log_density, axis=0) - np.log(flat_log_density.shape[0])
    )
    heldout_pit = np.mean(ndtr(standardized_draws), axis=(0, 1))

    rng = np.random.default_rng(random_seed)
    predictive_rate = model_rate + sigma_rate * rng.standard_normal(model_rate.shape)

    pointwise = heldout_data.observations.copy()
    pointwise["rate"] = observed_rate
    pointwise["heldout_log_predictive_density"] = heldout_log_predictive_density
    pointwise["heldout_pit"] = heldout_pit
    _add_summary(pointwise, "rate_model", model_rate)
    _add_summary(pointwise, "sigma_rate", sigma_rate)
    _add_summary(pointwise, "rate_predictive", predictive_rate)

    pointwise["model_residual"] = pointwise["rate"] - pointwise["rate_model_median"]
    pointwise["standardized_model_residual"] = (
        pointwise["model_residual"] / pointwise["sigma_rate_median"]
    )
    pointwise["observed_inside_model_95_hdi"] = (
        (pointwise["rate"] >= pointwise["rate_model_hdi95_lower"])
        & (pointwise["rate"] <= pointwise["rate_model_hdi95_upper"])
    )
    pointwise["observed_inside_predictive_95_hdi"] = (
        (pointwise["rate"] >= pointwise["rate_predictive_hdi95_lower"])
        & (pointwise["rate"] <= pointwise["rate_predictive_hdi95_upper"])
    )

    residual = pointwise["model_residual"].to_numpy(dtype=float)
    standardized = pointwise["standardized_model_residual"].to_numpy(dtype=float)
    predictive_width = (
        pointwise["rate_predictive_hdi95_upper"]
        - pointwise["rate_predictive_hdi95_lower"]
    ).to_numpy(dtype=float)

    summary = pd.DataFrame(
        [
            {
                "n_observations": len(pointwise),
                "n_model_points": len(heldout_data.model_points),
                "model_residual_mean": float(np.mean(residual)),
                "model_residual_rmse": float(np.sqrt(np.mean(residual**2))),
                "model_residual_mae": float(np.mean(np.abs(residual))),
                "median_abs_model_residual": float(np.median(np.abs(residual))),
                "standardized_residual_mean": float(np.mean(standardized)),
                "standardized_residual_rms": float(np.sqrt(np.mean(standardized**2))),
                "model_95_hdi_observation_coverage": float(
                    pointwise["observed_inside_model_95_hdi"].mean()
                ),
                "posterior_predictive_95_hdi_coverage": float(
                    pointwise["observed_inside_predictive_95_hdi"].mean()
                ),
                "mean_predictive_95_hdi_width": float(np.mean(predictive_width)),
                "heldout_log_predictive_density_sum": float(
                    np.sum(heldout_log_predictive_density)
                ),
                "heldout_log_predictive_density_mean": float(
                    np.mean(heldout_log_predictive_density)
                ),
                "heldout_pit_mean": float(np.mean(heldout_pit)),
            }
        ]
    )

    return HeldoutPrediction(pointwise=pointwise, summary=summary)
