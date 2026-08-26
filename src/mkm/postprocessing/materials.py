"""Material-resolved summaries for multi-material posterior fits."""

import numpy as np
import pandas as pd

from mkm.postprocessing.diagnostics import summarize_scalar_samples


def summarize_setup_offsets(posterior, inputs, observations):
    """Summarize standardized setup latents and the actual zero-sum log-rate offsets."""
    required_variables = ("z_ln_rate_setup", "ln_rate_setup_offset")
    missing = [name for name in required_variables if name not in posterior]
    if missing:
        raise ValueError(f"Posterior is missing setup variables: {missing}.")

    if inputs.setup_labels is None or inputs.observation_setup_index is None:
        raise ValueError("Setup metadata are unavailable for this likelihood.")

    n_setups = len(inputs.setup_labels)
    observation_setup_index = np.asarray(inputs.observation_setup_index, dtype=np.int64)

    if len(observation_setup_index) != len(observations):
        raise ValueError("Observation setup indices do not align with the observation table.")

    metadata_columns = ["material", "electrolyte_concentration_M", "replicate"]
    missing_columns = [column for column in metadata_columns if column not in observations.columns]
    if missing_columns:
        raise ValueError(f"Observation table is missing setup metadata columns: {missing_columns}.")

    setup_metadata = observations[metadata_columns].copy()
    setup_metadata.insert(0, "setup_index", observation_setup_index)
    setup_metadata = setup_metadata.drop_duplicates().sort_values("setup_index").reset_index(drop=True)

    if len(setup_metadata) != n_setups or not np.array_equal(
        setup_metadata["setup_index"].to_numpy(dtype=np.int64),
        np.arange(n_setups, dtype=np.int64),
    ):
        raise ValueError("Setup indices do not map one-to-one onto setup metadata.")

    if inputs.setup_experiment_index is not None:
        setup_experiment_index = np.asarray(inputs.setup_experiment_index, dtype=np.int64)
        if len(setup_experiment_index) != n_setups:
            raise ValueError("Setup experiment indices do not align with setup labels.")
        setup_metadata.insert(1, "setup_experiment_index", setup_experiment_index)

    setup_metadata["setup_label"] = list(inputs.setup_labels)
    result = setup_metadata.copy()

    for variable_name, prefix in (
        ("z_ln_rate_setup", "z"),
        ("ln_rate_setup_offset", "offset"),
    ):
        data = posterior[variable_name]
        if "setup" not in data.dims:
            raise ValueError(f"Posterior variable '{variable_name}' does not have a setup dimension.")

        values = np.asarray(data, dtype=float)
        setup_axis = data.dims.index("setup")
        values = np.moveaxis(values, setup_axis, -1)

        if values.ndim != 3 or values.shape[-1] != n_setups:
            raise ValueError(
                f"Posterior variable '{variable_name}' must reduce to chain x draw x setup; got {values.shape}."
            )

        summaries = [
            summarize_scalar_samples(values[..., setup_index])
            for setup_index in range(n_setups)
        ]
        for statistic in ("mean", "sd", "median", "hdi95_lower", "hdi95_upper"):
            result[f"{prefix}_{statistic}"] = [summary[statistic] for summary in summaries]

    return result


def summarize_material_noise(posterior):
    records = []

    for variable in ("sigma_ln_rate_material", "ell_E_V_material", "sigma_ln_rate_setup_material"):
        if variable not in posterior:
            continue

        data = posterior[variable]

        if "material" not in data.dims:
            raise ValueError(f"Posterior variable '{variable}' does not have a material dimension.")

        materials = np.asarray(data.coords["material"]).astype(str)
        values = np.asarray(data, dtype=float)

        material_axis = data.dims.index("material")
        values = np.moveaxis(values, material_axis, -1)

        if values.ndim != 3:
            raise ValueError(
                f"Posterior variable '{variable}' must reduce to chain x draw x material; got {values.shape}."
            )

        for material_index, material in enumerate(materials):
            summary = summarize_scalar_samples(values[..., material_index])

            records.append(
                {
                    "material": material,
                    "variable": variable,
                    **summary,
                }
            )

    return pd.DataFrame(records)


def summarize_observation_diagnostics_by_material(observations):
    required = {
        "material",
        "residual_mechanism",
        "residual_conditional",
        "standardized_residual_conditional",
        "observed_inside_predictive_95_hdi",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Observation diagnostics are missing columns: {sorted(missing)}.")

    records = []

    for material, frame in observations.groupby("material", sort=False):
        residual_mechanism = frame["residual_mechanism"].to_numpy(dtype=float)
        residual_conditional = frame["residual_conditional"].to_numpy(dtype=float)
        standardized = frame["standardized_residual_conditional"].to_numpy(dtype=float)

        records.append(
            {
                "material": material,
                "n_observations": len(frame),
                "mechanism_residual_mean": float(np.mean(residual_mechanism)),
                "mechanism_residual_rms": float(np.sqrt(np.mean(residual_mechanism**2))),
                "conditional_residual_mean": float(np.mean(residual_conditional)),
                "conditional_residual_rms": float(np.sqrt(np.mean(residual_conditional**2))),
                "median_abs_standardized_residual": float(np.median(np.abs(standardized))),
                "predictive_95_hdi_coverage": float(frame["observed_inside_predictive_95_hdi"].mean()),
            }
        )

    return pd.DataFrame(records)


def summarize_residual_structure_by_material(curve_residuals, shared_residuals):
    materials = []

    if not curve_residuals.empty and "material" in curve_residuals:
        materials.extend(curve_residuals["material"].drop_duplicates().tolist())

    if not shared_residuals.empty and "material" in shared_residuals:
        materials.extend(shared_residuals["material"].drop_duplicates().tolist())

    materials = list(dict.fromkeys(materials))
    records = []

    for material in materials:
        record = {"material": material}

        curve = curve_residuals.loc[curve_residuals["material"] == material]
        if not curve.empty:
            record["median_lag1_residual_correlation"] = float(
                curve["lag1_residual_correlation"].median()
            )
            record["median_abs_residual_slope_per_V"] = float(
                curve["residual_slope_per_V"].abs().median()
            )

        shared = shared_residuals.loc[shared_residuals["material"] == material]
        if not shared.empty:
            record["median_shared_squared_residual_fraction"] = float(
                shared["shared_fraction_squared_residual"].median()
            )

        records.append(record)

    return pd.DataFrame(records)


def summarize_pointwise_loo_by_material(pointwise):
    required = {"material", "elpd_loo", "pareto_k"}
    missing = required - set(pointwise.columns)
    if missing:
        raise ValueError(f"Pointwise LOO table is missing columns: {sorted(missing)}.")

    records = []

    for material, frame in pointwise.groupby("material", sort=False):
        elpd = frame["elpd_loo"].to_numpy(dtype=float)
        pareto_k = frame["pareto_k"].to_numpy(dtype=float)

        records.append(
            {
                "material": material,
                "n_observations": len(frame),
                "elpd_loo_contribution": float(np.sum(elpd)),
                "mean_elpd_loo": float(np.mean(elpd)),
                "max_pareto_k": float(np.max(pareto_k)),
                "n_pareto_k_above_0p7": int(np.sum(pareto_k > 0.7)),
            }
        )

    return pd.DataFrame(records)
