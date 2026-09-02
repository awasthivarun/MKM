"""Posterior transition-state degree-of-rate-control calculations."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import xarray as xr

from mkm.constants import K_B_EV_K
from mkm.mechanisms.agpd_basic import build_agpd_point_state
from mkm.models.agpd_basic import get_agpd_parameterization, get_agpd_model_definition
from mkm.postprocessing.diagnostics import summarize_samples


@dataclass(frozen=True)
class TransitionStateControl:
    name: str
    parameter: str
    label: str


@dataclass(frozen=True)
class TransitionStateDRC:
    draws: xr.Dataset
    summary: pd.DataFrame
    checks: pd.DataFrame


_TRANSITION_STATE_CONTROLS = {
    "BF": (
        TransitionStateControl(name="BF", parameter="Gact2_0", label="BF"),
    ),
    "BF_LH": (
        TransitionStateControl(name="BF", parameter="Gact2_BF_0", label="BF"),
        TransitionStateControl(name="LH", parameter="Gact2_LH_0", label="LH"),
    ),
    "CO_BF_ER_LH": (
        TransitionStateControl(name="CO_adsorption", parameter="Gact1_0", label="CO adsorption/desorption"),
        TransitionStateControl(name="BF", parameter="Gact2_BF_0", label="BF"),
        TransitionStateControl(name="ER", parameter="Gact2_ER_0", label="ER"),
        TransitionStateControl(name="LH", parameter="Gact2_LH_0", label="LH"),
    ),
    "CO_ER_LH": (
        TransitionStateControl(name="CO_adsorption", parameter="Gact1_0", label="CO adsorption/desorption"),
        TransitionStateControl(name="ER", parameter="Gact2_ER_0", label="ER"),
        TransitionStateControl(name="LH", parameter="Gact2_LH_0", label="LH"),
    ),
}


def transition_state_controls(model_name):
    try:
        return _TRANSITION_STATE_CONTROLS[model_name]
    except KeyError as error:
        available = tuple(_TRANSITION_STATE_CONTROLS)
        raise ValueError(f"No transition-state DRC definition for '{model_name}'. Available: {available}.") from error


def _compile_log_rate_evaluator(model_name, point_inputs, config):
    definition = get_agpd_model_definition(model_name)
    parameter_names = tuple(field.name for field in fields(definition.parameter_class))
    parameter_symbols = {name: pt.dscalar(name) for name in parameter_names}
    parameters = definition.parameter_class(**parameter_symbols)

    state = build_agpd_point_state(inputs=point_inputs, config=config)
    evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])

    function = pytensor.function(
        [parameter_symbols[name] for name in parameter_names],
        evaluated.mechanism_result.ln_rate,
        on_unused_input="raise",
    )
    return parameter_names, function


def _posterior_dataset(posterior):
    if isinstance(posterior, xr.Dataset):
        return posterior
    if hasattr(posterior, "to_dataset"):
        return posterior.to_dataset()
    raise TypeError("Posterior must be an xarray Dataset or DataTree node.")


def _compile_pointwise_log_rate_evaluator(model_name, point_inputs, config):
    """Compile a mechanism evaluator accepting one value per model point for every parameter."""
    definition = get_agpd_model_definition(model_name)
    parameter_names = tuple(field.name for field in fields(definition.parameter_class))
    parameter_symbols = {name: pt.dvector(name) for name in parameter_names}
    parameters = definition.parameter_class(**parameter_symbols)

    state = build_agpd_point_state(inputs=point_inputs, config=config)
    evaluated = definition.evaluator(state=state, parameters=parameters, temperature_K=config["temperature_K"])

    function = pytensor.function(
        [parameter_symbols[name] for name in parameter_names],
        evaluated.mechanism_result.ln_rate,
        on_unused_input="raise",
    )
    return parameter_names, function, state


def _posterior_scalar_parameter_draws(posterior, parameter_names):
    posterior = _posterior_dataset(posterior)
    n_chains = int(posterior.sizes["chain"])
    n_draws = int(posterior.sizes["draw"])
    result = {}

    for name in parameter_names:
        if name not in posterior:
            raise ValueError(f"Posterior is missing mechanism parameter '{name}'.")

        values = np.asarray(posterior[name], dtype=float)
        if values.shape[:2] != (n_chains, n_draws):
            raise ValueError(
                f"Posterior parameter '{name}' has leading shape {values.shape[:2]}, "
                f"expected {(n_chains, n_draws)}."
            )

        trailing_size = int(np.prod(values.shape[2:], dtype=int)) if values.ndim > 2 else 1
        if trailing_size != 1:
            raise ValueError(
                f"DRC currently requires scalar mechanism parameters; '{name}' has shape {values.shape[2:]}."
            )

        values = values.reshape(n_chains, n_draws)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Posterior parameter '{name}' contains non-finite values.")

        result[name] = values

    return result


def _evaluate_parameter_set(function, parameter_names, parameter_values):
    ln_rate = np.asarray(function(*(parameter_values[name] for name in parameter_names)), dtype=float)
    if ln_rate.ndim != 1:
        raise ValueError(f"Mechanism evaluator returned log-rate shape {ln_rate.shape}; expected one dimension.")
    if not np.all(np.isfinite(ln_rate)):
        raise ValueError("Mechanism evaluator returned non-finite log rates during DRC perturbation.")
    return ln_rate


def _summarize_transition_state_draws(draws, model_points):
    values = np.asarray(draws["X_TS"], dtype=float)
    n_chains, n_draws, n_controls, n_points = values.shape
    samples = values.reshape(n_chains * n_draws, n_controls, n_points)

    summary = summarize_samples(samples)

    metadata = model_points.sort_values("model_point_id").reset_index(drop=True).copy()
    if len(metadata) != n_points:
        raise ValueError(f"DRC has {n_points} model points but metadata has {len(metadata)} rows.")

    records = []
    for control_index, control_name in enumerate(draws.coords["control"].values.astype(str)):
        control_parameter = str(draws.coords["control_parameter"].values[control_index])
        control_label = str(draws.coords["control_label"].values[control_index])

        frame = metadata.copy()
        frame["control_type"] = "transition_state"
        frame["control"] = control_name
        frame["parameter"] = control_parameter
        frame["label"] = control_label
        frame["mean"] = summary["mean"][control_index]
        frame["sd"] = summary["sd"][control_index]
        frame["median"] = summary["median"][control_index]
        frame["hdi95_lower"] = summary["hdi95_lower"][control_index]
        frame["hdi95_upper"] = summary["hdi95_upper"][control_index]
        records.append(frame)

    return pd.concat(records, ignore_index=True)


def _build_transition_state_checks(draws, step_eV):
    values = np.asarray(draws["X_TS"], dtype=float)
    sum_X_TS = np.sum(values, axis=2)
    abs_error = np.abs(sum_X_TS - 1.0)

    return pd.DataFrame(
        [
            {
                "control_type": "transition_state",
                "n_controls": int(values.shape[2]),
                "step_eV": float(step_eV),
                "median_abs_sum_error": float(np.median(abs_error)),
                "q999_abs_sum_error": float(np.quantile(abs_error, 0.999)),
                "max_abs_sum_error": float(np.max(abs_error)),
                "minimum_X_TS": float(np.min(values)),
                "maximum_X_TS": float(np.max(values)),
                "fraction_nonfinite": float(np.mean(~np.isfinite(values))),
            }
        ]
    )


def compute_transition_state_drc(
    inference_data,
    model_name,
    point_inputs,
    model_points,
    config,
    step_eV=1e-4,
):
    """Compute posterior transition-state DRCs by central finite differences in barrier energy.

    For transition state j,

        X_TS,j = -k_B T * d ln(r) / d G_TS,j

    where the corresponding fitted activation-energy parameter is shifted by +/- step_eV
    while all equilibrium free energies and other transition-state energies are held fixed.
    """
    step_eV = float(step_eV)
    if not np.isfinite(step_eV) or step_eV <= 0:
        raise ValueError("DRC perturbation step must be finite and positive.")

    controls = transition_state_controls(model_name)
    parameter_names, function = _compile_log_rate_evaluator(model_name, point_inputs, config)
    posterior = _posterior_dataset(inference_data.posterior)
    posterior_draws = _posterior_scalar_parameter_draws(posterior, parameter_names)

    missing_controls = [control.parameter for control in controls if control.parameter not in parameter_names]
    if missing_controls:
        raise ValueError(f"DRC control parameters are not mechanism parameters for '{model_name}': {missing_controls}")

    n_chains = int(posterior.sizes["chain"])
    n_draws = int(posterior.sizes["draw"])
    n_points = len(point_inputs.E_V_SHE)
    X_TS = np.empty((n_chains, n_draws, len(controls), n_points), dtype=float)
    kBT_eV = K_B_EV_K * float(config["temperature_K"])

    for chain in range(n_chains):
        for draw in range(n_draws):
            base = {name: float(posterior_draws[name][chain, draw]) for name in parameter_names}

            for control_index, control in enumerate(controls):
                plus = dict(base)
                minus = dict(base)
                plus[control.parameter] += step_eV
                minus[control.parameter] -= step_eV

                ln_rate_plus = _evaluate_parameter_set(function, parameter_names, plus)
                ln_rate_minus = _evaluate_parameter_set(function, parameter_names, minus)
                derivative = (ln_rate_plus - ln_rate_minus) / (2.0 * step_eV)
                X_TS[chain, draw, control_index] = -kBT_eV * derivative

    coords = {
        "chain": np.asarray(posterior.coords["chain"]),
        "draw": np.asarray(posterior.coords["draw"]),
        "control": [control.name for control in controls],
        "model_point": np.arange(n_points, dtype=np.int64),
        "control_parameter": ("control", [control.parameter for control in controls]),
        "control_label": ("control", [control.label for control in controls]),
    }
    dataset = xr.Dataset(
        data_vars={"X_TS": (("chain", "draw", "control", "model_point"), X_TS)},
        coords=coords,
        attrs={
            "model_name": model_name,
            "temperature_K": float(config["temperature_K"]),
            "step_eV": step_eV,
            "definition": "X_TS = -k_B*T*d ln(rate)/d G_TS",
        },
    )

    summary = _summarize_transition_state_draws(dataset, model_points=model_points)
    checks = _build_transition_state_checks(dataset, step_eV=step_eV)
    return TransitionStateDRC(draws=dataset, summary=summary, checks=checks)


def compute_composition_transition_state_drc(
    inference_data,
    model_name,
    point_inputs,
    model_points,
    config,
    parameterization,
    step_eV=1e-4,
):
    """Compute TS DRCs using the effective transition-state energy at each composition.

    For shared energetics this reduces to the ordinary transition-state DRC. For linear_xAg,
    fitted reference values and slopes first generate pointwise effective mechanism parameters,

        p(x_Ag) = p_ref + s_p * (x_Ag - x_ref),

    and the finite-difference perturbation is then applied to the effective transition-state
    energy at every model point. The reported DRC remains the conventional local TS-energy DRC,
    not a sensitivity with respect to the fitted slope.
    """
    if parameterization == "shared":
        return compute_transition_state_drc(
            inference_data=inference_data,
            model_name=model_name,
            point_inputs=point_inputs,
            model_points=model_points,
            config=config,
            step_eV=step_eV,
        )

    step_eV = float(step_eV)
    if not np.isfinite(step_eV) or step_eV <= 0:
        raise ValueError("DRC perturbation step must be finite and positive.")

    x_reference, slope_specs = get_agpd_parameterization(
        config=config,
        model_name=model_name,
        parameterization=parameterization,
    )
    if not slope_specs:
        raise ValueError(
            f"Composition parameterization '{parameterization}' does not define slopes for '{model_name}'."
        )

    controls = transition_state_controls(model_name)
    parameter_names, function, state = _compile_pointwise_log_rate_evaluator(model_name, point_inputs, config)
    posterior = _posterior_dataset(inference_data.posterior)

    slope_names = tuple(f"{name}_xAg_slope" for name in slope_specs)
    posterior_draws = _posterior_scalar_parameter_draws(
        posterior,
        (*parameter_names, *slope_names),
    )

    missing_controls = [control.parameter for control in controls if control.parameter not in parameter_names]
    if missing_controls:
        raise ValueError(f"DRC control parameters are not mechanism parameters for '{model_name}': {missing_controls}")

    n_chains = int(posterior.sizes["chain"])
    n_draws = int(posterior.sizes["draw"])
    n_points = len(point_inputs.E_V_SHE)
    x_shift = np.asarray(state.Ag_fraction, dtype=float) - float(x_reference)

    X_TS = np.empty((n_chains, n_draws, len(controls), n_points), dtype=float)
    kBT_eV = K_B_EV_K * float(config["temperature_K"])

    for chain in range(n_chains):
        for draw in range(n_draws):
            effective = {
                name: np.full(n_points, posterior_draws[name][chain, draw], dtype=float)
                for name in parameter_names
            }

            for parameter_name in slope_specs:
                slope_name = f"{parameter_name}_xAg_slope"
                effective[parameter_name] = (
                    posterior_draws[parameter_name][chain, draw]
                    + posterior_draws[slope_name][chain, draw] * x_shift
                )

            for control_index, control in enumerate(controls):
                plus = dict(effective)
                minus = dict(effective)
                plus[control.parameter] = effective[control.parameter] + step_eV
                minus[control.parameter] = effective[control.parameter] - step_eV

                ln_rate_plus = _evaluate_parameter_set(function, parameter_names, plus)
                ln_rate_minus = _evaluate_parameter_set(function, parameter_names, minus)
                derivative = (ln_rate_plus - ln_rate_minus) / (2.0 * step_eV)
                X_TS[chain, draw, control_index] = -kBT_eV * derivative

    coords = {
        "chain": np.asarray(posterior.coords["chain"]),
        "draw": np.asarray(posterior.coords["draw"]),
        "control": [control.name for control in controls],
        "model_point": np.arange(n_points, dtype=np.int64),
        "control_parameter": ("control", [control.parameter for control in controls]),
        "control_label": ("control", [control.label for control in controls]),
    }
    dataset = xr.Dataset(
        data_vars={"X_TS": (("chain", "draw", "control", "model_point"), X_TS)},
        coords=coords,
        attrs={
            "model_name": model_name,
            "parameterization": parameterization,
            "x_reference": float(x_reference),
            "temperature_K": float(config["temperature_K"]),
            "step_eV": step_eV,
            "definition": "X_TS = -k_B*T*d ln(rate)/d G_TS_effective(x_Ag)",
        },
    )

    summary = _summarize_transition_state_draws(dataset, model_points=model_points)
    checks = _build_transition_state_checks(dataset, step_eV=step_eV)
    checks.insert(1, "parameterization", parameterization)
    return TransitionStateDRC(draws=dataset, summary=summary, checks=checks)


def compare_transition_state_drc_steps(reference, comparison):
    """Compare two DRC calculations performed with different finite-difference steps."""
    ref = reference.draws["X_TS"]
    other = comparison.draws["X_TS"]

    if ref.dims != other.dims or ref.shape != other.shape:
        raise ValueError("DRC step-convergence comparison requires identical dimensions and shapes.")
    if not np.array_equal(ref.coords["control"], other.coords["control"]):
        raise ValueError("DRC step-convergence comparison requires identical controls.")

    difference = np.abs(np.asarray(ref, dtype=float) - np.asarray(other, dtype=float))
    records = []

    for index, control in enumerate(ref.coords["control"].values.astype(str)):
        values = difference[:, :, index, :].reshape(-1)
        records.append(
            {
                "control": control,
                "reference_step_eV": float(reference.draws.attrs["step_eV"]),
                "comparison_step_eV": float(comparison.draws.attrs["step_eV"]),
                "median_abs_difference": float(np.median(values)),
                "q999_abs_difference": float(np.quantile(values, 0.999)),
                "max_abs_difference": float(np.max(values)),
            }
        )

    return pd.DataFrame(records)
