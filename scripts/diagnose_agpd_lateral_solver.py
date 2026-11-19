"""Replay failed AgPd Frumkin prior-predictive points with more solver iterations.

Read-only diagnostic: uses the exact saved prior parameters and corresponding model points.
Does not re-sample priors, change model settings, or overwrite the original results.
"""

from argparse import ArgumentParser
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytensor
import xarray as xr

from mkm.constants import K_B_EV_K

from mkm.mechanisms.agpd_basic.lateral_interactions import evaluate_agpd_lateral_interactions
from mkm.mechanisms.agpd_basic.state import build_agpd_point_state
from mkm.model_inputs import build_model_input_arrays, build_model_point_inputs
from mkm.models.agpd_physics import parse_agpd_physics
from mkm.models.agpd_registry import get_agpd_model_definition
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import build_agpd_model_data, load_agpd_model_config


def failed_draw_groups(ln_rate):
    """Return (chain, draw, bad point indices) without losing the model-point axis."""
    values = np.asarray(ln_rate, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected (chain, draw, model_point) data; received shape {values.shape}.")
    return [(c, d, np.flatnonzero(~np.isfinite(values[c, d])))
            for c, d in np.argwhere(np.any(~np.isfinite(values), axis=2))]


def replay_failed_draw(prior, model_name, state, point_ids, chain, draw, temperature_K, solver):
    """Compile once and evaluate only the failed points for one saved prior draw."""
    definition = get_agpd_model_definition(model_name)
    parameter_values = {}
    for field in fields(definition.parameter_class):
        if field.name not in prior:
            raise ValueError(f"Saved prior is missing physical parameter '{field.name}'.")
        value = np.asarray(prior[field.name].values[chain, draw], dtype=float)
        if value.ndim != 0:
            raise ValueError(f"Expected a scalar prior parameter '{field.name}'; got {value.shape}.")
        parameter_values[field.name] = float(value)

    selected = replace(
        state,
        **{name: np.asarray(getattr(state, name))[point_ids]
           for name in ("E_V_SHE", "ln_a_CO", "ln_a_OH", "Pd_fraction", "Ag_fraction", "material_index")},
        theta_CO_max=(None if state.theta_CO_max is None else np.asarray(state.theta_CO_max)[point_ids]),
    )
    physics = parse_agpd_physics(model_name)
    result = evaluate_agpd_lateral_interactions(
        selected, definition.parameter_class(**parameter_values), temperature_K,
        pathways=physics.pathways, finite_steps=physics.finite_steps,
        interactions=physics.interactions, lateral_config={**solver, "diagnose_branches": True},
    )
    names = ("coverage_solver_relative_residual", "coverage_branch_disagreement",
             "coverage_solver_iterations",
             "theta_CO", "theta_OH_Pd", "theta_COOH", "theta_OH_Ag",
             "theta_empty_Pd", "lateral_mu_CO_eV",
             *(f"branch{branch}_update_error" for branch in (0, 1)),
             *(f"branch{branch}_balance_residual" for branch in (0, 1)),
             *(f"branch{branch}_theta_{species}" for branch in (0, 1)
               for species in ("CO", "pOH", "COOH", "sOH", "empty_Pd")))
    expression = [result.ln_rate, *(result.pointwise[name] for name in names)]
    evaluated = pytensor.function([], expression, on_unused_input="ignore")()
    diagnostics = {"ln_rate_model": np.asarray(evaluated[0]),
                   **{name: np.asarray(value) for name, value in zip(names, evaluated[1:])}}
    if "CO" not in physics.finite_steps:
        # Reconstruct vacant-site coverage from CO QEA without subtracting nearly equal coverages.
        c = diagnostics["theta_CO"]
        delta_g_co = parameter_values["deltaG1_0"] + diagnostics["lateral_mu_CO_eV"]
        diagnostics["theta_empty_Pd_from_CO_QEA"] = np.exp(
            np.log(np.maximum(c, 1e-300)) + delta_g_co / (K_B_EV_K * temperature_K)
            - np.asarray(selected.ln_a_CO)
        )
    return parameter_values, diagnostics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Existing interacting AgPd model name")
    parser.add_argument("--material", default="Ag50Pd50", help="Individual-material prior-predictive run")
    parser.add_argument("--iterations", type=int, default=384, help="Replay iteration count (default: 384)")
    parser.add_argument("--max-draws", type=int, default=2, help="Maximum failed draws to replay")
    parser.add_argument("--max-points", type=int, default=0, help="Limit failed points per draw; 0 = all")
    parser.add_argument("--point-ids", nargs="+", type=int,
                        help="Only replay these original failed point IDs (e.g. 64 115 488)")
    args = parser.parse_args()
    if args.iterations < 1 or args.max_draws < 1 or args.max_points < 0:
        parser.error("Iterations/max-draws must be positive; max-points must be nonnegative.")

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    output_dir = paths.agpd_prior_predictive_output_dir(
        fit_scope="individual", model_name=args.model, material=args.material,
    )
    saved = output_dir / "prior_predictive.nc"
    model_data = build_agpd_model_data(paths, (args.material,))
    point_inputs = build_model_point_inputs(build_model_input_arrays(model_data))
    state = build_agpd_point_state(point_inputs, config)
    solver = {**config["lateral_interactions"], "fixed_point_iterations": args.iterations,
              "diagnose_branches": True}
    print(f"Saved prior: {saved}")
    mode = "maximum" if solver.get("adaptive_convergence", False) else "fixed"
    print(f"Replay: {args.iterations} {mode} iterations, damping={solver['fixed_point_damping']}")

    with xr.open_dataset(saved, group="prior", engine="h5netcdf") as prior:
        ln_rate = np.asarray(prior["ln_rate_model"], dtype=float)
        if ln_rate.shape[-1] != len(model_data.model_points):
            raise ValueError("Saved model-point count does not match the current processed dataset.")
        groups = failed_draw_groups(ln_rate)
        print(f"Failed parameter draws: {len(groups)} / {ln_rate.shape[0] * ln_rate.shape[1]}")
        if not groups:
            return
        for chain, draw, ids in groups[:args.max_draws]:
            original_ids = ids.copy()
            if args.point_ids is not None:
                ids = ids[np.isin(ids, args.point_ids)]
            if args.max_points:
                ids = ids[:args.max_points]
            if len(ids) == 0:
                continue
            parameters, new = replay_failed_draw(
                prior, args.model, state, ids, chain, draw, config["temperature_K"], solver,
            )
            print(f"\nChain {chain}, draw {draw}: {len(original_ids)} originally failed points; replayed {len(ids)}")
            print("Parameters: " + ", ".join(f"{name}={value:.5g}" for name, value in parameters.items()))
            points = model_data.model_points.iloc[ids]
            table = pd.DataFrame({
                "point_id": ids,
                "E_V_SHE": points["E_V_SHE"].to_numpy(),
                "CO_fraction": points["CO_mole_fraction"].to_numpy(),
                "KOH_M": points["electrolyte_concentration_M"].to_numpy(),
                "old_residual": prior["coverage_solver_relative_residual"].values[chain, draw, ids],
                "new_residual": new["coverage_solver_relative_residual"],
                "old_branch_diff": prior["coverage_branch_disagreement"].values[chain, draw, ids],
                "new_branch_diff": new["coverage_branch_disagreement"],
                "now_finite": np.isfinite(new["ln_rate_model"]),
                "solver_iterations": new["coverage_solver_iterations"],
                "theta_CO": new["theta_CO"],
                "theta_pOH": new["theta_OH_Pd"],
                "theta_COOH": new["theta_COOH"],
                "theta_sOH": new["theta_OH_Ag"],
                "branch0_step_error": new["branch0_update_error"],
                "branch1_step_error": new["branch1_update_error"],
                "branch0_balance": new["branch0_balance_residual"],
                "branch1_balance": new["branch1_balance_residual"],
                "branch0_CO": new["branch0_theta_CO"],
                "branch1_CO": new["branch1_theta_CO"],
                "branch0_pOH": new["branch0_theta_pOH"],
                "branch1_pOH": new["branch1_theta_pOH"],
            })
            if "theta_empty_Pd_from_CO_QEA" in new:
                vacant = new["theta_empty_Pd_from_CO_QEA"]
                table["empty_from_sum"] = new["theta_empty_Pd"]
                table["empty_from_QEA"] = vacant
                table["empty_relative_diff"] = abs(new["theta_empty_Pd"] - vacant) / np.maximum(vacant, 1e-300)
            print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
            print(f"Recovered: {int(table['now_finite'].sum())} / {len(table)} failed points")
            if np.any(table["new_branch_diff"] > solver["branch_tolerance"]):
                print("Distinct iterates remain. Compare both branch step errors before inferring multiple roots.")
                print("Small fixed-point step errors do not establish physical or dynamical stability.")


if __name__ == "__main__":
    main()
