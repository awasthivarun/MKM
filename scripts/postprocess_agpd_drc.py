"""Compute and persist transition-state DRCs for an AgPd posterior fit."""

from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import pandas as pd
import yaml

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays, build_model_point_inputs
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.drc import compare_transition_state_drc_steps, compute_transition_state_drc
from mkm.postprocessing.plotting import plot_transition_state_drc


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis"
DATA_PATH = ANALYSIS_ROOT / "AgPd_COOx_basic_selected.parquet"
CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
POSTERIOR_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--likelihood", choices=["iid", "setup_intercept"], default="setup_intercept")
    parser.add_argument("--step-eV", type=float, default=1e-4)
    parser.add_argument("--check-half-step", action="store_true")
    return parser.parse_args()


def _get_posterior_dir(model_name, likelihood_name):
    if likelihood_name == "setup_intercept":
        path = POSTERIOR_ROOT / MATERIAL / "setup_intercept" / model_name
    else:
        new_path = POSTERIOR_ROOT / MATERIAL / "iid" / model_name
        legacy_path = POSTERIOR_ROOT / MATERIAL / model_name
        path = new_path if (new_path / "posterior.nc").exists() else legacy_path

    posterior_path = path / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior not found: {posterior_path}")
    return path


def main():
    args = parse_args()

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    posterior_dir = _get_posterior_dir(args.model, args.likelihood)
    output_dir = posterior_dir / "postprocessing"
    tables_dir = output_dir / "tables"
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"

    for path in (tables_dir, derived_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    selected = pd.read_parquet(DATA_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()
    model_data = build_model_data(selected_replicates=selected, electrolyte_concentration_column="C_KOH_M")
    inputs = build_model_input_arrays(model_data)
    point_inputs = build_model_point_inputs(inputs)
    idata = az.from_netcdf(posterior_dir / "posterior.nc")

    result = compute_transition_state_drc(
        inference_data=idata,
        model_name=args.model,
        point_inputs=point_inputs,
        model_points=model_data.model_points,
        config=config,
        step_eV=args.step_eV,
    )

    result.draws.to_netcdf(derived_dir / "drc_transition_state_draws.nc")
    result.summary.to_parquet(derived_dir / "drc_transition_state.parquet", index=False)
    result.checks.to_csv(tables_dir / "drc_transition_state_checks.csv", index=False)

    plot_transition_state_drc(
        summary=result.summary,
        model_name=args.model,
        output_path=figures_dir / "drc_transition_states.png",
    )

    print(f"\n{MATERIAL}: {args.model}")
    print(f"Likelihood: {args.likelihood}")
    print("\n=== TRANSITION-STATE DRC CHECKS ===")
    print(result.checks.to_string(index=False))

    if args.check_half_step:
        half_step = compute_transition_state_drc(
            inference_data=idata,
            model_name=args.model,
            point_inputs=point_inputs,
            model_points=model_data.model_points,
            config=config,
            step_eV=0.5 * args.step_eV,
        )
        convergence = compare_transition_state_drc_steps(result, half_step)
        convergence.to_csv(tables_dir / "drc_transition_state_step_convergence.csv", index=False)

        print("\n=== FINITE-DIFFERENCE STEP CONVERGENCE ===")
        print(convergence.to_string(index=False))

    print(f"\nSaved transition-state DRC products to: {output_dir}")


if __name__ == "__main__":
    main()
