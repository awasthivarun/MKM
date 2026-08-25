"""Fit and evaluate one LOCO or LOMO fold for an AgPd composition model."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from mkm.inference.model import build_pymc_model
from mkm.inference.posterior import sample_posterior
from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_coords, build_model_point_inputs
from mkm.models.agpd_basic import (
    available_agpd_composition_models,
    available_agpd_composition_parameterizations,
    build_agpd_composition_mechanism,
)
from mkm.postprocessing.diagnostics import flatten_posterior_samples, summarize_samples
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_basic import build_agpd_inputs, load_agpd_model_config
from mkm.workflows.agpd_validation import split_agpd_loco, split_agpd_lomo


DEFAULT_PRIOR_MATERIAL = "Ag10Pd90"
RANDOM_SEED = 20260825


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("scheme", choices=["loco", "lomo"])
    parser.add_argument("model", choices=available_agpd_composition_models())
    parser.add_argument("--composition-model", choices=available_agpd_composition_parameterizations(), required=True)
    parser.add_argument("--likelihood", choices=["iid"], required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--koh", type=float)
    parser.add_argument("--co", type=float)
    parser.add_argument("--prior-material", default=DEFAULT_PRIOR_MATERIAL)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--resume-free", action="store_true")
    return parser.parse_args()


def _build_mechanism_prediction_model(inputs, mechanism):
    coords = build_model_coords(inputs)
    point_inputs = build_model_point_inputs(inputs)
    n_model_points = len(point_inputs.E_V_SHE)

    with pm.Model(coords=coords) as model:
        result = mechanism(point_inputs)
        ln_rate = pt.as_tensor_variable(result.ln_rate)
        ln_rate = pt.specify_shape(ln_rate, (n_model_points,))
        pm.Deterministic("ln_rate_model", ln_rate, dims="model_point")
    return model


def _posterior_dataset(idata):
    posterior = idata.posterior
    if isinstance(posterior, xr.Dataset):
        return posterior
    return posterior.to_dataset()


def _compute_heldout_predictions(idata, heldout_model, heldout_data):
    free_names = [rv.name for rv in heldout_model.free_RVs]
    posterior = _posterior_dataset(idata)[free_names]

    with heldout_model:
        predicted = pm.compute_deterministics(
            posterior,
            var_names=["ln_rate_model"],
            model=heldout_model,
            extend_dataset=False,
            progressbar=True,
            backend="numba",
        )

    draws = flatten_posterior_samples(predicted["ln_rate_model"])
    summary = summarize_samples(draws)
    pointwise = pd.DataFrame(
        {
            "model_point_id": np.arange(draws.shape[1], dtype=int),
            "ln_rate_model_mean": summary["mean"],
            "ln_rate_model_sd": summary["sd"],
            "ln_rate_model_median": summary["median"],
            "ln_rate_model_hdi95_lower": summary["hdi95_lower"],
            "ln_rate_model_hdi95_upper": summary["hdi95_upper"],
        }
    )

    observations = heldout_data.observations.merge(pointwise, on="model_point_id", how="left", validate="many_to_one")
    observations["mechanism_residual"] = observations["ln_rate"] - observations["ln_rate_model_median"]
    observations["mechanism_hdi_contains_observation"] = (
        (observations["ln_rate"] >= observations["ln_rate_model_hdi95_lower"])
        & (observations["ln_rate"] <= observations["ln_rate_model_hdi95_upper"])
    )

    residual = observations["mechanism_residual"].to_numpy(dtype=float)
    validation_summary = pd.DataFrame(
        [
            {
                "n_observations": len(observations),
                "n_model_points": len(pointwise),
                "mechanism_residual_mean": float(np.mean(residual)),
                "mechanism_residual_rms": float(np.sqrt(np.mean(residual**2))),
                "median_abs_mechanism_residual": float(np.median(np.abs(residual))),
                "mechanism_95_hdi_observation_coverage": float(
                    observations["mechanism_hdi_contains_observation"].mean()
                ),
            }
        ]
    )
    return observations, validation_summary


def _output_dir(paths, args):
    base = (
        paths.root
        / "results"
        / "AgPd_COOx_basic"
        / "validation"
        / "composition"
        / args.composition_model
        / args.likelihood
        / args.model
        / args.scheme
    )
    if args.scheme == "lomo":
        return base / args.material
    return base / args.material / f"KOH_{args.koh:g}_CO_{args.co:g}"


def main():
    args = parse_args()
    if args.scheme == "loco" and (args.koh is None or args.co is None):
        raise ValueError("LOCO requires both --koh and --co.")
    if args.scheme == "lomo" and (args.koh is not None or args.co is not None):
        raise ValueError("LOMO does not use --koh or --co.")

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    selected = pd.read_parquet(paths.agpd_selected_path)

    if args.scheme == "loco":
        train_selected, heldout_selected = split_agpd_loco(
            selected, args.material, args.koh, args.co
        )
    else:
        train_selected, heldout_selected = split_agpd_lomo(selected, args.material)

    train_data = build_model_data(train_selected, electrolyte_concentration_column="C_KOH_M")
    heldout_data = build_model_data(heldout_selected, electrolyte_concentration_column="C_KOH_M")
    train_inputs = build_agpd_inputs(train_data, config, args.likelihood)
    heldout_inputs = build_agpd_inputs(heldout_data, config, args.likelihood)

    train_mechanism = build_agpd_composition_mechanism(
        model_name=args.model,
        materials=tuple(train_inputs.materials),
        config=config,
        prior_material=args.prior_material,
        composition_model=args.composition_model,
    )
    likelihood_config = config["likelihood"]
    built = build_pymc_model(
        inputs=train_inputs,
        mechanism=train_mechanism,
        sigma_prior_median=likelihood_config["sigma_prior_median"],
        sigma_prior_log_sd=likelihood_config["sigma_prior_log_sd"],
        setup_intercept=False,
    )

    output_dir = _output_dir(paths, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    heldout_selected.to_parquet(output_dir / "heldout_selected.parquet", index=False)

    print(f"\nValidation scheme: {args.scheme.upper()}")
    print(f"Model: {args.model}")
    print(f"Composition parameterization: {args.composition_model}")
    print(f"Held-out material: {args.material}")
    if args.scheme == "loco":
        print(f"Held-out condition: {args.koh:g} M KOH, CO mole fraction {args.co:g}")
    print(f"Training materials: {', '.join(train_inputs.materials)}")
    print(f"Held-out observations: {len(heldout_data.observations)}")

    posterior_free_path = output_dir / "posterior_free.nc"
    if args.resume_free:
        if not posterior_free_path.exists():
            raise FileNotFoundError(f"Cannot resume; posterior not found: {posterior_free_path}")
        print(f"Reusing sampled posterior: {posterior_free_path}")
        idata = az.from_netcdf(posterior_free_path)
        elapsed = None
    else:
        start = perf_counter()
        idata = sample_posterior(
            built,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=RANDOM_SEED,
            nuts_sampler="nutpie",
            backend="numba",
            compute_convergence_checks=False,
        )
        elapsed = perf_counter() - start
        idata.to_netcdf(posterior_free_path, engine="h5netcdf")

    free_names = [rv.name for rv in built.model.free_RVs]
    diagnostics = az.summary(idata, var_names=free_names, kind="diagnostics", round_to=None)
    diagnostics.to_csv(output_dir / "sampler_diagnostics.csv")

    heldout_mechanism = build_agpd_composition_mechanism(
        model_name=args.model,
        materials=tuple(heldout_inputs.materials),
        config=config,
        prior_material=args.prior_material,
        composition_model=args.composition_model,
        prediction_only=True,
    )
    heldout_model = _build_mechanism_prediction_model(heldout_inputs, heldout_mechanism)
    predictions, validation_summary = _compute_heldout_predictions(idata, heldout_model, heldout_data)
    predictions.to_csv(output_dir / "heldout_predictions.csv", index=False)
    validation_summary.to_csv(output_dir / "validation_summary.csv", index=False)

    if elapsed is not None:
        print(f"Sampling wall time: {elapsed:.2f} s")
    print(f"Maximum R-hat: {diagnostics['r_hat'].max():.4f}")
    print(f"Minimum bulk ESS: {diagnostics['ess_bulk'].min():.1f}")
    print(f"Minimum tail ESS: {diagnostics['ess_tail'].min():.1f}")
    print("\nHeld-out mechanism validation:")
    print(validation_summary.to_string(index=False))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
