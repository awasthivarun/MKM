"""Refit Pt100 after holding out a complete CO/electrolyte condition (LOCO)."""

from argparse import ArgumentParser

import pandas as pd

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.ptru_registry import available_ptru_models
from mkm.models.pt100 import build_pt100_mechanism
from mkm.postprocessing.validation import (
    plot_validation_parameter_posteriors,
    summarize_validation_posterior_shift,
)
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_validation import build_mechanism_prediction_model, compute_heldout_rate_predictions
from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
    run_posterior_lifecycle,
)
from mkm.workflows.pt100_fit import resolve_pt100_parameter_specs, load_pt100_config
from mkm.workflows.ptru_posterior import load_pt100_posterior_run
from mkm.workflows.ptru_validation import (
    build_ptru_loco_fit,
    ptru_loco_metadata,
    ptru_validation_output_dir,
    split_ptru_loco,
    validate_ptru_loco_metadata,
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--electrolyte-M", type=float, required=True)
    parser.add_argument("--co-mole-fraction", type=float, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--nuts-sampler", default="nutpie")
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compare-full-posterior", action="store_true",
        help="Also compare parameter posteriors against a compatible full-data Pt100 checkpoint.",
    )
    return parser.parse_args()


def sampler_settings(args):
    return {
        "nuts_sampler": args.nuts_sampler,
        "backend": args.backend,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_seed": args.random_seed,
    }


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    paths = ProjectPaths.discover(__file__)
    selected = pd.read_parquet(paths.ptru_selected_path(args.medium))
    selected = selected.loc[selected["material"] == "Pt100"].copy()
    if "medium" in selected and not selected["medium"].eq(args.medium).all():
        raise ValueError("Selected Pt100 observations contain a different medium.")
    training, heldout = split_ptru_loco(selected, args.electrolyte_M, args.co_mole_fraction)
    heldout_data = build_model_data(heldout, electrolyte_concentration_column="C_electrolyte_M")
    heldout_inputs = build_model_input_arrays(heldout_data)

    output_dir = ptru_validation_output_dir(
        paths, args.medium, args.model, args.electrolyte_M, args.co_mole_fraction,
    )
    fit = build_ptru_loco_fit(paths, args.medium, args.model, training, output_dir)
    config, prior_config = load_pt100_config(paths)
    physical = resolve_pt100_parameter_specs(prior_config, args.medium, args.model)
    heldout_mechanism = build_pt100_mechanism(
        args.model, args.medium, config["temperature_K"], physical,
    )
    prediction_model = build_mechanism_prediction_model(heldout_inputs, heldout_mechanism)
    sampler = sampler_settings(args)

    def metadata_factory():
        return ptru_loco_metadata(
            paths, fit, sampler, args.electrolyte_M, args.co_mole_fraction,
            len(heldout_data.observations),
        )

    def checkpoint_validator(metadata):
        validate_ptru_loco_metadata(
            metadata, paths, fit, args.electrolyte_M, args.co_mole_fraction,
            len(heldout_data.observations),
            allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        )

    result = run_posterior_lifecycle(
        built=fit.built_model,
        output_dir=output_dir,
        parameter_specs=fit.parameter_specs,
        sampler=sampler,
        resume=args.resume,
        overwrite=args.overwrite,
        metadata_factory=metadata_factory,
        checkpoint_validator=checkpoint_validator,
        progressbar=True,
    )

    predicted = compute_heldout_rate_predictions(
        result.inference_data,
        prediction_model,
        heldout_data,
        heldout_inputs,
        error_structure="shared",
        random_seed=args.random_seed,
        backend=args.backend,
        progressbar=True,
    )
    predicted.pointwise.to_parquet(output_dir / "heldout_predictions.parquet", index=False)
    predicted.summary.to_csv(output_dir / "validation_summary.csv", index=False)

    if args.compare_full_posterior:
        # The full-data checkpoint must pass P1 provenance validation; never bypass it.
        full_run = load_pt100_posterior_run(
            paths, args.medium, args.model, progressbar=False,
        )
        shift = summarize_validation_posterior_shift(
            full_run.inference_data, result.inference_data, fit.parameter_specs,
        )
        shift.to_csv(output_dir / "posterior_shift.csv", index=False)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_validation_parameter_posteriors(
            full_run.inference_data,
            result.inference_data,
            fit.parameter_specs,
            figures_dir / "posterior_vs_full.png",
            context_label=f"Pt100 LOCO, {args.medium}",
        )

    print(f"Pt100 LOCO validation saved to: {output_dir}")
    print(f"Training observations: {len(fit.model_data.observations):,}")
    print(f"Held-out observations: {len(heldout_data.observations):,}")
    print(predicted.summary.to_string(index=False))
    print("LOCO holds out an entire concentration/CO condition, including all replicate curves.")


if __name__ == "__main__":
    main()
