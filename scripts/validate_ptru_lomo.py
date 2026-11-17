"""Refit a pooled PtRu model with one catalyst material completely held out (LOMO)."""

from argparse import ArgumentParser
from dataclasses import replace

import pandas as pd

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.models.ptru import available_ptru_models, build_ptru_mechanism
from mkm.postprocessing.plotting import plot_observation_grid
from mkm.postprocessing.validation import (
    plot_heldout_pit_conditions, plot_validation_parameter_posteriors, summarize_validation_posterior_shift,
)
from mkm.project_paths import ProjectPaths
from mkm.workflows.agpd_validation import build_mechanism_prediction_model, compute_heldout_rate_predictions
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE, RUN_STATUS_SAMPLED, run_posterior_lifecycle
from mkm.workflows.ptru_fit import build_ptru_fit, load_ptru_config, resolve_ptru_fit_specification
from mkm.workflows.ptru_posterior import load_ptru_posterior_run
from mkm.workflows.ptru_validation import (
    ptru_lomo_metadata, ptru_lomo_output_dir, split_ptru_lomo, validate_ptru_lomo_metadata,
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=available_ptru_models())
    parser.add_argument("--medium", choices=("acidic", "basic"), required=True)
    parser.add_argument("--holdout-material", required=True)
    parser.add_argument("--parameterization", choices=("shared", "linear_xRu"), default="shared")
    parser.add_argument("--error-structure", choices=("shared",), default="shared")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--nuts-sampler", default="nutpie")
    parser.add_argument("--backend", default="numba")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare-full-posterior", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    paths = ProjectPaths.discover(__file__)
    config, priors = load_ptru_config(paths)
    if args.holdout_material not in config["materials"]:
        raise ValueError(f"Unknown PtRu holdout material: {args.holdout_material}")
    if args.model == "CO_BF":
        raise ValueError("BF-only PtRu LOMO is impossible with only two Ru-containing materials: "
                         "Pt100 cannot be included in a BF-only training fit or prediction.")
    selected = pd.read_parquet(paths.ptru_selected_path(args.medium))
    if "medium" in selected and not selected["medium"].eq(args.medium).all():
        raise ValueError("Selected PtRu observations contain the wrong medium.")
    training, heldout = split_ptru_lomo(selected, args.holdout_material)
    heldout_data = build_model_data(heldout, electrolyte_concentration_column="C_electrolyte_M")
    heldout_inputs = build_model_input_arrays(heldout_data)

    spec = resolve_ptru_fit_specification(
        config, args.medium, args.model, all_materials=True,
        parameterization=args.parameterization, error_structure="shared",
        excluded_materials=(args.holdout_material,),
    )
    output_dir = ptru_lomo_output_dir(
        paths, args.medium, args.model, args.holdout_material, args.parameterization, args.error_structure,
    )
    fit = build_ptru_fit(paths, spec, selected=training, output_dir=output_dir)
    # Prediction uses training's parameter names but evaluates on the missing material's composition.
    prediction_spec = replace(spec, materials=(args.holdout_material,))
    prediction_mechanism = build_ptru_mechanism(prediction_spec, config, priors)
    prediction_model = build_mechanism_prediction_model(heldout_inputs, prediction_mechanism)
    sampler = {
        "nuts_sampler": args.nuts_sampler, "backend": args.backend,
        "draws": args.draws, "tune": args.tune, "chains": args.chains, "cores": args.cores,
        "target_accept": args.target_accept, "random_seed": args.random_seed,
    }

    def metadata_factory():
        return ptru_lomo_metadata(paths, fit, sampler, args.holdout_material, heldout_data)

    def validator(metadata):
        validate_ptru_lomo_metadata(
            metadata, paths, fit, sampler, args.holdout_material, heldout_data,
            allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        )

    result = run_posterior_lifecycle(
        built=fit.built_model, output_dir=output_dir, parameter_specs=fit.parameter_specs,
        sampler=sampler, resume=args.resume, overwrite=args.overwrite,
        metadata_factory=metadata_factory, checkpoint_validator=validator, progressbar=True,
    )
    predicted = compute_heldout_rate_predictions(
        result.inference_data, prediction_model, heldout_data, heldout_inputs,
        error_structure="shared", random_seed=args.random_seed, backend=args.backend, progressbar=True,
    )
    predicted.pointwise.to_parquet(output_dir / "heldout_predictions.parquet", index=False)
    predicted.summary.to_csv(output_dir / "validation_summary.csv", index=False)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    context = f"{args.holdout_material} LOMO / {args.medium} / {args.model}"
    electrolyte_label = "HClO4" if args.medium == "acidic" else "KOH"
    plot_frame = predicted.pointwise.copy()
    plot_frame["residual"] = plot_frame["model_residual"]
    for filename, kwargs in (
        ("heldout_rates_model.png", {"distribution": "model", "y_scale": "log"}),
        ("heldout_rates_predictive.png", {"distribution": "predictive", "y_scale": "linear"}),
        ("heldout_residuals.png", {"residual": True, "y_scale": "linear"}),
    ):
        plot_observation_grid(plot_frame, figures_dir / filename,
                              context_label=context, electrolyte_label=electrolyte_label, **kwargs)
    plot_heldout_pit_conditions(plot_frame, figures_dir / "heldout_pit_conditions.png",
                                context_label=context, electrolyte_label=electrolyte_label)
    if args.compare_full_posterior:
        full_spec = resolve_ptru_fit_specification(
            config, args.medium, args.model, all_materials=True,
            parameterization=args.parameterization, error_structure="shared",
        )
        full_run = load_ptru_posterior_run(paths, full_spec, progressbar=False)
        shift = summarize_validation_posterior_shift(full_run.inference_data, result.inference_data,
                                                      fit.parameter_specs)
        shift.to_csv(output_dir / "posterior_shift.csv", index=False)
        plot_validation_parameter_posteriors(
            full_run.inference_data, result.inference_data, fit.parameter_specs,
            figures_dir / "posterior_vs_full.png", context_label=context,
        )
    print(f"PtRu LOMO validation saved to: {output_dir}")
    print(predicted.summary.to_string(index=False))
    print("LOMO holds out every observation of one material; its error parameters must be shared.")


if __name__ == "__main__":
    main()
