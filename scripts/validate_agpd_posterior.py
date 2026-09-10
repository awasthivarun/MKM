"""Run one LOCO or LOMO validation fit for an all-material AgPd model."""

from argparse import ArgumentParser

import numpy as np

from mkm.model_data import build_model_data
from mkm.models.agpd_basic import available_agpd_all_material_models
from mkm.project_paths import ProjectPaths
from mkm.postprocessing.plotting import plot_observation_grid
from mkm.postprocessing.validation import (
    plot_heldout_pit_conditions,
    plot_validation_parameter_posteriors,
    summarize_validation_posterior_shift,
)
from mkm.provenance import build_fit_metadata
from mkm.workflows.agpd_basic import (
    available_agpd_materials,
    build_agpd_inputs,
    load_agpd_model_config,
    load_agpd_selected_materials,
)
from mkm.workflows.agpd_fit import (
    all_parameter_specs,
    build_agpd_fit_model,
    build_fit_mechanism,
    resolved_parameterization_metadata,
    resolve_agpd_fit_specification,
)
from mkm.workflows.agpd_posterior import (
    load_agpd_posterior_run,
    validate_agpd_run_metadata,
)
from mkm.workflows.agpd_validation import (
    build_mechanism_prediction_model,
    compute_heldout_rate_predictions,
    split_agpd_loco,
    split_agpd_lomo,
    validate_heldout_error_support,
    validate_validation_error_structure,
)
from mkm.workflows.posterior_lifecycle import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
    run_posterior_lifecycle,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_all_material_models())
    parser.add_argument("scheme", choices=("loco", "lomo"))
    parser.add_argument("--holdout-material", required=True)
    parser.add_argument("--koh-M", type=float)
    parser.add_argument("--co-mole-fraction", type=float)
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument(
        "--error-structure",
        choices=("shared", "material"),
        default="shared",
    )
    parser.add_argument("--prior-material", default="Ag10Pd90")
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
    return parser.parse_args()


def _sampler_settings(args):
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


def _validate_scheme_arguments(args):
    validate_validation_error_structure(args.scheme, args.error_structure)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    if args.scheme == "loco":
        if args.koh_M is None or args.co_mole_fraction is None:
            raise ValueError("LOCO requires --koh-M and --co-mole-fraction.")
    elif args.koh_M is not None or args.co_mole_fraction is not None:
        raise ValueError("LOMO does not use --koh-M or --co-mole-fraction.")


def _split_selected(selected, args):
    if args.scheme == "lomo":
        return split_agpd_lomo(selected, args.holdout_material)
    return split_agpd_loco(
        selected,
        args.holdout_material,
        args.koh_M,
        args.co_mole_fraction,
    )


def _build_tables(selected):
    return build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )


def _composition_extrapolation(train_inputs, heldout_inputs, config):
    composition = config["surface_composition"]
    train_x = np.asarray(
        [composition[material]["Ag_fraction"] for material in train_inputs.materials],
        dtype=float,
    )
    heldout_x = np.asarray(
        [composition[material]["Ag_fraction"] for material in heldout_inputs.materials],
        dtype=float,
    )
    return bool(np.any(heldout_x < train_x.min()) or np.any(heldout_x > train_x.max()))


def main():
    args = parse_args()
    _validate_scheme_arguments(args)

    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    specification = resolve_agpd_fit_specification(
        config,
        model_name=args.model,
        all_materials=True,
        parameterization=args.parameterization,
        error_structure=args.error_structure,
        prior_material=args.prior_material,
    )

    materials = available_agpd_materials(config)
    if args.holdout_material not in materials:
        raise ValueError(
            f"Unknown holdout material '{args.holdout_material}'. "
            f"Configured materials: {materials}."
        )

    source_run = load_agpd_posterior_run(
        paths,
        config,
        specification,
        reconstruct_pointwise=False,
        progressbar=False,
    )

    selected = load_agpd_selected_materials(paths, materials)
    train_selected, heldout_selected = _split_selected(selected, args)
    train_data = _build_tables(train_selected)
    heldout_data = _build_tables(heldout_selected)

    train_fit = build_agpd_fit_model(specification, train_data, config)
    heldout_inputs = build_agpd_inputs(heldout_data)
    validate_heldout_error_support(
        specification.error_structure,
        train_fit.inputs.materials,
        heldout_inputs.materials,
    )
    heldout_mechanism = build_fit_mechanism(
        specification,
        heldout_inputs,
        config,
        prediction_only=True,
    )
    prediction_model = build_mechanism_prediction_model(
        heldout_inputs,
        heldout_mechanism,
    )

    output_dir = paths.agpd_validation_output_dir(
        scheme=args.scheme,
        model_name=specification.model_name,
        parameterization=specification.parameterization,
        error_structure=specification.error_structure,
        material=args.holdout_material,
        koh_M=args.koh_M,
        co_mole_fraction=args.co_mole_fraction,
    )
    sampler = _sampler_settings(args)
    composition_extrapolation = _composition_extrapolation(
        train_fit.inputs,
        heldout_inputs,
        config,
    )

    def metadata_factory():
        metadata = build_fit_metadata(
            root=paths.root,
            fit_scope="all_materials",
            materials=train_fit.inputs.materials,
            model_name=specification.model_name,
            error_structure=specification.error_structure,
            data_path=paths.agpd_selected_path,
            model_config_path=paths.agpd_model_config_path,
            sampler=sampler,
            parameterization=specification.parameterization,
            parameterization_specification=resolved_parameterization_metadata(
                specification,
                config,
            ),
            prior_material=specification.prior_material,
        )
        metadata.update(
            {
                "validation_scheme": args.scheme,
                "holdout_material": args.holdout_material,
                "holdout_koh_M": args.koh_M,
                "holdout_co_mole_fraction": args.co_mole_fraction,
                "n_training_observations": len(train_data.observations),
                "n_heldout_observations": len(heldout_data.observations),
                "composition_extrapolation": composition_extrapolation,
            }
        )
        return metadata

    def checkpoint_validator(metadata):
        validate_agpd_run_metadata(
            metadata,
            paths,
            specification,
            train_fit.inputs.materials,
            config,
            allowed_statuses=(RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        )
        expected = {
            "validation_scheme": args.scheme,
            "holdout_material": args.holdout_material,
            "holdout_koh_M": args.koh_M,
            "holdout_co_mole_fraction": args.co_mole_fraction,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            details = "; ".join(
                f"{key}: stored={stored!r}, expected={expected_value!r}"
                for key, (stored, expected_value) in mismatches.items()
            )
            raise ValueError(
                f"Validation checkpoint does not match this request: {details}"
            )

    lifecycle = run_posterior_lifecycle(
        built=train_fit.built_model,
        output_dir=output_dir,
        parameter_specs=all_parameter_specs(specification, config),
        sampler=sampler,
        resume=args.resume,
        overwrite=args.overwrite,
        metadata_factory=metadata_factory,
        checkpoint_validator=checkpoint_validator,
        progressbar=True,
    )

    heldout = compute_heldout_rate_predictions(
        lifecycle.inference_data,
        prediction_model,
        heldout_data,
        heldout_inputs,
        error_structure=specification.error_structure,
        random_seed=args.random_seed,
        backend=args.backend,
        progressbar=True,
    )
    heldout.pointwise.to_parquet(
        output_dir / "heldout_predictions.parquet",
        index=False,
    )
    heldout.summary.to_csv(output_dir / "validation_summary.csv", index=False)

    parameter_specs = all_parameter_specs(specification, config)
    posterior_shift = summarize_validation_posterior_shift(
        source_run.inference_data,
        lifecycle.inference_data,
        parameter_specs,
    )
    posterior_shift.to_csv(output_dir / "posterior_shift.csv", index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    context_label = (
        f"{args.scheme.upper()} {args.holdout_material}"
        if args.scheme == "lomo"
        else (
            f"LOCO {args.holdout_material}, "
            f"{args.koh_M:g} M KOH, {100 * args.co_mole_fraction:g}% CO"
        )
    )
    plot_validation_parameter_posteriors(
        source_run.inference_data,
        lifecycle.inference_data,
        parameter_specs,
        figures_dir / "posterior_vs_full.png",
        context_label=context_label,
    )

    plot_frame = heldout.pointwise.copy()
    plot_frame["residual"] = plot_frame["model_residual"]
    plot_observation_grid(
        plot_frame,
        figures_dir / "heldout_rates_model.png",
        distribution="model",
        y_scale="log",
        context_label=context_label,
    )
    plot_observation_grid(
        plot_frame,
        figures_dir / "heldout_rates_predictive.png",
        distribution="predictive",
        y_scale="linear",
        context_label=context_label,
    )
    plot_observation_grid(
        plot_frame,
        figures_dir / "heldout_residuals.png",
        residual=True,
        y_scale="linear",
        context_label=context_label,
    )
    plot_heldout_pit_conditions(
        plot_frame,
        figures_dir / "heldout_pit_conditions.png",
        context_label=context_label,
    )

    if lifecycle.sampling_seconds is not None:
        print(f"Sampling wall time: {lifecycle.sampling_seconds:.2f} s")
    print(heldout.summary.to_string(index=False))
    print(f"Composition extrapolation: {composition_extrapolation}")
    print(f"Validation products saved to: {output_dir}")


if __name__ == "__main__":
    main()
