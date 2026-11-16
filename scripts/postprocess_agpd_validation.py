"""Postprocess one AgPd LOCO/LOMO training refit using the regular workflow."""

from argparse import ArgumentParser
from dataclasses import replace

from mkm.model_data import build_model_data
from mkm.inference.posterior import (
    add_log_likelihood, compute_posterior_deterministics, load_inference_data,
    validate_inference_data_alignment,
)
from mkm.project_paths import ProjectPaths
from mkm.provenance import model_data_observation_sha256, read_run_metadata, sha256_file
from mkm.workflows.agpd_basic import (
    available_agpd_materials, load_agpd_model_config, load_agpd_selected_materials,
)
from mkm.workflows.agpd_fit import (
    all_parameter_specs, build_agpd_fit_model, resolve_agpd_fit_specification,
)
from mkm.workflows.agpd_posterior import AgPdPosteriorRun, validate_agpd_run_metadata
from mkm.workflows.agpd_postprocess import (
    postprocess_agpd_composition, postprocess_agpd_drc, postprocess_agpd_run,
)
from mkm.workflows.agpd_validation import split_agpd_loco, split_agpd_lomo, validate_validation_error_structure
from mkm.workflows.posterior_lifecycle import RUN_STATUS_COMPLETE


def parse_args():
    parser = ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("model")
    parser.add_argument("scheme", choices=("loco", "lomo"))
    parser.add_argument("--holdout-material", required=True)
    parser.add_argument("--koh-M", type=float)
    parser.add_argument("--co-mole-fraction", type=float)
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument("--error-structure", choices=("shared", "material"), default="shared")
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--second-order-step-v", type=float, default=0.05)
    parser.add_argument("--drc-step-eV", type=float, default=1e-4)
    parser.add_argument("--plot-level", choices=("none", "core", "full"), default="full")
    return parser.parse_known_args()[0]


def _validation_output_dir(paths, args):
    return paths.agpd_validation_output_dir(
        scheme=args.scheme, model_name=args.model,
        parameterization=args.parameterization, error_structure=args.error_structure,
        material=args.holdout_material,
        koh_M=args.koh_M if args.scheme == "loco" else None,
        co_mole_fraction=args.co_mole_fraction if args.scheme == "loco" else None,
    )


def _load_validation_run(paths, config, specification, args):
    validate_validation_error_structure(args.scheme, specification.error_structure)
    materials = available_agpd_materials(config)
    if args.holdout_material not in materials:
        raise ValueError(f"Unknown held-out material: {args.holdout_material}")
    selected = load_agpd_selected_materials(paths, materials)
    if args.scheme == "lomo":
        if args.koh_M is not None or args.co_mole_fraction is not None:
            raise ValueError("LOMO does not accept a CO/electrolyte condition.")
        train, heldout = split_agpd_lomo(selected, args.holdout_material)
    else:
        if args.koh_M is None or args.co_mole_fraction is None:
            raise ValueError("LOCO requires --koh-M and --co-mole-fraction.")
        train, heldout = split_agpd_loco(selected, args.holdout_material, args.koh_M, args.co_mole_fraction)

    model_data = build_model_data(train, electrolyte_concentration_column="C_KOH_M")
    heldout_data = build_model_data(heldout, electrolyte_concentration_column="C_KOH_M")
    fit = build_agpd_fit_model(specification, model_data, config)
    output_dir = _validation_output_dir(paths, args)
    metadata = read_run_metadata(output_dir / "run_metadata.yaml")
    validate_agpd_run_metadata(
        metadata, paths, specification, fit.inputs.materials, config,
        allowed_statuses=(RUN_STATUS_COMPLETE,),
    )
    expected = {
        "validation_scheme": args.scheme,
        "holdout_material": args.holdout_material,
        "holdout_koh_M": args.koh_M,
        "holdout_co_mole_fraction": args.co_mole_fraction,
        "n_training_observations": len(model_data.observations),
        "n_heldout_observations": len(heldout_data.observations),
        "training_observations_sha256": model_data_observation_sha256(model_data),
        "heldout_observations_sha256": model_data_observation_sha256(heldout_data),
        "validation_source_sha256": sha256_file(paths.root / "src/mkm/workflows/agpd_validation.py"),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError("AgPd validation run identity mismatch: " + ", ".join(mismatches))

    posterior = load_inference_data(output_dir / "posterior.nc")
    backend = metadata["sampler"]["backend"]
    missing = [var.name for var in fit.built_model.model.free_RVs if var.name not in posterior.posterior]
    if missing:
        raise ValueError(f"Validation posterior missing fitted variables: {missing}")
    deterministic_names = ["ln_rate_model", *fit.built_model.mechanism_result.pointwise]
    missing = [name for name in dict.fromkeys(deterministic_names) if name not in posterior.posterior]
    if missing:
        rebuilt = compute_posterior_deterministics(
            posterior, fit.built_model, var_names=missing, backend=backend, progressbar=True,
        )
        posterior = posterior.copy()
        posterior.posterior = rebuilt.to_dataset() if hasattr(rebuilt, "to_dataset") else rebuilt
    if not hasattr(posterior, "log_likelihood") or "rate_observed" not in posterior.log_likelihood:
        posterior = add_log_likelihood(posterior, fit.built_model, backend=backend, progressbar=True)
    validate_inference_data_alignment(posterior)

    return AgPdPosteriorRun(
        specification=specification, metadata=metadata, output_dir=output_dir,
        model_data=model_data, inputs=fit.inputs, built_model=fit.built_model,
        inference_data=posterior, parameter_specs=all_parameter_specs(specification, config),
        free_parameter_names=tuple(rv.name for rv in fit.built_model.model.free_RVs),
    )


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    specification = resolve_agpd_fit_specification(
        config, model_name=args.model, all_materials=True,
        parameterization=args.parameterization, error_structure=args.error_structure,
        prior_material=args.prior_material,
    )
    run = _load_validation_run(paths, config, specification, args)
    products = replace(run, output_dir=run.output_dir / "postprocessing")
    # LOCO has a missing concentration/CO cell, so full-grid observable maps
    # cannot be evaluated without changing the established numerical method.
    skip_observables = args.scheme == "loco"
    overall = postprocess_agpd_run(
        products, config, paths, random_seed=args.random_seed,
        skip_observables=skip_observables,
        second_order_step_v=args.second_order_step_v, plot_level=args.plot_level,
    )
    drc = postprocess_agpd_drc(
        products, config, paths, step_eV=args.drc_step_eV,
        check_half_step=True, make_plots=args.plot_level != "none",
    )
    if products.specification.is_all_materials and products.specification.parameterization != "shared":
        postprocess_agpd_composition(products, config, paths)
    print(drc.checks.to_string(index=False))
    print(f"Validation refit postprocessing: {products.output_dir}")
    print(f"Postprocessing status: {overall}")
    from mkm.workflows.postprocessing_common import postprocessing_execution_failed
    import pandas as pd
    status = pd.read_csv(paths.fit_tables_dir(products.output_dir) / "postprocessing_status.csv")
    if postprocessing_execution_failed(status.to_dict(orient="records")):
        raise RuntimeError("AgPd validation postprocessing encountered an execution error; inspect status CSV.")


if __name__ == "__main__":
    main()
