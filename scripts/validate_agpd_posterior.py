"""Run one LOCO or LOMO validation fit for an all-material AgPd model."""

from argparse import ArgumentParser
from shutil import rmtree
from time import perf_counter

import numpy as np

from mkm.inference.posterior import (
    add_log_likelihood,
    compute_posterior_deterministics,
    load_inference_data,
    sample_posterior,
    write_inference_data,
)
from mkm.model_data import build_model_data
from mkm.models.agpd_basic import available_agpd_all_material_models
from mkm.postprocessing.diagnostics import build_posterior_parameter_summary
from mkm.postprocessing.sampling import build_sampling_health
from mkm.project_paths import ProjectPaths
from mkm.provenance import (
    build_fit_metadata,
    read_run_metadata,
    sha256_file,
    write_run_metadata,
)
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
    resolve_agpd_fit_specification,
)
from mkm.workflows.agpd_posterior import (
    RUN_STATUS_COMPLETE,
    RUN_STATUS_SAMPLED,
)
from mkm.workflows.agpd_validation import (
    build_mechanism_prediction_model,
    compute_heldout_rate_predictions,
    split_agpd_loco,
    split_agpd_lomo,
    validate_heldout_error_support,
    validate_validation_error_structure,
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
    parser.add_argument("--random-seed", type=int, default=20260826)
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


def _validation_metadata_matches(
    metadata,
    paths,
    specification,
    args,
    train_materials,
):
    expected = {
        "status": (RUN_STATUS_SAMPLED, RUN_STATUS_COMPLETE),
        "fit_scope": "all_materials",
        "materials": list(train_materials),
        "model": specification.model_name,
        "likelihood": "rate_normal",
        "error_structure": specification.error_structure,
        "parameterization": specification.parameterization,
        "prior_material": specification.prior_material,
        "validation_scheme": args.scheme,
        "holdout_material": args.holdout_material,
        "holdout_koh_M": args.koh_M,
        "holdout_co_mole_fraction": args.co_mole_fraction,
    }
    mismatches = {}
    for key, value in expected.items():
        stored = metadata.get(key)
        if key == "status":
            if stored not in value:
                mismatches[key] = (stored, value)
        elif stored != value:
            mismatches[key] = (stored, value)

    input_metadata = metadata.get("inputs", {})
    current_hashes = {
        "data_sha256": sha256_file(paths.agpd_selected_path),
        "model_config_sha256": sha256_file(paths.agpd_model_config_path),
    }
    for key, value in current_hashes.items():
        if input_metadata.get(key) != value:
            mismatches[f"inputs.{key}"] = (input_metadata.get(key), value)

    if mismatches:
        details = "; ".join(
            f"{key}: stored={stored!r}, expected={expected_value!r}"
            for key, (stored, expected_value) in mismatches.items()
        )
        raise ValueError(f"Validation checkpoint does not match this request: {details}")


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

    selected = load_agpd_selected_materials(paths, materials)
    train_selected, heldout_selected = _split_selected(selected, args)
    train_data = _build_tables(train_selected)
    heldout_data = _build_tables(heldout_selected)

    train_fit = build_agpd_fit_model(specification, train_data, config)
    train_built = train_fit.built_model
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
    posterior_path = output_dir / "posterior.nc"
    metadata_path = output_dir / "run_metadata.yaml"

    if args.overwrite and output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling_seconds = None
    if args.resume:
        if not posterior_path.exists():
            raise FileNotFoundError(f"Validation checkpoint not found: {posterior_path}")
        metadata = read_run_metadata(metadata_path)
        _validation_metadata_matches(
            metadata,
            paths,
            specification,
            args,
            train_fit.inputs.materials,
        )
        inference_data = load_inference_data(posterior_path)
    else:
        if posterior_path.exists():
            raise FileExistsError(
                f"Validation posterior already exists: {posterior_path}. "
                "Use --resume or --overwrite."
            )
        start = perf_counter()
        inference_data = sample_posterior(
            train_built,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
            nuts_sampler=args.nuts_sampler,
            backend=args.backend,
            compute_convergence_checks=False,
        )
        sampling_seconds = perf_counter() - start
        write_inference_data(inference_data, posterior_path)

        metadata = build_fit_metadata(
            root=paths.root,
            fit_scope="all_materials",
            materials=train_fit.inputs.materials,
            model_name=specification.model_name,
            error_structure=specification.error_structure,
            data_path=paths.agpd_selected_path,
            model_config_path=paths.agpd_model_config_path,
            sampler=_sampler_settings(args),
            parameterization=specification.parameterization,
            prior_material=specification.prior_material,
        )
        metadata.update(
            {
                "status": RUN_STATUS_SAMPLED,
                "validation_scheme": args.scheme,
                "holdout_material": args.holdout_material,
                "holdout_koh_M": args.koh_M,
                "holdout_co_mole_fraction": args.co_mole_fraction,
                "n_training_observations": len(train_data.observations),
                "n_heldout_observations": len(heldout_data.observations),
                "composition_extrapolation": _composition_extrapolation(
                    train_fit.inputs,
                    heldout_inputs,
                    config,
                ),
            }
        )
        write_run_metadata(metadata, metadata_path)

    free_names = tuple(variable.name for variable in train_built.model.free_RVs)
    missing = [name for name in free_names if name not in inference_data.posterior]
    if missing:
        raise ValueError(f"Validation checkpoint is missing fitted variables: {missing}")

    sampling_health = build_sampling_health(
        inference_data,
        parameter_names=free_names,
    )
    metadata["sampling_health"] = sampling_health
    metadata["status"] = RUN_STATUS_SAMPLED
    write_run_metadata(metadata, metadata_path)

    if "ln_rate_model" not in inference_data.posterior:
        posterior_for_check = compute_posterior_deterministics(
            inference_data,
            train_built,
            var_names=["ln_rate_model"],
            backend=args.backend,
            progressbar=True,
        )
    else:
        posterior_for_check = inference_data.posterior

    if not np.all(
        np.isfinite(np.asarray(posterior_for_check["ln_rate_model"], dtype=float))
    ):
        raise RuntimeError(
            'Validation posterior deterministic "ln_rate_model" contains '
            "non-finite values."
        )

    inference_data.posterior = posterior_for_check
    if (
        not hasattr(inference_data, "log_likelihood")
        or "rate_observed" not in inference_data.log_likelihood
    ):
        inference_data = add_log_likelihood(
            inference_data,
            train_built,
            backend=args.backend,
            progressbar=True,
        )

    log_likelihood = np.asarray(
        inference_data.log_likelihood["rate_observed"],
        dtype=float,
    )
    if not np.all(np.isfinite(log_likelihood)):
        raise RuntimeError(
            'Validation posterior log likelihood "rate_observed" contains '
            "non-finite values."
        )

    inference_data.posterior = inference_data.posterior[list(free_names)]
    write_inference_data(inference_data, posterior_path)

    parameter_summary = build_posterior_parameter_summary(
        inference_data,
        all_parameter_specs(specification, config),
    )
    parameter_summary.to_csv(output_dir / "posterior_parameters.csv", index=False)

    heldout = compute_heldout_rate_predictions(
        inference_data,
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

    metadata["status"] = RUN_STATUS_COMPLETE
    metadata["stored_posterior_variables"] = list(inference_data.posterior.data_vars)
    write_run_metadata(metadata, metadata_path)

    if sampling_seconds is not None:
        print(f"Sampling wall time: {sampling_seconds:.2f} s")
    print(heldout.summary.to_string(index=False))
    print(f"Composition extrapolation: {metadata['composition_extrapolation']}")
    print(f"Validation products saved to: {output_dir}")


if __name__ == "__main__":
    main()
