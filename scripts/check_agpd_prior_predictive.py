"""Run prior-predictive checks for an individual or all-material AgPd model."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from mkm.inference.prior_predictive import (
    sample_prior_predictive,
    summarize_prior_predictive,
)
from mkm.models.agpd_basic import available_agpd_models
from mkm.postprocessing.diagnostics import build_physical_checks
from mkm.project_paths import ProjectPaths
from mkm.provenance import build_fit_metadata, write_run_metadata
from mkm.workflows.agpd_posterior import agpd_model_specification
from mkm.workflows.agpd_basic import build_agpd_model_data, load_agpd_model_config
from mkm.workflows.agpd_fit import (
    build_agpd_fit_model,
    fit_materials,
    resolved_parameterization_metadata, resolved_prior_metadata,
    resolve_agpd_fit_specification,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    parser.add_argument("--material", default="Ag10Pd90")
    parser.add_argument("--all-materials", action="store_true")
    parser.add_argument("--parameterization", default="shared")
    parser.add_argument(
        "--error-structure",
        choices=("shared", "material"),
        default="shared",
    )
    parser.add_argument("--prior-material", default="Ag10Pd90")
    parser.add_argument("--exclude-materials", nargs="*", default=(),
                        help="All-material only: omit these material datasets.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--save-draws", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = ProjectPaths.discover(__file__)
    config = load_agpd_model_config(paths)
    specification = resolve_agpd_fit_specification(
        config,
        model_name=args.model,
        all_materials=args.all_materials,
        material=args.material,
        parameterization=args.parameterization,
        error_structure=args.error_structure,
        prior_material=args.prior_material,
        excluded_materials=args.exclude_materials,
    )

    model_data = build_agpd_model_data(paths, fit_materials(specification, config))
    fit = build_agpd_fit_model(specification, model_data, config)
    prior_predictive = sample_prior_predictive(
        fit.built_model,
        draws=args.draws,
        random_seed=args.random_seed,
    )
    parameter_names = tuple(variable.name for variable in fit.built_model.model.free_RVs)
    summary = summarize_prior_predictive(
        prior_predictive,
        model_data,
        parameter_names=parameter_names,
    )

    output_dir = paths.agpd_prior_predictive_output_dir(
        fit_scope=specification.fit_scope,
        model_name=specification.model_name,
        material=specification.material,
        parameterization=specification.parameterization,
        error_structure=specification.error_structure,
        excluded_materials=specification.excluded_materials,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.parameter_summary.to_csv(
        output_dir / "prior_parameters.csv",
        index=False,
    )
    summary.model_point_summary.to_parquet(
        output_dir / "prior_model_points.parquet",
        index=False,
    )
    summary.observation_summary.to_parquet(
        output_dir / "prior_observations.parquet",
        index=False,
    )
    build_physical_checks(prior_predictive.prior).to_csv(
        output_dir / "prior_physical_checks.csv", index=False,
    )
    ln_rate = np.asarray(prior_predictive.prior["ln_rate_model"], dtype=float)
    pd.DataFrame([{
        "n_draw_values": ln_rate.size,
        "n_nonfinite_ln_rate": int(np.sum(~np.isfinite(ln_rate))),
        "n_ln_rate_above_exp_range": int(np.sum(ln_rate > np.log(np.finfo(float).max))),
    }]).to_csv(output_dir / "prior_numerical_checks.csv", index=False)
    metadata = build_fit_metadata(
        root=paths.root, fit_scope=specification.fit_scope, materials=fit.inputs.materials,
        model_name=specification.model_name, error_structure=specification.error_structure,
        data_path=paths.agpd_selected_path,
        model_config_path=paths.agpd_model_config_path,
        model_config_paths=paths.agpd_model_config_paths,
        sampler={"draws": args.draws, "random_seed": args.random_seed},
        parameterization=specification.parameterization,
        parameterization_specification=resolved_parameterization_metadata(specification, config),
        prior_material=specification.prior_material,
        model_specification=agpd_model_specification(paths, specification, config),
        prior_specification=resolved_prior_metadata(specification, config),
    )
    metadata["analysis"] = "prior_predictive"
    write_run_metadata(metadata, output_dir / "run_metadata.yaml")
    if args.save_draws:
        prior_predictive.to_netcdf(
            output_dir / "prior_predictive.nc",
            engine="h5netcdf",
        )

    print(f"Prior-predictive summaries saved to: {output_dir}")


if __name__ == "__main__":
    main()
