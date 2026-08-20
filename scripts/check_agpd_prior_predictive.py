from pathlib import Path

import pandas as pd
import yaml

from mkm.inference.model import (
    build_pymc_model,
)
from mkm.inference.prior_predictive import (
    sample_prior_predictive,
    summarize_prior_predictive,
)
from mkm.model_data import (
    build_model_data,
)
from mkm.model_inputs import (
    build_model_input_arrays,
)
from mkm.models.agpd_basic import (
    build_agpd_mechanism,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AgPd_COOx_basic"
    / "analysis"
    / "AgPd_COOx_basic_selected.parquet"
)

CONFIG_PATH = (
    ROOT
    / "config"
    / "models"
    / "agpd_basic.yaml"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "AgPd_COOx_basic"
    / "prior_predictive"
)

MATERIAL = "Ag10Pd90"

MODELS = [
    "BF",
    "BF_LH",
    "CO_BF_ER_LH",
]

DRAWS = 2000
RANDOM_SEED = 20260820


def main():
    with open(
        CONFIG_PATH,
        "r",
    ) as file:
        config = yaml.safe_load(
            file
        )

    selected = pd.read_parquet(
        DATA_PATH
    )

    selected = selected.loc[
        selected["material"]
        == MATERIAL
    ].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column=(
            "C_KOH_M"
        ),
    )

    inputs = build_model_input_arrays(
        model_data
    )

    for model_name in MODELS:
        print(
            f"\n{MATERIAL}: {model_name}"
        )

        mechanism = (
            build_agpd_mechanism(
                model_name=model_name,
                material=MATERIAL,
                config=config,
            )
        )

        built = build_pymc_model(
            inputs=inputs,
            mechanism=mechanism,
            sigma_prior_median=(
                config[
                    "likelihood"
                ][
                    "sigma_prior_median"
                ]
            ),
            sigma_prior_log_sd=(
                config[
                    "likelihood"
                ][
                    "sigma_prior_log_sd"
                ]
            ),
        )

        prior = sample_prior_predictive(
            built_model=built,
            draws=DRAWS,
            random_seed=RANDOM_SEED,
        )

        summary = (
            summarize_prior_predictive(
                prior_predictive=prior,
                model_data=model_data,
            )
        )

        output_dir = (
            OUTPUT_ROOT
            / MATERIAL
            / model_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            summary
            .parameter_summary
            .to_csv(
                output_dir
                / "parameters.csv",
                index=False,
            )
        )

        (
            summary
            .model_point_summary
            .to_parquet(
                output_dir
                / "model_points.parquet",
                index=False,
            )
        )

        (
            summary
            .observation_summary
            .to_parquet(
                output_dir
                / "observations.parquet",
                index=False,
            )
        )

        print(
            summary.parameter_summary
            .to_string(
                index=False
            )
        )

        print_model_point_diagnostics(
            summary.model_point_summary
        )

        prior.to_netcdf(
            output_dir
            / "prior_predictive.nc",
            engine="h5netcdf",
        )

def print_model_point_diagnostics(
    model_point_summary,
):
    print(
        "\nLatent ln(rate) prior envelope:"
    )

    print(
        "  median:"
        f" {model_point_summary['ln_rate_model_q50'].min():.3f}"
        " to"
        f" {model_point_summary['ln_rate_model_q50'].max():.3f}"
    )

    print(
        "  95% lower-bound range:"
        f" {model_point_summary['ln_rate_model_q025'].min():.3f}"
        " to"
        f" {model_point_summary['ln_rate_model_q025'].max():.3f}"
    )

    print(
        "  95% upper-bound range:"
        f" {model_point_summary['ln_rate_model_q975'].min():.3f}"
        " to"
        f" {model_point_summary['ln_rate_model_q975'].max():.3f}"
    )

    coverage_names = [
        "theta_CO",
        "theta_OH_Pd",
        "theta_empty_Pd",
        "theta_OH_Ag",
        "theta_empty_Ag",
    ]

    available_coverages = [
        name
        for name in coverage_names
        if f"{name}_q50"
        in model_point_summary.columns
    ]

    if available_coverages:
        print(
            "\nCoverage prior ranges:"
        )

        for name in available_coverages:
            q025 = model_point_summary[
                f"{name}_q025"
            ]

            q50 = model_point_summary[
                f"{name}_q50"
            ]

            q975 = model_point_summary[
                f"{name}_q975"
            ]

            print(
                f"  {name}:"
                f" median {q50.min():.3g}"
                f" to {q50.max():.3g};"
                f" overall 95% envelope"
                f" {q025.min():.3g}"
                f" to {q975.max():.3g}"
            )

    fraction_names = [
        "rate_fraction_BF",
        "rate_fraction_ER",
        "rate_fraction_LH",
    ]

    available_fractions = [
        name
        for name in fraction_names
        if f"{name}_q50"
        in model_point_summary.columns
    ]

    if available_fractions:
        print(
            "\nPathway-fraction prior ranges:"
        )

        for name in available_fractions:
            q025 = model_point_summary[
                f"{name}_q025"
            ]

            q50 = model_point_summary[
                f"{name}_q50"
            ]

            q975 = model_point_summary[
                f"{name}_q975"
            ]

            print(
                f"  {name}:"
                f" median {q50.min():.3g}"
                f" to {q50.max():.3g};"
                f" overall 95% envelope"
                f" {q025.min():.3g}"
                f" to {q975.max():.3g}"
            )


if __name__ == "__main__":
    main()