from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import yaml

from mkm.inference.posterior_diagnostics import (
    summarize_pointwise_pathway_fractions,
    summarize_posterior_linear_observable,
)
from mkm.model_data import build_model_data
from mkm.observable_maps import (
    build_adjacent_log_order_map,
    build_alpha_map,
    build_log_slope_order_map,
)
from mkm.models.agpd_basic import available_agpd_models


ROOT = Path(__file__).resolve().parents[1]

SELECTED_PATH = ROOT / "data" / "processed" / "AgPd_COOx_basic" / "analysis" / "AgPd_COOx_basic_selected.parquet"
MODEL_CONFIG_PATH = ROOT / "config" / "models" / "agpd_basic.yaml"
PREPROCESSING_CONFIG_PATH = ROOT / "config" / "preprocessing" / "agpd_basic.yaml"

POSTERIOR_ROOT = ROOT / "results" / "AgPd_COOx_basic" / "posterior"

MATERIAL = "Ag10Pd90"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", choices=available_agpd_models())
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    with open(MODEL_CONFIG_PATH, "r") as file:
        model_config = yaml.safe_load(file)

    with open(PREPROCESSING_CONFIG_PATH, "r") as file:
        preprocessing_config = yaml.safe_load(file)

    selected = pd.read_parquet(SELECTED_PATH)
    selected = selected.loc[selected["material"] == MATERIAL].copy()

    model_data = build_model_data(
        selected_replicates=selected,
        electrolyte_concentration_column="C_KOH_M",
    )

    model_points = model_data.model_points

    posterior_dir = POSTERIOR_ROOT / MATERIAL / model_name
    idata = az.from_netcdf(posterior_dir / "posterior.nc")

    alpha_map = build_alpha_map(
        model_points=model_points,
        temperature_K=model_config["temperature_K"],
    )

    delta_oh_map = build_log_slope_order_map(
        model_points=model_points,
        varying_column="electrolyte_concentration_M",
        varying_values=preprocessing_config["KOH_concentrations_M"],
        group_columns=["material", "CO_mole_fraction"],
    )

    delta_co_map = build_adjacent_log_order_map(
        model_points=model_points,
        varying_column="CO_mole_fraction",
        varying_values=preprocessing_config["CO_mole_fractions"],
        group_columns=["material", "electrolyte_concentration_M"],
        lower_value_column="lower_CO_mole_fraction",
        upper_value_column="upper_CO_mole_fraction",
    )

    output_dir = posterior_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    observable_maps = {
        "alpha": alpha_map,
        "delta_OH": delta_oh_map,
        "delta_CO": delta_co_map,
    }

    print(f"\n{MATERIAL}: {model_name}")

    for name, observable_map in observable_maps.items():
        summary = summarize_posterior_linear_observable(idata, observable_map)

        summary.pooled.to_parquet(output_dir / f"{name}.parquet", index=False)
        summary.by_chain.to_parquet(output_dir / f"{name}_by_chain.parquet", index=False)

        print(f"\n{name}:")
        print(f"  posterior median range: {summary.pooled['q50'].min():.3f} to {summary.pooled['q50'].max():.3f}")
        print(
            f"  posterior 95% envelope: "
            f"{summary.pooled['q025'].min():.3f} to {summary.pooled['q975'].max():.3f}"
        )

    pathway_summaries = summarize_pointwise_pathway_fractions(
        inference_data=idata,
        model_points=model_points,
    )

    for name, summary in pathway_summaries.items():
        summary.to_parquet(output_dir / f"{name}.parquet", index=False)

        print(f"\n{name}:")
        print(
            f"  posterior median pointwise range: "
            f"{summary[f'{name}_q50'].min():.4f} to "
            f"{summary[f'{name}_q50'].max():.4f}"
        )

    posterior = idata.posterior

    print("\nCoverage ranges:")

    for name in ["theta_CO", "theta_OH_Pd", "theta_empty_Pd", "theta_OH_Ag", "theta_empty_Ag"]:
        if name not in posterior:
            continue

        values = np.asarray(posterior[name], dtype=float)

        print(
            f"  {name}: "
            f"median {np.quantile(values, 0.50):.4g}; "
            f"95% {np.quantile(values, 0.025):.4g} to "
            f"{np.quantile(values, 0.975):.4g}"
        )


if __name__ == "__main__":
    main()