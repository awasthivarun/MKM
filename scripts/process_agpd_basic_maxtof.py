"""Build the AgPd basic-media dataset truncated from 1e-3 s^-1 through maximum mean TOF."""

import yaml

from mkm.preprocessing.agpd_basic import (
    add_agpd_rates,
    add_agpd_transfer_coefficients,
    build_agpd_analysis_grid,
    calculate_agpd_co_order,
    calculate_agpd_co_order_replicates,
    calculate_agpd_oh_order,
    load_agpd_dataset,
    summarize_agpd_replicates,
)
from mkm.preprocessing.agpd_basic_maxtof import truncate_agpd_analysis_maxtof
from mkm.project_paths import ProjectPaths


def main():
    paths = ProjectPaths.discover(__file__).with_agpd_data_variant("maxtof")
    with open(paths.agpd_preprocessing_config_path, "r") as file:
        config = yaml.safe_load(file)

    standardized = load_agpd_dataset(raw_dir=paths.agpd_raw_dir, config=config)
    analysis_full = build_agpd_analysis_grid(standardized=standardized, config=config)
    analysis_full = add_agpd_rates(analysis=analysis_full, config=config)
    analysis_selected, truncation = truncate_agpd_analysis_maxtof(
        analysis_rates=analysis_full,
        config=config,
    )
    analysis_selected = add_agpd_transfer_coefficients(
        analysis_rates=analysis_selected,
        config=config,
    )
    summary_selected = summarize_agpd_replicates(
        analysis_rates=analysis_selected,
        config=config,
    )
    delta_OH = calculate_agpd_oh_order(summary=summary_selected, config=config)
    delta_CO_replicates = calculate_agpd_co_order_replicates(
        analysis_rates=analysis_selected,
        config=config,
    )
    delta_CO = calculate_agpd_co_order(
        analysis_rates=analysis_selected,
        config=config,
    )

    paths.agpd_standardized_dir.mkdir(parents=True, exist_ok=True)
    paths.agpd_analysis_dir.mkdir(parents=True, exist_ok=True)
    delta_OH.to_parquet(paths.agpd_delta_oh_path, index=False)
    delta_CO_replicates.to_parquet(paths.agpd_delta_co_replicates_path, index=False)
    delta_CO.to_parquet(paths.agpd_delta_co_path, index=False)
    standardized.to_parquet(paths.agpd_standardized_path, index=False)
    analysis_full.to_parquet(paths.agpd_full_path, index=False)
    analysis_selected.to_parquet(paths.agpd_selected_path, index=False)
    summary_selected.to_parquet(paths.agpd_summary_path, index=False)
    truncation.to_parquet(paths.agpd_truncation_path, index=False)

    print(f"Dataset: {paths.agpd_dataset_name}")
    print(f"Standardized data: {len(standardized)} rows")
    print(f"Full analysis-grid data: {len(analysis_full)} rows")
    print(f"Selected analysis-grid data: {len(analysis_selected)} rows")
    print(f"Selected summary data: {len(summary_selected)} rows")
    print(f"Conditions truncated: {(truncation['n_points_retained'] < truncation['n_points_original']).sum()}")
    print(f"OH-order data: {len(delta_OH)} rows")
    print(f"CO-order replicate data: {len(delta_CO_replicates)} rows")
    print(f"CO-order summary data: {len(delta_CO)} rows")
    print(f"Saved standardized data to: {paths.agpd_standardized_path}")
    print(f"Saved full analysis data to: {paths.agpd_full_path}")
    print(f"Saved selected analysis data to: {paths.agpd_selected_path}")
    print(f"Saved selected summary to: {paths.agpd_summary_path}")
    print(f"Saved truncation metadata to: {paths.agpd_truncation_path}")
    print(f"Saved OH-order data to: {paths.agpd_delta_oh_path}")
    print(f"Saved CO-order replicate data to: {paths.agpd_delta_co_replicates_path}")
    print(f"Saved CO-order summary data to: {paths.agpd_delta_co_path}")


if __name__ == "__main__":
    main()
