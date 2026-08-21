from pathlib import Path

import yaml

from mkm.preprocessing.agpd_basic import (
    add_agpd_rates,
    add_agpd_transfer_coefficients,
    build_agpd_analysis_grid,
    load_agpd_dataset,
    summarize_agpd_replicates,
    truncate_agpd_analysis,
    calculate_agpd_co_order,
    calculate_agpd_oh_order,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = REPO_ROOT / "config" / "preprocessing" / "agpd_basic.yaml"

RAW_DIR = REPO_ROOT / "data" / "raw" / "AgPd_COOx_basic"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "AgPd_COOx_basic"

STANDARDIZED_DIR = PROCESSED_DIR / "standardized"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"

STANDARDIZED_PATH = STANDARDIZED_DIR / "AgPd_COOx_basic_replicates.parquet"
ANALYSIS_FULL_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_full.parquet"
ANALYSIS_SELECTED_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_selected.parquet"
SUMMARY_SELECTED_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_summary.parquet"
TRUNCATION_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_truncation.parquet"
DELTA_OH_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_delta_OH.parquet"
DELTA_CO_PATH = ANALYSIS_DIR / "AgPd_COOx_basic_delta_CO.parquet"


def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    standardized = load_agpd_dataset(raw_dir=RAW_DIR, config=config)
    analysis_full = build_agpd_analysis_grid(standardized=standardized, config=config)
    analysis_full = add_agpd_rates(analysis=analysis_full, config=config)

    analysis_selected, truncation = truncate_agpd_analysis(analysis_rates=analysis_full, config=config)
    analysis_selected = add_agpd_transfer_coefficients(analysis_rates=analysis_selected, config=config)
    summary_selected = summarize_agpd_replicates(analysis_rates=analysis_selected, config=config)

    delta_OH = calculate_agpd_oh_order(summary=summary_selected, config=config)
    delta_CO = calculate_agpd_co_order(summary=summary_selected, config=config)

    delta_OH.to_parquet(DELTA_OH_PATH, index=False)
    delta_CO.to_parquet(DELTA_CO_PATH, index=False)

    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    standardized.to_parquet(STANDARDIZED_PATH, index=False)
    analysis_full.to_parquet(ANALYSIS_FULL_PATH, index=False)
    analysis_selected.to_parquet(ANALYSIS_SELECTED_PATH, index=False)
    summary_selected.to_parquet(SUMMARY_SELECTED_PATH, index=False)
    truncation.to_parquet(TRUNCATION_PATH, index=False)

    print(f"Standardized data: {len(standardized)} rows")
    print(f"Full analysis-grid data: {len(analysis_full)} rows")
    print(f"Selected analysis-grid data: {len(analysis_selected)} rows")
    print(f"Selected summary data: {len(summary_selected)} rows")
    print(f"Conditions truncated: {(truncation['n_points_retained'] < truncation['n_points_original']).sum()}")
    print(f"OH-order data: {len(delta_OH)} rows")
    print(f"CO-order data: {len(delta_CO)} rows")

    print(f"Saved standardized data to: {STANDARDIZED_PATH}")
    print(f"Saved full analysis data to: {ANALYSIS_FULL_PATH}")
    print(f"Saved selected analysis data to: {ANALYSIS_SELECTED_PATH}")
    print(f"Saved selected summary to: {SUMMARY_SELECTED_PATH}")
    print(f"Saved truncation metadata to: {TRUNCATION_PATH}")
    print(f"Saved OH-order data to: {DELTA_OH_PATH}")
    print(f"Saved CO-order data to: {DELTA_CO_PATH}")


if __name__ == "__main__":
    main()