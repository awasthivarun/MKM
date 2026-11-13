
"""Process Todd's Pt/PtRu experimental data and apply configured truncation rules."""

import yaml

from mkm.preprocessing.ptru_selection import select_ptru_analysis
from mkm.preprocessing.ptru_todd import build_todd_analysis_grid, load_todd_dataset
from mkm.project_paths import ProjectPaths


def main():
    paths = ProjectPaths.discover(__file__)
    config_path = paths.config_dir / "preprocessing" / "ptru_todd.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium in ("acidic", "basic"):
        dataset_name = f"PtRu_COOx_{medium}"
        output_dir = paths.processed_data_dir / dataset_name
        standardized_dir = output_dir / "standardized"
        analysis_dir = output_dir / "analysis"

        standardized = load_todd_dataset(paths.raw_data_dir, medium, config)
        analysis_full = build_todd_analysis_grid(standardized, config)
        analysis_selected, truncation = select_ptru_analysis(analysis_full, config)

        standardized_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        standardized_path = standardized_dir / f"{dataset_name}_replicates.parquet"
        full_path = analysis_dir / f"{dataset_name}_full.parquet"
        selected_path = analysis_dir / f"{dataset_name}_selected.parquet"
        truncation_path = analysis_dir / f"{dataset_name}_truncation.parquet"

        standardized.to_parquet(standardized_path, index=False)
        analysis_full.to_parquet(full_path, index=False)
        analysis_selected.to_parquet(selected_path, index=False)
        truncation.to_parquet(truncation_path, index=False)

        print(f"\n{dataset_name}")
        print(f"  Standardized observations: {len(standardized):,}")
        print(f"  Full analysis observations: {len(analysis_full):,}")
        print(f"  Selected observations: {len(analysis_selected):,}")
        print(f"  Excluded observations: {len(analysis_full) - len(analysis_selected):,}")
        print(f"  Truncated conditions: {(truncation['low_potential_method'] != 'none').sum():,}")
        print(f"  Saved: {standardized_path}")
        print(f"  Saved: {full_path}")
        print(f"  Saved: {selected_path}")
        print(f"  Saved: {truncation_path}")


if __name__ == "__main__":
    main()
