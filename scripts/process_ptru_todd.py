
"""Process Todd's Pt/PtRu data and apply the configured selection rules."""

from mkm.preprocessing.ptru_selection import select_ptru_analysis
from mkm.preprocessing.ptru_todd import build_todd_analysis_grid, load_todd_dataset
from mkm.project_paths import ProjectPaths
from mkm.workflows.configuration import load_yaml


def main():
    paths = ProjectPaths.discover(__file__)
    config = load_yaml(paths.ptru_preprocessing_config_path)

    for medium in ("acidic", "basic"):
        dataset = paths.ptru_dataset_name(medium)

        standardized = load_todd_dataset(paths.raw_data_dir, medium, config)
        analysis_full = build_todd_analysis_grid(standardized, config)
        analysis_selected, truncation = select_ptru_analysis(analysis_full, config)

        outputs = {
            paths.ptru_standardized_path(medium): standardized,
            paths.ptru_full_path(medium): analysis_full,
            paths.ptru_selected_path(medium): analysis_selected,
            paths.ptru_truncation_path(medium): truncation,
        }

        for path, frame in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)

        print(f"\n{dataset}")
        print(f"  Standardized observations: {len(standardized):,}")
        print(f"  Full analysis observations: {len(analysis_full):,}")
        print(f"  Selected observations: {len(analysis_selected):,}")
        print(f"  Excluded observations: {len(analysis_full) - len(analysis_selected):,}")
        print(f"  Truncated conditions: {(truncation['low_potential_method'] != 'none').sum():,}")

        for path in outputs:
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
