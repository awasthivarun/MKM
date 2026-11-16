
"""Process Todd's Pt/PtRu data and apply the configured selection rules."""

from mkm.preprocessing.ptru_selection import select_ptru_analysis
from mkm.preprocessing.ptru_todd import TODD_FILES, build_todd_analysis_grid, load_todd_dataset
from mkm.project_paths import ProjectPaths
from mkm.provenance import write_preprocessing_manifest
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

        write_preprocessing_manifest(
            paths.ptru_analysis_dir(medium) / f"{dataset}_preprocessing_manifest.yaml",
            root=paths.root,
            raw_paths=[paths.ptru_raw_dir(medium) / name for name in TODD_FILES[medium]],
            config_paths=[paths.ptru_preprocessing_config_path],
            source_files=[
                "scripts/process_ptru_todd.py", "src/mkm/preprocessing/ptru_todd.py",
                "src/mkm/preprocessing/ptru_common.py", "src/mkm/preprocessing/ptru_selection.py",
                "src/mkm/preprocessing/interpolation.py", "src/mkm/preprocessing/potential.py",
            ],
            products={path.stem: (path, len(frame)) for path, frame in outputs.items()},
        )

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
