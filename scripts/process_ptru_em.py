
"""Standardize and interpolate Em's Pt/C datasets without merging preparations."""

from mkm.preprocessing.ptru_em import EM_WORKBOOKS, load_em_ptc
from mkm.preprocessing.ptru_todd import build_todd_analysis_grid
from mkm.project_paths import ProjectPaths
from mkm.provenance import write_preprocessing_manifest
from mkm.workflows.configuration import load_yaml


def main():
    paths = ProjectPaths.discover(__file__)
    config = load_yaml(paths.ptru_preprocessing_config_path)

    for medium, workbook in EM_WORKBOOKS.items():
        raw_path = paths.ptru_raw_dir(medium) / workbook
        standardized = load_em_ptc(raw_path, medium, config)

        for sample_id, sample in standardized.groupby("sample_id", sort=False):
            analysis = build_todd_analysis_grid(sample, config)
            analysis["sample_id"] = sample_id

            outputs = {
                paths.ptru_em_path(medium, sample_id, "replicates"): sample,
                paths.ptru_em_path(medium, sample_id, "full"): analysis,
            }

            for path, frame in outputs.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)

            write_preprocessing_manifest(
                paths.ptru_analysis_dir(medium) / f"{paths.ptru_dataset_name(medium)}_Em_{sample_id}_manifest.yaml",
                root=paths.root, raw_paths=[raw_path],
                config_paths=[paths.ptru_preprocessing_config_path],
                source_files=[
                    "scripts/process_ptru_em.py", "src/mkm/preprocessing/ptru_em.py",
                    "src/mkm/preprocessing/ptru_common.py", "src/mkm/preprocessing/ptru_todd.py",
                    "src/mkm/preprocessing/interpolation.py", "src/mkm/preprocessing/potential.py",
                ],
                products={path.stem: (path, len(frame)) for path, frame in outputs.items()},
            )

            print(f"{medium} | {sample_id}")
            print(f"  Standardized: {len(sample):,}")
            print(f"  Analysis: {len(analysis):,}")

            for path in outputs:
                print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
