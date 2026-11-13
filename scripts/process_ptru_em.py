
"""Standardize and interpolate Em's Pt/C datasets without merging preparations."""

from mkm.preprocessing.ptru_em import EM_WORKBOOKS, load_em_ptc
from mkm.preprocessing.ptru_todd import build_todd_analysis_grid
from mkm.project_paths import ProjectPaths
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

            print(f"{medium} | {sample_id}")
            print(f"  Standardized: {len(sample):,}")
            print(f"  Analysis: {len(analysis):,}")

            for path in outputs:
                print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
