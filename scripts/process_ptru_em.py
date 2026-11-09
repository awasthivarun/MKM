
"""Standardize and interpolate Em's Pt/C datasets."""

import yaml

from mkm.preprocessing.ptru_em import EM_WORKBOOKS, load_em_ptc
from mkm.preprocessing.ptru_todd import build_todd_analysis_grid
from mkm.project_paths import ProjectPaths


def main():
    paths = ProjectPaths.discover(__file__)

    with open(paths.config_dir / "preprocessing" / "ptru_todd.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium, workbook in EM_WORKBOOKS.items():
        dataset = f"PtRu_COOx_{medium}"
        raw_path = paths.raw_data_dir / dataset / workbook
        standardized = load_em_ptc(raw_path, medium, config)

        for sample_id, sample in standardized.groupby("sample_id", sort=False):
            analysis = build_todd_analysis_grid(sample, config)
            analysis["sample_id"] = sample_id

            output_root = paths.processed_data_dir / dataset
            standardized_dir = output_root / "standardized"
            analysis_dir = output_root / "analysis"
            standardized_dir.mkdir(parents=True, exist_ok=True)
            analysis_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"{dataset}_Em_{sample_id}"
            standardized.to_save = None  # No changes to the raw workbook.
            sample.to_parquet(standardized_dir / f"{prefix}_replicates.parquet", index=False)
            analysis.to_parquet(analysis_dir / f"{prefix}_full.parquet", index=False)

            print(f"{medium} | {sample_id}")
            print(f"  Standardized: {len(sample):,}")
            print(f"  Analysis: {len(analysis):,}")


if __name__ == "__main__":
    main()
