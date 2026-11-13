
"""Generate Todd and Em Pt/PtRu rate figures using the shared preprocessing plotting module."""

from pathlib import Path

import pandas as pd
import yaml

from mkm.preprocessing.ptru_plotting import save_rate_figures

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "preprocessing" / "ptru_todd.yaml"



def main():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        base = ROOT / "data" / "processed" / dataset
        analysis_dir = base / "analysis"
        figure_dir = ROOT / "figures" / "preprocessing" / dataset

        # Use full data for plotting so excluded regions remain visible.
        analysis = pd.read_parquet(analysis_dir / f"{dataset}_full.parquet")
        truncation = pd.read_parquet(analysis_dir / f"{dataset}_truncation.parquet")

        for material in config["materials"]:
            material_analysis = analysis.loc[analysis["material"] == material]
            if material_analysis.empty:
                raise ValueError(f"Missing Todd data for {material} in {medium}")

            material_cutoffs = truncation.loc[
                (truncation["material"] == material)
                & (truncation["low_potential_method"] != "none")
            ]

            cutoffs = {
                (float(row.C_electrolyte_M), float(row.CO_mole_fraction)): float(row.E_cut_V_SHE)
                for row in material_cutoffs.itertuples(index=False)
            }

            save_rate_figures(
                material_analysis, material, medium, config,
                figure_dir / material, cutoffs=cutoffs or None,
            )

        # Em remains untruncated and separate from Todd.
        em_paths = sorted(analysis_dir.glob(f"{dataset}_Em_*_full.parquet"))

        if not em_paths:
            print(f"No processed Em data found for {medium}; skipping Em plots.")

        for analysis_path in em_paths:
            prefix = analysis_path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")
            em_analysis = pd.read_parquet(analysis_path)

            em_config = {
                **config,
                "electrolyte_concentrations_M": sorted(em_analysis["C_electrolyte_M"].unique()),
                "CO_mole_fractions": sorted(em_analysis["CO_mole_fraction"].unique()),
            }

            save_rate_figures(
                em_analysis, "Pt100", medium, em_config,
                figure_dir / f"Pt100_Em_{sample_id}",
                source_label=f"Em - {sample_id}",
            )


if __name__ == "__main__":
    main()
