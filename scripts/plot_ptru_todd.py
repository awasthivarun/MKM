
"""Generate Todd and Em Pt/PtRu rate plots from processed data."""

import pandas as pd

from mkm.preprocessing.ptru_plotting import save_rate_figures
from mkm.project_paths import ProjectPaths
from mkm.workflows.configuration import load_yaml


def main():
    paths = ProjectPaths.discover(__file__)
    config = load_yaml(paths.ptru_preprocessing_config_path)

    for medium in ("acidic", "basic"):
        dataset = paths.ptru_dataset_name(medium)
        analysis_dir = paths.ptru_analysis_dir(medium)
        figure_dir = paths.ptru_preprocessing_figure_dir(medium)

        # Use full Todd curves so excluded regions remain visible.
        analysis = pd.read_parquet(paths.ptru_full_path(medium))
        truncation = pd.read_parquet(paths.ptru_truncation_path(medium))

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

        # Em preparations remain independent and untruncated.
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
