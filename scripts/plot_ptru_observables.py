
"""Plot processed Todd and Em PtRu observables without recalculating them."""

from argparse import ArgumentParser

from mkm.project_paths import ProjectPaths
from mkm.workflows.configuration import load_yaml

from process_ptru_observables import plot_source


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--medium", choices=("acidic", "basic"), default=None)
    args = parser.parse_args()

    paths = ProjectPaths.discover(__file__)
    config = load_yaml(paths.ptru_preprocessing_config_path)

    media = (args.medium,) if args.medium else ("acidic", "basic")

    for medium in media:
        dataset = paths.ptru_dataset_name(medium)
        analysis_dir = paths.ptru_analysis_dir(medium)
        figure_dir = paths.ptru_preprocessing_figure_dir(medium)

        plot_source(
            analysis_path=paths.ptru_full_path(medium),
            figure_dir=figure_dir,
            config=config,
            medium=medium,
        )

        for analysis_path in sorted(analysis_dir.glob(f"{dataset}_Em_*_full.parquet")):
            prefix = analysis_path.name.removesuffix("_full.parquet")
            sample_id = prefix.removeprefix(f"{dataset}_Em_")

            plot_source(
                analysis_path=analysis_path,
                figure_dir=figure_dir / f"Pt100_Em_{sample_id}",
                config=config,
                medium=medium,
                source_label=f"Em - {sample_id}",
            )


if __name__ == "__main__":
    main()
