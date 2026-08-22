from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SELECTED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "AgPd_COOx_basic"
    / "analysis"
    / "AgPd_COOx_basic_selected.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "AgPd_COOx_basic"
    / "experimental_diagnostics"
    / "setup_structure.csv"
)


def main():
    data = pd.read_parquet(SELECTED_PATH).copy()

    point_columns = [
        "material",
        "C_KOH_M",
        "CO_mole_fraction",
        "analysis_grid_index",
    ]

    data["ln_rate_point_mean"] = data.groupby(point_columns)["ln_rate"].transform("mean")
    data["replicate_deviation"] = data["ln_rate"] - data["ln_rate_point_mean"]

    setup_columns = ["material", "C_KOH_M", "replicate"]

    data["setup_offset"] = data.groupby(setup_columns)["replicate_deviation"].transform("mean")
    data["deviation_after_setup_offset"] = data["replicate_deviation"] - data["setup_offset"]

    records = []

    for setup, setup_data in data.groupby(setup_columns, sort=False):
        raw = setup_data["replicate_deviation"].to_numpy()
        residual = setup_data["deviation_after_setup_offset"].to_numpy()

        sse_raw = np.sum(raw**2)
        sse_residual = np.sum(residual**2)

        fraction_explained = np.nan
        if sse_raw > 0:
            fraction_explained = 1.0 - sse_residual / sse_raw

        potential_slopes = []

        for _, co_data in setup_data.groupby("CO_mole_fraction", sort=True):
            if len(co_data) < 2:
                continue

            slope = np.polyfit(
                co_data["E_V_SHE"].to_numpy(),
                co_data["deviation_after_setup_offset"].to_numpy(),
                1,
            )[0]

            potential_slopes.append(slope)

        records.append(
            {
                "material": setup[0],
                "C_KOH_M": setup[1],
                "replicate": setup[2],
                "n_points": len(setup_data),
                "setup_offset_ln_rate": setup_data["setup_offset"].iloc[0],
                "rms_deviation_before_offset": np.sqrt(np.mean(raw**2)),
                "rms_deviation_after_offset": np.sqrt(np.mean(residual**2)),
                "fraction_squared_deviation_explained_by_offset": fraction_explained,
                "median_abs_potential_slope_after_offset": (
                    np.median(np.abs(potential_slopes)) if potential_slopes else np.nan
                ),
            }
        )

    summary = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\n=== ALL MATERIALS ===")
    print(
        summary[
            [
                "setup_offset_ln_rate",
                "rms_deviation_before_offset",
                "rms_deviation_after_offset",
                "fraction_squared_deviation_explained_by_offset",
                "median_abs_potential_slope_after_offset",
            ]
        ].describe().to_string()
    )

    print("\n=== Ag10Pd90 ===")

    ag10 = summary[summary["material"] == "Ag10Pd90"].copy()

    print(
        ag10[
            [
                "C_KOH_M",
                "replicate",
                "setup_offset_ln_rate",
                "rms_deviation_before_offset",
                "rms_deviation_after_offset",
                "fraction_squared_deviation_explained_by_offset",
                "median_abs_potential_slope_after_offset",
            ]
        ].to_string(index=False)
    )

    print("\n=== CO-PRESSURE CORRELATION OF REPLICATE DEVIATIONS ===")

    correlation_records = []

    for (material, c_koh), condition in data.groupby(["material", "C_KOH_M"], sort=False):
        pivot = condition.pivot_table(
            index=["replicate", "analysis_grid_index"],
            columns="CO_mole_fraction",
            values="replicate_deviation",
        )

        correlation = pivot.corr()

        for lower in correlation.columns:
            for upper in correlation.columns:
                if upper <= lower:
                    continue

                correlation_records.append(
                    {
                        "material": material,
                        "C_KOH_M": c_koh,
                        "CO_1": lower,
                        "CO_2": upper,
                        "correlation": correlation.loc[lower, upper],
                    }
                )

    correlations = pd.DataFrame(correlation_records)

    print(
        correlations[correlations["material"] == "Ag10Pd90"]
        .sort_values(["C_KOH_M", "CO_1", "CO_2"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()