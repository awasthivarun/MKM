"""AgPd preprocessing variant that retains only the rising branch through maximum mean TOF."""

from copy import deepcopy

import pandas as pd

from mkm.preprocessing.agpd_basic import truncate_agpd_analysis
from mkm.preprocessing.validation import DataValidationError


CONDITION_COLUMNS = ["material", "C_KOH_M", "CO_mole_fraction"]


def truncate_agpd_analysis_maxtof(analysis_rates, config):
    """Apply the canonical low-rate cutoff and truncate each condition at maximum replicate-mean TOF.

    The lower cutoff is delegated to ``truncate_agpd_analysis`` so this variant remains identical to
    the base dataset on the low-potential side. The upper cutoff is the existing replicate-mean TOF
    maximum recorded by the base truncation routine. All replicates therefore retain a common grid.
    """

    high_method = config["truncation"]["high_potential"]["method"]
    if high_method != "max_tof":
        raise DataValidationError(
            "The maxtof preprocessing variant requires high_potential.method = 'max_tof'."
        )

    base_config = deepcopy(config)
    base_config["truncation"]["high_potential"]["method"] = "none"
    low_selected, cutoffs = truncate_agpd_analysis(analysis_rates=analysis_rates, config=base_config)

    selected_frames = []
    updated_records = []

    for cutoff in cutoffs.itertuples(index=False):
        condition_mask = (
            (low_selected["material"] == cutoff.material)
            & (low_selected["C_KOH_M"] == cutoff.C_KOH_M)
            & (low_selected["CO_mole_fraction"] == cutoff.CO_mole_fraction)
        )
        condition = low_selected.loc[condition_mask].copy()
        condition = condition.loc[
            condition["analysis_grid_index"] <= cutoff.peak_analysis_grid_index
        ].copy()

        retained_grid = (
            condition[["analysis_grid_index", "E_V_SHE"]]
            .drop_duplicates()
            .sort_values("E_V_SHE")
        )
        if len(retained_grid) < 3:
            raise DataValidationError(
                "Maximum-TOF truncation leaves fewer than three potential points for condition "
                f"{(cutoff.material, cutoff.C_KOH_M, cutoff.CO_mole_fraction)}."
            )

        if int(retained_grid.iloc[-1]["analysis_grid_index"]) != int(cutoff.peak_analysis_grid_index):
            raise DataValidationError(
                "Maximum-TOF truncation did not retain the recorded peak grid point for condition "
                f"{(cutoff.material, cutoff.C_KOH_M, cutoff.CO_mole_fraction)}."
            )

        selected_frames.append(condition)
        record = cutoff._asdict()
        record.update(
            {
                "high_potential_method": "max_tof",
                "high_cut_analysis_grid_index": int(cutoff.peak_analysis_grid_index),
                "high_cut_E_V_SHE": float(cutoff.peak_E_V_SHE),
                "retained_max_E_V_SHE": float(retained_grid["E_V_SHE"].max()),
                "n_points_retained": int(len(retained_grid)),
            }
        )
        updated_records.append(record)

    selected = pd.concat(selected_frames, ignore_index=True)
    maxtof_cutoffs = pd.DataFrame(updated_records)
    return selected, maxtof_cutoffs
