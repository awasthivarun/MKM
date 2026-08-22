from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .validation import (
    DataValidationError,
    validate_common_grid,
    validate_condition,
    validate_curve_arrays,
    validate_material,
    validate_replicate,
    validate_standardized_dataframe,
)
from .observables import add_transfer_coefficients, calculate_log_slope_order, calculate_paired_adjacent_log_orders


def _parse_co_mole_fraction(sheet_name, expected_values):
    sheet_name = str(sheet_name).strip()

    if not sheet_name.endswith("% CO"):
        raise DataValidationError(f"Unexpected worksheet name '{sheet_name}'. Expected a name such as '0.1% CO'.")

    percent_text = sheet_name.removesuffix("% CO").strip()

    try:
        percent = float(percent_text)
    except ValueError as exc:
        raise DataValidationError(f"Could not determine CO composition from worksheet '{sheet_name}'.") from exc

    mole_fraction = percent / 100.0

    validate_condition(mole_fraction, expected_values, "CO mole fraction")

    return mole_fraction


def _parse_current_column(column_name, config):
    column_name = str(column_name).strip()
    parts = column_name.split()

    if len(parts) != 4:
        raise DataValidationError(f"Unexpected current-density column name '{column_name}'.")

    KOH_text, representation, unit, replicate_text = parts

    if not KOH_text.endswith("M"):
        raise DataValidationError(f"Could not determine KOH concentration from column '{column_name}'.")

    if representation != "lnj":
        raise DataValidationError(f"Unexpected current representation in column '{column_name}'.")

    if unit != "uA/cm2":
        raise DataValidationError(f"Unexpected current-density unit in column '{column_name}'.")

    if not (replicate_text.startswith("(") and replicate_text.endswith(")")):
        raise DataValidationError(f"Could not determine replicate from column '{column_name}'.")

    try:
        C_KOH_M = float(KOH_text.removesuffix("M"))
    except ValueError as exc:
        raise DataValidationError(f"Could not determine KOH concentration from column '{column_name}'.") from exc

    replicate = replicate_text.removeprefix("(").removesuffix(")")

    validate_condition(C_KOH_M, config["KOH_concentrations_M"], "KOH concentration")
    validate_replicate(replicate, config["replicates"])

    return C_KOH_M, replicate


def load_agpd_workbook(path, config):
    path = Path(path)

    material = path.stem.removesuffix("_current_densities")

    validate_material(material, config["materials"])

    workbook = pd.ExcelFile(path)

    expected_sheet_count = len(config["CO_mole_fractions"])

    if len(workbook.sheet_names) != expected_sheet_count:
        raise DataValidationError(
            f"Workbook '{path.name}' contains {len(workbook.sheet_names)} worksheets; expected {expected_sheet_count}."
        )

    frames = []
    observed_CO = set()

    for sheet_name in workbook.sheet_names:
        CO_mole_fraction = _parse_co_mole_fraction(sheet_name, config["CO_mole_fractions"])

        if CO_mole_fraction in observed_CO:
            raise DataValidationError(
                f"Workbook '{path.name}' contains duplicate CO composition {CO_mole_fraction}."
            )

        observed_CO.add(CO_mole_fraction)

        sheet = pd.read_excel(workbook, sheet_name=sheet_name)

        if sheet.columns[0] != "E (V vs SHE)":
            raise DataValidationError(
                f"Unexpected potential column '{sheet.columns[0]}' in worksheet '{sheet_name}'. Expected 'E (V vs SHE)'."
            )

        potential = sheet["E (V vs SHE)"].to_numpy()

        observed_conditions = set()

        for column_name in sheet.columns[1:]:
            C_KOH_M, replicate = _parse_current_column(column_name, config)

            condition = (C_KOH_M, replicate)

            if condition in observed_conditions:
                raise DataValidationError(
                    f"Duplicate KOH/replicate combination {condition} in worksheet '{sheet_name}'."
                )

            observed_conditions.add(condition)

            ln_current = sheet[column_name].to_numpy()

            validate_curve_arrays(potential, ln_current, expected_points=config["potential"].get("expected_points"))

            current = np.exp(ln_current)

            curve = pd.DataFrame(
                {
                    "material": material,
                    "C_KOH_M": C_KOH_M,
                    "CO_mole_fraction": CO_mole_fraction,
                    "replicate": replicate,
                    "point_index": np.arange(len(potential)),
                    "E_V_SHE": potential,
                    "ln_j_uA_cm2": ln_current,
                    "j_uA_cm2": current,
                }
            )

            frames.append(curve)

        expected_conditions = {
            (C_KOH_M, replicate)
            for C_KOH_M in config["KOH_concentrations_M"]
            for replicate in config["replicates"]
        }

        if observed_conditions != expected_conditions:
            missing = expected_conditions - observed_conditions
            unexpected = observed_conditions - expected_conditions

            raise DataValidationError(
                f"Unexpected condition structure in worksheet '{sheet_name}'. "
                f"Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
            )

    expected_CO = set(config["CO_mole_fractions"])

    if observed_CO != expected_CO:
        missing = expected_CO - observed_CO
        unexpected = observed_CO - expected_CO

        raise DataValidationError(
            f"Unexpected CO worksheet structure in '{path.name}'. "
            f"Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
        )

    return pd.concat(frames, ignore_index=True)


def load_agpd_dataset(raw_dir, config):
    raw_dir = Path(raw_dir)

    frames = []

    for material in config["materials"]:
        path = raw_dir / f"{material}_current_densities.xlsx"

        if not path.exists():
            raise FileNotFoundError(f"Expected raw workbook not found: {path}")

        frames.append(load_agpd_workbook(path, config))

    standardized = pd.concat(frames, ignore_index=True)

    validate_standardized_dataframe(standardized, config)

    return standardized


_CONDITION_COLUMNS = ["material", "C_KOH_M", "CO_mole_fraction"]


def _monotonic_direction(potential):
    differences = np.diff(potential)

    if np.all(differences > 0):
        return 1

    if np.all(differences < 0):
        return -1

    return 0


def _as_increasing_curve(potential, values, label):
    potential = np.asarray(potential)
    values = np.asarray(values)

    direction = _monotonic_direction(potential)

    if direction == 0:
        raise DataValidationError(
            f"{label} has a non-monotonic potential vector. The relevant sweep must be selected before interpolation."
        )

    if direction < 0:
        potential = potential[::-1]
        values = values[::-1]

    return potential, values


def _build_analysis_grid(overlap_min, overlap_max, spacing, origin):
    if spacing <= 0:
        raise DataValidationError("Analysis-grid spacing must be greater than zero.")

    if overlap_max <= overlap_min:
        raise DataValidationError("Replicate potential ranges do not have a usable overlap.")

    tolerance = 1e-10

    first_index = int(np.ceil((overlap_min - origin - tolerance) / spacing))
    last_index = int(np.floor((overlap_max - origin + tolerance) / spacing))

    if last_index < first_index:
        raise DataValidationError("Replicate potential overlap contains no points on the configured analysis grid.")

    analysis_grid_index = np.arange(first_index, last_index + 1, dtype=int)
    potential = origin + analysis_grid_index * spacing

    return analysis_grid_index, potential


def _interpolate_log_current(potential, ln_current, analysis_potential, condition, replicate):
    potential, ln_current = _as_increasing_curve(potential, ln_current, label=f"{condition}, replicate {replicate}")

    tolerance = 1e-10

    if analysis_potential[0] < potential[0] - tolerance or analysis_potential[-1] > potential[-1] + tolerance:
        raise DataValidationError(
            f"Analysis grid for {condition} extends outside the measured potential range of replicate '{replicate}'."
        )

    interpolator = PchipInterpolator(potential, ln_current, extrapolate=False)
    interpolated = interpolator(analysis_potential)

    if not np.all(np.isfinite(interpolated)):
        raise DataValidationError(f"Interpolation produced non-finite values for {condition}, replicate '{replicate}'.")

    return interpolated


def build_agpd_analysis_grid(standardized, config):
    spacing = config["analysis_grid"]["spacing_V"]
    origin = config["analysis_grid"]["origin_V"]

    frames = []

    grouped = standardized.groupby(_CONDITION_COLUMNS, sort=False)

    for condition_values, condition_df in grouped:
        condition = tuple(condition_values)

        observed_replicates = set(condition_df["replicate"])
        expected_replicates = set(config["replicates"])

        if observed_replicates != expected_replicates:
            missing = expected_replicates - observed_replicates
            unexpected = observed_replicates - expected_replicates

            raise DataValidationError(
                f"Unexpected replicate structure for {condition}. "
                f"Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
            )

        curves = {}

        for replicate in config["replicates"]:
            replicate_df = condition_df[condition_df["replicate"] == replicate].sort_values("point_index")

            potential, ln_current = _as_increasing_curve(
                replicate_df["E_V_SHE"].to_numpy(),
                replicate_df["ln_j_uA_cm2"].to_numpy(),
                label=f"{condition}, replicate {replicate}",
            )

            curves[replicate] = {"potential": potential, "ln_current": ln_current}

        overlap_min = max(curve["potential"][0] for curve in curves.values())
        overlap_max = min(curve["potential"][-1] for curve in curves.values())

        analysis_grid_index, analysis_potential = _build_analysis_grid(
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            spacing=spacing,
            origin=origin,
        )

        aligned_log_current = {}

        for replicate in config["replicates"]:
            curve = curves[replicate]

            aligned_log_current[replicate] = _interpolate_log_current(
                potential=curve["potential"],
                ln_current=curve["ln_current"],
                analysis_potential=analysis_potential,
                condition=condition,
                replicate=replicate,
            )

        validate_common_grid(analysis_potential, aligned_log_current)

        for replicate in config["replicates"]:
            ln_current = aligned_log_current[replicate]

            frame = pd.DataFrame(
                {
                    "material": condition[0],
                    "C_KOH_M": condition[1],
                    "CO_mole_fraction": condition[2],
                    "replicate": replicate,
                    "analysis_grid_index": analysis_grid_index,
                    "E_V_SHE": analysis_potential,
                    "ln_j_uA_cm2": ln_current,
                    "j_uA_cm2": np.exp(ln_current),
                }
            )

            frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def add_agpd_rates(analysis, config):
    normalization = config["rate_normalization"]

    electrons_per_CO = normalization["electrons_per_CO_oxidation"]
    site_charge_density = normalization["site_charge_density_uC_cm2_Pd"]

    if electrons_per_CO <= 0:
        raise DataValidationError("Electrons per CO oxidation must be greater than zero.")

    if site_charge_density <= 0:
        raise DataValidationError("Site charge density must be greater than zero.")

    rate_denominator = electrons_per_CO * site_charge_density

    result = analysis.copy()

    result["rate_s_inv"] = result["j_uA_cm2"] / rate_denominator
    result["ln_rate"] = result["ln_j_uA_cm2"] - np.log(rate_denominator)

    if not np.all(np.isfinite(result["rate_s_inv"])):
        raise DataValidationError("Rate conversion produced non-finite values.")

    if not np.all(result["rate_s_inv"] > 0):
        raise DataValidationError("Rate must be strictly positive.")

    if not np.all(np.isfinite(result["ln_rate"])):
        raise DataValidationError("Log-rate conversion produced non-finite values.")

    return result


def add_agpd_transfer_coefficients(analysis_rates, config):
    temperature_K = config["temperature_K"]

    if temperature_K <= 0:
        raise DataValidationError("Temperature must be greater than zero kelvin.")

    return add_transfer_coefficients(
        data=analysis_rates,
        group_columns=["material", "C_KOH_M", "CO_mole_fraction", "replicate"],
        temperature_K=temperature_K,
        potential_column="E_V_SHE",
        log_rate_column="ln_rate",
        output_column="alpha",
    )


def summarize_agpd_replicates(analysis_rates, config):
    group_columns = ["material", "C_KOH_M", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE"]

    expected_replicates = len(config["replicates"])

    replicate_counts = analysis_rates.groupby(group_columns, sort=False)["replicate"].nunique()
    invalid_counts = replicate_counts[replicate_counts != expected_replicates]

    if not invalid_counts.empty:
        raise DataValidationError(
            f"Expected {expected_replicates} replicates at every AgPd analysis-grid point, "
            f"but found {len(invalid_counts)} point(s) with a different count."
        )

    aggregations = {
        "n_replicates": ("replicate", "nunique"),
        "j_mean_uA_cm2": ("j_uA_cm2", "mean"),
        "j_sd_uA_cm2": ("j_uA_cm2", "std"),
        "ln_j_mean": ("ln_j_uA_cm2", "mean"),
        "ln_j_sd": ("ln_j_uA_cm2", "std"),
        "rate_mean_s_inv": ("rate_s_inv", "mean"),
        "rate_sd_s_inv": ("rate_s_inv", "std"),
        "ln_rate_mean": ("ln_rate", "mean"),
        "ln_rate_sd": ("ln_rate", "std"),
    }

    if "alpha" in analysis_rates.columns:
        aggregations["alpha_mean"] = ("alpha", "mean")
        aggregations["alpha_sd"] = ("alpha", "std")

    summary = analysis_rates.groupby(group_columns, sort=False).agg(**aggregations).reset_index()

    required_finite_columns = [
        "j_mean_uA_cm2",
        "j_sd_uA_cm2",
        "ln_j_mean",
        "ln_j_sd",
        "rate_mean_s_inv",
        "rate_sd_s_inv",
        "ln_rate_mean",
        "ln_rate_sd",
    ]

    if "alpha_mean" in summary.columns:
        required_finite_columns.extend(["alpha_mean", "alpha_sd"])

    if not np.all(np.isfinite(summary[required_finite_columns].to_numpy())):
        raise DataValidationError("Replicate summary contains non-finite values.")

    if not np.all(summary["rate_mean_s_inv"].to_numpy() > 0):
        raise DataValidationError("Mean rate must be strictly positive.")

    return summary


def _find_sustained_low_rate_cutoff(condition_rates, threshold_s_inv, require_all_replicates):
    grid_summary = (
        condition_rates.groupby(["analysis_grid_index", "E_V_SHE"], sort=True)
        .agg(mean_rate_s_inv=("rate_s_inv", "mean"), min_rate_s_inv=("rate_s_inv", "min"))
        .reset_index()
        .sort_values("E_V_SHE")
    )

    peak_position = int(np.argmax(grid_summary["mean_rate_s_inv"].to_numpy()))

    pre_peak = grid_summary.iloc[: peak_position + 1].copy()

    if require_all_replicates:
        threshold_rate = pre_peak["min_rate_s_inv"].to_numpy()
    else:
        threshold_rate = pre_peak["mean_rate_s_inv"].to_numpy()

    above_threshold = threshold_rate >= threshold_s_inv
    sustained_above_threshold = np.logical_and.accumulate(above_threshold[::-1])[::-1]
    valid_positions = np.flatnonzero(sustained_above_threshold)

    if len(valid_positions) == 0:
        raise DataValidationError(
            f"No sustained low-potential region was found above the rate threshold "
            f"{threshold_s_inv:g} s^-1 before the mean-rate maximum."
        )

    cutoff_position = int(valid_positions[0])

    cutoff_row = pre_peak.iloc[cutoff_position]
    peak_row = grid_summary.iloc[peak_position]

    return {
        "cutoff_grid_index": int(cutoff_row["analysis_grid_index"]),
        "cutoff_E_V_SHE": float(cutoff_row["E_V_SHE"]),
        "peak_grid_index": int(peak_row["analysis_grid_index"]),
        "peak_E_V_SHE": float(peak_row["E_V_SHE"]),
        "peak_rate_mean_s_inv": float(peak_row["mean_rate_s_inv"]),
    }


def truncate_agpd_analysis(analysis_rates, config):
    truncation_config = config["truncation"]

    low_config = truncation_config["low_potential"]
    high_config = truncation_config["high_potential"]

    low_method = low_config["method"]
    high_method = high_config["method"]

    if low_method not in {"none", "rate_threshold"}:
        raise DataValidationError(f"Unsupported low-potential truncation method '{low_method}'.")

    if high_method != "none":
        raise DataValidationError(f"Unsupported high-potential truncation method '{high_method}'.")

    condition_columns = ["material", "C_KOH_M", "CO_mole_fraction"]

    truncated_frames = []
    cutoff_records = []

    grouped = analysis_rates.groupby(condition_columns, sort=False)

    for condition_values, condition_data in grouped:
        condition = tuple(condition_values)

        condition_data = condition_data.sort_values(["E_V_SHE", "replicate"]).copy()

        grid_points = condition_data[["analysis_grid_index", "E_V_SHE"]].drop_duplicates().sort_values("E_V_SHE")

        original_min_E = float(grid_points["E_V_SHE"].min())
        original_max_E = float(grid_points["E_V_SHE"].max())
        original_n_points = len(grid_points)

        if low_method == "none":
            low_cut_grid_index = int(grid_points.iloc[0]["analysis_grid_index"])
            low_cut_E = original_min_E

            mean_rates = (
                condition_data.groupby(["analysis_grid_index", "E_V_SHE"], sort=True)["rate_s_inv"]
                .mean()
                .reset_index()
            )

            peak_position = int(np.argmax(mean_rates["rate_s_inv"].to_numpy()))
            peak_row = mean_rates.iloc[peak_position]

            peak_grid_index = int(peak_row["analysis_grid_index"])
            peak_E = float(peak_row["E_V_SHE"])
            peak_rate = float(peak_row["rate_s_inv"])

            threshold_s_inv = np.nan
            require_all_replicates = np.nan

        else:
            threshold_s_inv = float(low_config["threshold_s_inv"])
            require_all_replicates = bool(low_config["require_all_replicates"])

            if threshold_s_inv <= 0:
                raise DataValidationError("Low-potential rate threshold must be greater than zero.")

            cutoff = _find_sustained_low_rate_cutoff(
                condition_rates=condition_data,
                threshold_s_inv=threshold_s_inv,
                require_all_replicates=require_all_replicates,
            )

            low_cut_grid_index = cutoff["cutoff_grid_index"]
            low_cut_E = cutoff["cutoff_E_V_SHE"]
            peak_grid_index = cutoff["peak_grid_index"]
            peak_E = cutoff["peak_E_V_SHE"]
            peak_rate = cutoff["peak_rate_mean_s_inv"]

        retained = condition_data[condition_data["analysis_grid_index"] >= low_cut_grid_index].copy()
        retained_grid = retained[["analysis_grid_index", "E_V_SHE"]].drop_duplicates().sort_values("E_V_SHE")
        retained_n_points = len(retained_grid)

        if retained_n_points < 3:
            raise DataValidationError(f"Truncation leaves fewer than three potential points for condition {condition}.")

        retained_min_E = float(retained_grid["E_V_SHE"].min())
        retained_max_E = float(retained_grid["E_V_SHE"].max())

        truncated_frames.append(retained)

        cutoff_records.append(
            {
                "material": condition[0],
                "C_KOH_M": condition[1],
                "CO_mole_fraction": condition[2],
                "low_potential_method": low_method,
                "high_potential_method": high_method,
                "threshold_s_inv": threshold_s_inv,
                "require_all_replicates": require_all_replicates,
                "original_min_E_V_SHE": original_min_E,
                "original_max_E_V_SHE": original_max_E,
                "retained_min_E_V_SHE": retained_min_E,
                "retained_max_E_V_SHE": retained_max_E,
                "peak_E_V_SHE": peak_E,
                "peak_rate_mean_s_inv": peak_rate,
                "low_cut_analysis_grid_index": low_cut_grid_index,
                "peak_analysis_grid_index": peak_grid_index,
                "n_points_original": original_n_points,
                "n_points_retained": retained_n_points,
            }
        )

    truncated = pd.concat(truncated_frames, ignore_index=True)
    cutoffs = pd.DataFrame(cutoff_records)

    return truncated, cutoffs


def calculate_agpd_oh_order(summary, config):
    return calculate_log_slope_order(
        data=summary,
        varying_column="C_KOH_M",
        varying_values=config["KOH_concentrations_M"],
        group_columns=["material", "CO_mole_fraction"],
        output_column="delta_OH",
        output_sd_column="delta_OH_sd",
        mean_log_rate_column="ln_rate_mean",
        sd_log_rate_column="ln_rate_sd",
        grid_column="analysis_grid_index",
        potential_column="E_V_SHE",
    )


def calculate_agpd_co_order_replicates(analysis_rates, config):
    return calculate_paired_adjacent_log_orders(
        data=analysis_rates,
        varying_column="CO_mole_fraction",
        varying_values=config["CO_mole_fractions"],
        group_columns=["material", "C_KOH_M"],
        replicate_column="replicate",
        output_column="delta_CO",
        lower_value_column="CO_lower_mole_fraction",
        upper_value_column="CO_upper_mole_fraction",
        log_rate_column="ln_rate",
        grid_column="analysis_grid_index",
        potential_column="E_V_SHE",
    )


def summarize_agpd_co_order_replicates(delta_CO_replicates, config):
    group_columns = [
        "material",
        "C_KOH_M",
        "CO_lower_mole_fraction",
        "CO_upper_mole_fraction",
        "analysis_grid_index",
        "E_V_SHE",
    ]

    expected_replicates = len(config["replicates"])

    replicate_counts = delta_CO_replicates.groupby(group_columns, sort=False)["replicate"].nunique()
    invalid_counts = replicate_counts[replicate_counts != expected_replicates]

    if not invalid_counts.empty:
        raise DataValidationError(
            f"Expected {expected_replicates} paired CO-order replicates at every point, "
            f"but found {len(invalid_counts)} point(s) with a different count."
        )

    summary = (
        delta_CO_replicates.groupby(group_columns, sort=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            delta_CO=("delta_CO", "mean"),
            delta_CO_sd=("delta_CO", "std"),
        )
        .reset_index()
    )

    if not np.all(np.isfinite(summary[["delta_CO", "delta_CO_sd"]].to_numpy())):
        raise DataValidationError("Paired CO-order summary contains non-finite values.")

    return summary


def calculate_agpd_co_order(analysis_rates, config):
    delta_CO_replicates = calculate_agpd_co_order_replicates(analysis_rates, config)

    return summarize_agpd_co_order_replicates(delta_CO_replicates, config)