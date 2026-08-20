import numpy as np


class DataValidationError(ValueError):
    """Raised when experimental data violate required structural constraints."""


def validate_material(material, expected_materials):
    if material not in expected_materials:
        raise DataValidationError(
            f"Unexpected material '{material}'. "
            f"Expected one of: {expected_materials}."
        )


def validate_condition(value, expected_values, name):
    if value not in expected_values:
        raise DataValidationError(
            f"Unexpected {name} value '{value}'. "
            f"Expected one of: {expected_values}."
        )


def validate_replicate(replicate, expected_replicates):
    if replicate not in expected_replicates:
        raise DataValidationError(
            f"Unexpected replicate '{replicate}'. "
            f"Expected one of: {expected_replicates}."
        )


def validate_curve_arrays(potential, ln_current, expected_points=None):
    potential = np.asarray(potential)
    ln_current = np.asarray(ln_current)

    if potential.ndim != 1:
        raise DataValidationError(
            f"Potential must be one-dimensional; got shape {potential.shape}."
        )

    if ln_current.ndim != 1:
        raise DataValidationError(
            f"ln(j) must be one-dimensional; got shape {ln_current.shape}."
        )

    if len(potential) == 0:
        raise DataValidationError("Potential array is empty.")

    if len(ln_current) == 0:
        raise DataValidationError("ln(j) array is empty.")

    if len(potential) != len(ln_current):
        raise DataValidationError(
            f"Potential and ln(j) lengths do not match: "
            f"{len(potential)} vs {len(ln_current)}."
        )

    if expected_points is not None and len(potential) != expected_points:
        raise DataValidationError(
            f"Expected {expected_points} points; found {len(potential)}."
        )

    if not np.issubdtype(potential.dtype, np.number):
        raise DataValidationError("Potential contains non-numeric values.")

    if not np.issubdtype(ln_current.dtype, np.number):
        raise DataValidationError("ln(j) contains non-numeric values.")

    if not np.all(np.isfinite(potential)):
        raise DataValidationError(
            "Potential contains missing or non-finite values."
        )

    if not np.all(np.isfinite(ln_current)):
        raise DataValidationError(
            "ln(j) contains missing or non-finite values."
        )


def validate_common_grid(potential, replicate_arrays):
    potential = np.asarray(potential)

    if potential.ndim != 1:
        raise DataValidationError(
            f"Common potential grid must be one-dimensional; "
            f"got shape {potential.shape}."
        )

    if len(potential) == 0:
        raise DataValidationError("Common potential grid is empty.")

    if not np.issubdtype(potential.dtype, np.number):
        raise DataValidationError(
            "Common potential grid contains non-numeric values."
        )

    if not np.all(np.isfinite(potential)):
        raise DataValidationError(
            "Common potential grid contains missing or non-finite values."
        )

    for replicate, values in replicate_arrays.items():
        values = np.asarray(values)

        if values.ndim != 1:
            raise DataValidationError(
                f"Replicate '{replicate}' must be one-dimensional; "
                f"got shape {values.shape}."
            )

        if len(values) != len(potential):
            raise DataValidationError(
                f"Replicate '{replicate}' has {len(values)} points, "
                f"but the common potential grid has {len(potential)}."
            )

        if not np.issubdtype(values.dtype, np.number):
            raise DataValidationError(
                f"Replicate '{replicate}' contains non-numeric values."
            )

        if not np.all(np.isfinite(values)):
            raise DataValidationError(
                f"Replicate '{replicate}' contains missing or non-finite values."
            )


def validate_standardized_dataframe(df, config):
    required_columns = {
        "material",
        "C_KOH_M",
        "CO_mole_fraction",
        "replicate",
        "point_index",
        "E_V_SHE",
        "ln_j_uA_cm2",
        "j_uA_cm2",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise DataValidationError(
            f"Standardized dataset is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    if df.empty:
        raise DataValidationError("Standardized dataset is empty.")

    if df[list(required_columns)].isna().any().any():
        raise DataValidationError(
            "Standardized dataset contains missing values."
        )

    unexpected_materials = set(df["material"]) - set(config["materials"])
    if unexpected_materials:
        raise DataValidationError(
            f"Unexpected materials in standardized dataset: "
            f"{sorted(unexpected_materials)}."
        )

    unexpected_KOH = (
        set(df["C_KOH_M"]) - set(config["KOH_concentrations_M"])
    )
    if unexpected_KOH:
        raise DataValidationError(
            f"Unexpected KOH concentrations in standardized dataset: "
            f"{sorted(unexpected_KOH)}."
        )

    unexpected_CO = (
        set(df["CO_mole_fraction"]) - set(config["CO_mole_fractions"])
    )
    if unexpected_CO:
        raise DataValidationError(
            f"Unexpected CO mole fractions in standardized dataset: "
            f"{sorted(unexpected_CO)}."
        )

    unexpected_replicates = (
        set(df["replicate"]) - set(config["replicates"])
    )
    if unexpected_replicates:
        raise DataValidationError(
            f"Unexpected replicates in standardized dataset: "
            f"{sorted(unexpected_replicates)}."
        )

    current = df["j_uA_cm2"].to_numpy()

    if not np.all(np.isfinite(current)):
        raise DataValidationError(
            "Reconstructed current density contains non-finite values."
        )

    if not np.all(current > 0):
        raise DataValidationError(
            "Reconstructed current density must be strictly positive."
        )

    key_columns = [
        "material",
        "C_KOH_M",
        "CO_mole_fraction",
        "replicate",
        "point_index",
    ]

    if df.duplicated(key_columns).any():
        raise DataValidationError(
            "Standardized dataset contains duplicate "
            "material × KOH × CO × replicate × point-index entries."
        )

    expected_points = config.get("potential", {}).get("expected_points")

    if expected_points is not None:
        curve_columns = [
            "material",
            "C_KOH_M",
            "CO_mole_fraction",
            "replicate",
        ]

        curve_sizes = df.groupby(curve_columns, sort=False).size()
        invalid_sizes = curve_sizes[curve_sizes != expected_points]

        if not invalid_sizes.empty:
            raise DataValidationError(
                f"Expected {expected_points} points per curve, but "
                f"{len(invalid_sizes)} curve(s) have a different number."
            )