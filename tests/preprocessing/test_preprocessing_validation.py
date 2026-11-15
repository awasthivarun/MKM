"""Direct contracts for validating raw experimental curves and standardized AgPd tables."""

import numpy as np
import pandas as pd
import pytest

from mkm.preprocessing.validation import (
    DataValidationError,
    validate_common_grid,
    validate_condition,
    validate_curve_arrays,
    validate_material,
    validate_replicate,
    validate_standardized_dataframe,
)


def _config():
    return {
        "materials": ["Pd100"], "KOH_concentrations_M": [0.5],
        "CO_mole_fractions": [0.01], "replicates": ["A"],
        "potential": {"expected_points": 2},
    }


def _standardized():
    return pd.DataFrame({
        "material": ["Pd100", "Pd100"], "C_KOH_M": [0.5, 0.5],
        "CO_mole_fraction": [0.01, 0.01], "replicate": ["A", "A"],
        "point_index": [0, 1], "E_V_SHE": [0.1, 0.2],
        "ln_j_uA_cm2": [0.0, np.log(2.0)], "j_uA_cm2": [1.0, 2.0],
    })


def test_validate_curve_arrays_accepts_finite_aligned_arrays():
    validate_curve_arrays([0.1, 0.2], [-3.0, -2.0], expected_points=2)


@pytest.mark.parametrize("potential,current,message", [
    ([[0.1, 0.2]], [-3.0, -2.0], "Potential must be one-dimensional"),
    ([0.1, 0.2], [[-3.0, -2.0]], "ln\\(j\\) must be one-dimensional"),
    ([], [], "Potential array is empty"),
    ([0.1], [], "ln\\(j\\) array is empty"),
    ([0.1, 0.2], [-3.0], "lengths do not match"),
    ([0.1, np.nan], [-3.0, -2.0], "Potential contains missing or non-finite"),
    ([0.1, 0.2], [-3.0, np.inf], "ln\\(j\\) contains missing or non-finite"),
    (["bad", "0.2"], [-3.0, -2.0], "Potential contains non-numeric"),
    ([0.1, 0.2], ["bad", "-2"], "ln\\(j\\) contains non-numeric"),
])
def test_validate_curve_arrays_rejects_bad_inputs(potential, current, message):
    with pytest.raises(DataValidationError, match=message):
        validate_curve_arrays(potential, current)


def test_validate_curve_arrays_rejects_wrong_point_count():
    with pytest.raises(DataValidationError, match="Expected 3 points"):
        validate_curve_arrays([0.1, 0.2], [0.0, 1.0], expected_points=3)


@pytest.mark.parametrize("grid,replicates,message", [
    ([[0.1, 0.2]], {"A": [1.0, 2.0]}, "grid must be one-dimensional"),
    ([], {}, "grid is empty"),
    (["a", "b"], {}, "grid contains non-numeric"),
    ([0.1, np.nan], {}, "grid contains missing or non-finite"),
    ([0.1, 0.2], {"A": [[1.0, 2.0]]}, "must be one-dimensional"),
    ([0.1, 0.2], {"A": [1.0]}, "has 1 points"),
    ([0.1, 0.2], {"A": ["bad", "2"]}, "contains non-numeric"),
    ([0.1, 0.2], {"A": [1.0, np.inf]}, "contains missing or non-finite"),
])
def test_validate_common_grid_rejects_bad_inputs(grid, replicates, message):
    with pytest.raises(DataValidationError, match=message):
        validate_common_grid(grid, replicates)


def test_validate_common_grid_accepts_complete_replicates():
    validate_common_grid([0.1, 0.2], {"A": [2.0, 3.0], "B": [4.0, 5.0]})


@pytest.mark.parametrize("validator,args,message", [
    (validate_material, ("Ag100", ["Pd100"]), "Unexpected material"),
    (validate_condition, (0.25, [0.5, 1.0], "KOH"), "Unexpected KOH"),
    (validate_replicate, ("D", ["A", "B", "C"]), "Unexpected replicate"),
])
def test_validate_scalar_categories_reject_unknown_values(validator, args, message):
    with pytest.raises(DataValidationError, match=message):
        validator(*args)


def test_validate_standardized_dataframe_accepts_expected_schema():
    validate_standardized_dataframe(_standardized(), _config())


@pytest.mark.parametrize("mutation,message", [
    (lambda df: df.drop(columns=["E_V_SHE"]), "missing required columns"),
    (lambda df: df.iloc[:0], "dataset is empty"),
    (lambda df: df.assign(E_V_SHE=[0.1, np.nan]), "contains missing values"),
    (lambda df: df.assign(material="Ag100"), "Unexpected materials"),
    (lambda df: df.assign(C_KOH_M=0.25), "Unexpected KOH"),
    (lambda df: df.assign(CO_mole_fraction=0.1), "Unexpected CO"),
    (lambda df: df.assign(replicate="D"), "Unexpected replicates"),
    (lambda df: df.assign(j_uA_cm2=[1.0, np.inf]), "non-finite values"),
    (lambda df: df.assign(j_uA_cm2=[1.0, 0.0]), "strictly positive"),
    (lambda df: df.assign(point_index=[0, 0]), "duplicate material"),
    (lambda df: df.iloc[:1], "Expected 2 points per curve"),
])
def test_validate_standardized_dataframe_rejects_invalid_data(mutation, message):
    with pytest.raises(DataValidationError, match=message):
        validate_standardized_dataframe(mutation(_standardized()), _config())
