
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.constants import F_C_mol, R_J_mol_K
from mkm.model_data import build_model_data
from mkm.postprocessing.diagnostics import build_physical_checks
from mkm.postprocessing.model_points import summarize_model_points
from mkm.postprocessing.second_order import build_second_order_outputs
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_postprocess import (
    build_ptru_observable_comparisons,
    load_ptru_experimental_observables,
    ptru_electrolyte_label,
    ptru_ion_name,
)


def synthetic_ptru_run(medium):
    records = []

    for concentration in (0.25, 0.50):
        for co_fraction in (0.01, 0.10):
            for grid_index in range(5):
                potential = 0.01 * grid_index
                ln_rate = (
                    -2.0 + 2.0 * potential
                    + 0.3 * np.log(concentration)
                    + 0.2 * np.log(co_fraction)
                )

                records.append({
                    "material": "Pt100",
                    "C_electrolyte_M": concentration,
                    "CO_mole_fraction": co_fraction,
                    "replicate": "1",
                    "analysis_grid_index": grid_index,
                    "E_V_SHE": potential,
                    "rate_s_inv": np.exp(ln_rate),
                    "ln_rate": ln_rate,
                })

    data = build_model_data(
        pd.DataFrame(records),
        electrolyte_concentration_column="C_electrolyte_M",
    )

    points = data.model_points
    ln_rate = (
        -2.0 + 2.0 * points["E_V_SHE"].to_numpy()
        + 0.3 * np.log(points["electrolyte_concentration_M"].to_numpy())
        + 0.2 * np.log(points["CO_mole_fraction"].to_numpy())
    )

    n_points = len(points)
    shape = (2, 3, n_points)

    posterior = xr.Dataset({
        "ln_rate_model": (
            ("chain", "draw", "model_point"),
            np.broadcast_to(ln_rate, shape).copy(),
        ),
        "theta_CO": (
            ("chain", "draw", "model_point"),
            np.full(shape, 0.3),
        ),
        "theta_OH": (
            ("chain", "draw", "model_point"),
            np.full(shape, 0.2),
        ),
        "theta_empty": (
            ("chain", "draw", "model_point"),
            np.full(shape, 0.5),
        ),
        "rate_fraction_ER": (
            ("chain", "draw", "model_point"),
            np.full(shape, 0.7),
        ),
        "rate_fraction_LH": (
            ("chain", "draw", "model_point"),
            np.full(shape, 0.3),
        ),
    })

    return SimpleNamespace(
        specification=SimpleNamespace(medium=medium),
        model_data=data,
        inference_data=SimpleNamespace(posterior=posterior),
    )


def write_synthetic_observables(paths, run, temperature_K=293.15):
    medium = run.specification.medium
    dataset = paths.ptru_dataset_name(medium)
    folder = paths.ptru_analysis_dir(medium)
    folder.mkdir(parents=True, exist_ok=True)

    points = run.model_data.model_points
    concentration_column = "C_electrolyte_M"
    ion_name = ptru_ion_name(medium)

    alpha = points[[
        "material", "electrolyte_concentration_M", "CO_mole_fraction",
        "analysis_grid_index", "E_V_SHE",
    ]].rename(columns={"electrolyte_concentration_M": concentration_column})

    alpha["alpha_mean"] = 2.0 * R_J_mol_K * temperature_K / F_C_mol
    alpha["alpha_sd"] = 0.01
    alpha["medium"] = medium

    ion = points[[
        "material", "CO_mole_fraction", "analysis_grid_index", "E_V_SHE",
    ]].drop_duplicates().copy()
    ion[ion_name] = 0.3
    ion[f"{ion_name}_sd"] = 0.01
    ion["medium"] = medium

    co = alpha.loc[np.isclose(alpha["CO_mole_fraction"], 0.01), [
        "material", concentration_column, "analysis_grid_index", "E_V_SHE",
    ]].copy()
    co["CO_lower_mole_fraction"] = 0.01
    co["CO_upper_mole_fraction"] = 0.10
    co["delta_CO"] = 0.2
    co["delta_CO_sd"] = 0.01
    co["medium"] = medium

    alpha.to_parquet(folder / f"{dataset}_alpha_summary_selected.parquet", index=False)
    ion.to_parquet(folder / f"{dataset}_{ion_name}_selected.parquet", index=False)
    co.to_parquet(folder / f"{dataset}_delta_CO_selected.parquet", index=False)


@pytest.mark.parametrize("medium,ion,label", [
    ("acidic", "delta_Hplus", "HClO4"),
    ("basic", "delta_OH", "KOH"),
])
def test_ptru_medium_labels(medium, ion, label):
    assert ptru_ion_name(medium) == ion
    assert ptru_electrolyte_label(medium) == label


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_ptru_model_point_summaries_and_balance(medium):
    run = synthetic_ptru_run(medium)
    posterior = run.inference_data.posterior

    points = summarize_model_points(
        run.inference_data,
        run.model_data.model_points,
        ("theta_empty", "theta_CO", "theta_OH", "rate_fraction_ER", "rate_fraction_LH"),
    )

    assert len(points) == len(run.model_data.model_points)
    assert {"rate_model_median", "rate_model_hdi95_lower", "theta_empty_median"} <= set(points)

    checks = build_physical_checks(posterior)
    balances = checks.loc[checks["check_type"] == "balance"].set_index("variable")

    assert {"Pt_site_balance", "pathway_fraction_sum"} <= set(balances.index)
    np.testing.assert_allclose(balances["max_abs_error"], 0.0, atol=1e-12)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_ptru_observable_comparisons_preserve_medium(tmp_path, medium):
    paths = ProjectPaths(root=tmp_path)
    run = synthetic_ptru_run(medium)
    write_synthetic_observables(paths, run)

    experimental = load_ptru_experimental_observables(paths, medium)
    ion = ptru_ion_name(medium)

    assert set(experimental) == {"alpha", ion, "delta_CO"}

    points, summary = build_ptru_observable_comparisons(run, paths, 293.15)

    assert not points.empty
    assert set(points["observable"]) == {"alpha", ion, "delta_CO"}
    assert set(summary["observable"]) == {"alpha", ion, "delta_CO"}
    assert "C_electrolyte_M" in points.columns
    assert "C_KOH_M" not in points.columns

    for name in ("alpha", ion, "delta_CO"):
        subset = points.loc[points["observable"] == name]
        assert not subset.empty
        np.testing.assert_allclose(subset["residual_median"], 0, atol=1e-10)


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_ptru_second_order_uses_correct_ion_name(tmp_path, medium):
    paths = ProjectPaths(root=tmp_path)
    run = synthetic_ptru_run(medium)
    write_synthetic_observables(paths, run)

    observable_points, _ = build_ptru_observable_comparisons(run, paths, 293.15)
    ion = ptru_ion_name(medium)

    result = build_second_order_outputs(
        inference_data=run.inference_data,
        model_points=run.model_data.model_points,
        conditions=run.model_data.conditions,
        observable_points=observable_points,
        temperature_K=293.15,
        koh_values=(0.25, 0.50),
        co_values=(0.01, 0.10),
        potential_step_V=0.01,
        concentration_column="C_electrolyte_M",
        ion_name=ion,
    )

    assert not result.points.empty
    assert set(result.points["observable"]) == {
        "dalpha_dE", f"d{ion}_dE", "ddelta_CO_dE", "delta2",
    }
    assert "C_KOH_M" not in result.points.columns

    delta2 = result.points.loc[result.points["observable"] == "delta2"]
    assert not delta2.empty
    np.testing.assert_allclose(delta2["median"], 0, atol=1e-9)
