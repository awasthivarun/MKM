"""Exercise Pt100's tabular postprocessing without MCMC or experimental files."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mkm.model_data import build_model_data
from mkm.model_inputs import build_model_input_arrays
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_postprocess import postprocess_ptru_run


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_ptru_postprocessing_core_generates_aligned_products(tmp_path, medium):
    records = []
    for concentration in (0.1, 0.5):
        for co_fraction in (0.01, 0.1):
            for replicate in ("1", "2"):
                for index in range(4):
                    potential = 0.25 + index * 0.01
                    ln_rate = -3.0 + potential + 0.3 * np.log(concentration) + 0.2 * np.log(co_fraction)
                    records.append({
                        "material": "Pt100", "C_electrolyte_M": concentration,
                        "CO_mole_fraction": co_fraction, "replicate": replicate,
                        "analysis_grid_index": index, "E_V_SHE": potential,
                        "rate_s_inv": np.exp(ln_rate), "ln_rate": ln_rate,
                    })

    model_data = build_model_data(
        pd.DataFrame(records), electrolyte_concentration_column="C_electrolyte_M",
    )
    inputs = build_model_input_arrays(model_data)
    points = model_data.model_points
    ln_rate = (
        -3.0 + points["E_V_SHE"].to_numpy()
        + 0.3 * np.log(points["electrolyte_concentration_M"].to_numpy())
        + 0.2 * np.log(points["CO_mole_fraction"].to_numpy())
    )
    rng = np.random.default_rng(12)
    shape = (2, 20, len(points))
    perturbation = rng.normal(0.0, 0.015, size=shape)

    posterior = xr.Dataset({
        "ln_rate_model": (("chain", "draw", "model_point"), ln_rate + perturbation),
        "theta_empty": (("chain", "draw", "model_point"), np.full(shape, 0.50)),
        "theta_CO": (("chain", "draw", "model_point"), np.full(shape, 0.30)),
        "theta_OH": (("chain", "draw", "model_point"), np.full(shape, 0.20)),
        "rate_fraction_ER": (("chain", "draw", "model_point"), np.full(shape, 0.70)),
        "rate_fraction_LH": (("chain", "draw", "model_point"), np.full(shape, 0.30)),
        "Gact1_0": (("chain", "draw"), 0.7 + rng.normal(0, 0.01, size=(2, 20))),
        "sigma_rate_abs": (("chain", "draw"), np.exp(rng.normal(-6.5, 0.05, size=(2, 20)))),
        "sigma_rate_rel": (("chain", "draw"), np.exp(rng.normal(-1.6, 0.05, size=(2, 20)))),
    })
    specs = {
        "Gact1_0": {"distribution": "normal", "mu": 0.7, "sigma": 0.1},
        "sigma_rate_abs": {"distribution": "lognormal", "median": 0.0015, "log_sd": 0.5},
        "sigma_rate_rel": {"distribution": "lognormal", "median": 0.2, "log_sd": 0.5},
    }
    paths = ProjectPaths(root=tmp_path)
    run = SimpleNamespace(
        output_dir=paths.ptru_posterior_output_dir(medium, "CO_ER_LH"),
        specification=SimpleNamespace(medium=medium, model_name="CO_ER_LH"),
        inference_data=SimpleNamespace(posterior=posterior),
        model_data=model_data,
        inputs=inputs,
        parameter_specs=specs,
        free_parameter_names=tuple(specs),
        built_model=SimpleNamespace(mechanism_result=SimpleNamespace(pointwise={
            "theta_empty": None, "theta_CO": None, "theta_OH": None,
            "rate_fraction_ER": None, "rate_fraction_LH": None,
        })),
    )

    status = postprocess_ptru_run(
        run, paths, temperature_K=293.15, random_seed=7,
        skip_loo=True, skip_observables=True, plot_level="none",
    )
    assert status == "complete"

    tables = paths.fit_tables_dir(run.output_dir)
    observations = pd.read_parquet(tables / "observation_diagnostics.parquet")
    points_summary = pd.read_parquet(tables / "model_point_diagnostics.parquet")
    checks = pd.read_csv(tables / "physical_checks.csv")
    stages = pd.read_csv(tables / "postprocessing_status.csv").set_index("stage")

    assert len(observations) == len(model_data.observations)
    assert len(points_summary) == len(model_data.model_points)
    assert "rate_model_hdi95_lower" in observations
    assert "theta_CO_median" in points_summary
    assert {"Pt_site_balance", "pathway_fraction_sum"} <= set(checks["variable"])
    assert stages.loc["core", "status"] == "complete"
    assert stages.loc["loo", "status"] == "skipped_user"
    assert stages.loc["observables", "status"] == "skipped_user"
    assert stages.loc["overall", "status"] == "complete"
