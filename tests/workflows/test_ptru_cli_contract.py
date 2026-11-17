"""Unified PtRu CLI and preprocessing output contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mkm.project_paths import ProjectPaths
from mkm.workflows import ptru_fit

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    """Import a script without invoking its main function."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("medium", ["acidic", "basic"])
@pytest.mark.parametrize("model", ["CO_LH", "CO_ER", "CO_ER_LH", "CO_BF_ER_LH"])
def test_unified_fit_cli_options(monkeypatch, medium, model):
    script = load_script("fit_ptru_posterior")
    monkeypatch.setattr(sys, "argv", [
        "fit_ptru_posterior.py", model, "--medium", medium,
        "--all-materials", "--parameterization", "linear_xRu", "--overwrite",
    ])
    args = script.parse_args()
    assert args.model == model and args.medium == medium
    assert args.all_materials and args.parameterization == "linear_xRu"
    assert args.overwrite and not args.resume
    assert args.draws == 1000 and args.tune == 1000 and args.chains == 4 and args.cores == 4
    assert args.target_accept == 0.90 and args.nuts_sampler == "nutpie" and args.backend == "numba"


@pytest.mark.parametrize("medium", ["acidic", "basic"])
def test_unified_postprocess_options(monkeypatch, medium):
    script = load_script("postprocess_ptru_posterior")
    monkeypatch.setattr(sys, "argv", [
        "postprocess_ptru_posterior.py", "CO_ER_LH", "--medium", medium,
        "--material", "Pt100", "--plot-level", "none",
    ])
    args = script.parse_args()
    assert args.model == "CO_ER_LH" and args.medium == medium
    assert args.material == "Pt100" and args.random_seed == 1
    assert not hasattr(args, "draws_per_chain")


def test_fit_requires_medium(monkeypatch):
    script = load_script("fit_ptru_posterior")
    monkeypatch.setattr(sys, "argv", ["fit_ptru_posterior.py", "CO_LH"])
    with pytest.raises(SystemExit):
        script.parse_args()


def test_unified_fit_data_and_output_paths(monkeypatch):
    paths = ProjectPaths(root=ROOT)
    config, _ = ptru_fit.load_ptru_config(paths)
    spec = ptru_fit.resolve_ptru_fit_specification(config, "basic", "CO_ER_LH", material="Pt100")
    sentinel = object()

    def read_parquet(path):
        assert path == paths.ptru_selected_path("basic")
        return pd.DataFrame({"material": ["Pt100", "Pt66Ru33"], "medium": ["basic", "basic"]})

    monkeypatch.setattr(ptru_fit.pd, "read_parquet", read_parquet)
    monkeypatch.setattr(ptru_fit, "build_model_data", lambda frame, **kwargs: object())
    monkeypatch.setattr(ptru_fit, "build_model_input_arrays", lambda _: SimpleNamespace(materials=("Pt100",)))
    monkeypatch.setattr(ptru_fit, "build_ptru_mechanism", lambda *args: object())
    monkeypatch.setattr(ptru_fit, "build_pymc_model", lambda *args, **kwargs: sentinel)
    fit = ptru_fit.build_ptru_fit(paths, spec)
    assert fit.built_model is sentinel
    assert fit.output_dir.parts[-3:] == ("individual", "Pt100", "CO_ER_LH")
    assert fit.model_config_path == paths.config_dir / "models" / "ptru.yaml"
    assert fit.prior_config_path == paths.config_dir / "priors" / "ptru.yaml"
    assert {"sigma_rate_abs", "sigma_rate_rel"} <= set(fit.parameter_specs)


def test_todd_processing_output_contract(tmp_path, monkeypatch):
    script = load_script("process_ptru_todd")
    config_dir = tmp_path / "config" / "preprocessing"
    config_dir.mkdir(parents=True)
    (config_dir / "ptru_todd.yaml").write_text("{}", encoding="utf-8")

    paths = ProjectPaths(root=tmp_path)
    monkeypatch.setattr(script.ProjectPaths, "discover", lambda anchor: paths)

    frame = pd.DataFrame({"medium": ["acidic"], "rate_s_inv": [1.0]})
    cutoffs = pd.DataFrame({"low_potential_method": ["none"]})

    monkeypatch.setattr(script, "load_todd_dataset", lambda *args: frame.copy())
    monkeypatch.setattr(script, "build_todd_analysis_grid", lambda *args: frame.copy())
    monkeypatch.setattr(
        script, "select_ptru_analysis",
        lambda *args: (frame.copy(), cutoffs.copy()),
    )

    written = []
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet",
        lambda self, path, index=False: written.append(Path(path)),
    )

    # Parquet writes are mocked in this contract test; lineage has its own file-based tests.
    monkeypatch.setattr(script, "write_preprocessing_manifest", lambda *args, **kwargs: None)
    script.main()

    expected = set()
    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        root = paths.processed_data_dir / dataset
        expected.update({
            root / "standardized" / f"{dataset}_replicates.parquet",
            root / "analysis" / f"{dataset}_full.parquet",
            root / "analysis" / f"{dataset}_selected.parquet",
            root / "analysis" / f"{dataset}_truncation.parquet",
        })

    assert set(written) == expected


def test_em_processing_output_contract(tmp_path, monkeypatch):
    script = load_script("process_ptru_em")
    config_dir = tmp_path / "config" / "preprocessing"
    config_dir.mkdir(parents=True)
    (config_dir / "ptru_todd.yaml").write_text("{}", encoding="utf-8")

    paths = ProjectPaths(root=tmp_path)
    monkeypatch.setattr(script.ProjectPaths, "discover", lambda anchor: paths)

    frame = pd.DataFrame({
        "sample_id": ["PtC_20pct", "PtC_20pct_1mg"],
        "rate_s_inv": [1.0, 2.0],
    })
    monkeypatch.setattr(script, "load_em_ptc", lambda *args: frame.copy())
    monkeypatch.setattr(script, "build_todd_analysis_grid", lambda data, config: data.copy())

    written = []
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet",
        lambda self, path, index=False: written.append(Path(path)),
    )

    # Parquet writes are mocked in this contract test; lineage has its own file-based tests.
    monkeypatch.setattr(script, "write_preprocessing_manifest", lambda *args, **kwargs: None)
    script.main()

    expected = set()
    for medium in ("acidic", "basic"):
        dataset = f"PtRu_COOx_{medium}"
        root = paths.processed_data_dir / dataset

        for sample_id in ("PtC_20pct", "PtC_20pct_1mg"):
            prefix = f"{dataset}_Em_{sample_id}"
            expected.update({
                root / "standardized" / f"{prefix}_replicates.parquet",
                root / "analysis" / f"{prefix}_full.parquet",
            })

    assert set(written) == expected


@pytest.mark.parametrize(
    "medium,source_label",
    [("acidic", None), ("basic", None), ("acidic", "Em - PtC_20pct")],
)
def test_ptru_observable_output_contract(tmp_path, monkeypatch, medium, source_label):
    script = load_script("process_ptru_observables")
    dataset = f"PtRu_COOx_{medium}"
    suffix = "_Em_PtC_20pct" if source_label else ""
    prefix = f"{dataset}{suffix}"
    analysis_path = tmp_path / f"{prefix}_full.parquet"

    frame = pd.DataFrame({
        "medium": [medium],
        "material": ["Pt100"],
        "C_electrolyte_M": [0.10],
        "CO_mole_fraction": [0.01],
        "analysis_grid_index": [0],
        "E_V_SHE": [0.0],
        "replicate": ["1"],
        "rate_s_inv": [1.0],
        "ln_rate": [0.0],
    })

    monkeypatch.setattr(script.pd, "read_parquet", lambda path: frame.copy())
    monkeypatch.setattr(
        script, "calculate_alpha_products",
        lambda data, temperature: (frame.copy(), frame.copy()),
    )
    monkeypatch.setattr(script, "calculate_co_orders", lambda data: frame.copy())
    monkeypatch.setattr(script, "calculate_ion_orders", lambda *args: frame.copy())
    monkeypatch.setattr(script, "plot_alpha_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "plot_co_order_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "plot_ion_order_grid", lambda *args, **kwargs: None)

    written = []
    figures = []
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet",
        lambda self, path, index=False: written.append(Path(path).name),
    )
    monkeypatch.setattr(
        script, "save_figure",
        lambda fig, path: figures.append(Path(path).name),
    )

    # Parquet writes are mocked in this contract test; lineage has its own file-based tests.
    monkeypatch.setattr(script, "write_preprocessing_manifest", lambda *args, **kwargs: None)
    script.process_source(
        analysis_path=analysis_path,
        figure_dir=tmp_path / "figures",
        config={
            "temperature_K": 293.15,
            "electrolyte_concentrations_M": [0.10],
            "CO_mole_fractions": [0.01],
        },
        medium=medium,
        source_label=source_label,
    )

    expected_tables = {
        f"{prefix}_alpha_replicates.parquet",
        f"{prefix}_alpha_summary.parquet",
        f"{prefix}_delta_CO.parquet",
    }
    expected_figures = {"Pt100_alpha.png", "Pt100_delta_CO.png"}

    if source_label is None:
        ion = "delta_Hplus" if medium == "acidic" else "delta_OH"
        expected_tables.update({
            f"{prefix}_alpha_replicates_selected.parquet",
            f"{prefix}_alpha_summary_selected.parquet",
            f"{prefix}_delta_CO_selected.parquet",
            f"{prefix}_{ion}.parquet",
            f"{prefix}_{ion}_selected.parquet",
        })
        expected_figures.add(f"Pt100_{ion}.png")

    assert set(written) == expected_tables
    assert set(figures) == expected_figures



def test_pt100_posterior_table_interval_contract():
    from types import SimpleNamespace

    import xarray as xr

    from mkm.postprocessing.model_points import summarize_model_points

    ln_rate = np.log(np.array([
        [[1.0, 2.0], [2.0, 3.0]],
        [[3.0, 4.0], [4.0, 5.0]],
    ]))

    theta_co = np.array([
        [[0.2, 0.3], [0.3, 0.4]],
        [[0.4, 0.5], [0.5, 0.6]],
    ])

    posterior = xr.Dataset({
        "ln_rate_model": (("chain", "draw", "model_point"), ln_rate),
        "theta_CO": (("chain", "draw", "model_point"), theta_co),
    })
    inference_data = SimpleNamespace(posterior=posterior)
    model_points = pd.DataFrame({"model_point_id": [0, 1]})

    result = summarize_model_points(
        inference_data,
        model_points,
        pointwise_variables=("theta_CO",),
    )

    expected_statistics = (
        "mean", "sd", "median",
        "hdi80_lower", "hdi80_upper",
        "hdi95_lower", "hdi95_upper",
    )

    for prefix in ("ln_rate_model", "rate_model", "theta_CO"):
        for statistic in expected_statistics:
            assert f"{prefix}_{statistic}" in result

    np.testing.assert_allclose(result["rate_model_median"], [2.5, 3.5])
    np.testing.assert_allclose(result["theta_CO_median"], [0.35, 0.45])
    assert result["rate_model_hdi95_lower"].le(result["rate_model_median"]).all()
    assert result["rate_model_hdi95_upper"].ge(result["rate_model_median"]).all()

