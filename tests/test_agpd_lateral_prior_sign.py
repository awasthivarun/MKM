"""Keep the AgPd repulsive-CO prior separate from signed cross/OH interactions."""

from pathlib import Path
import yaml


def test_default_agpd_co_co_prior_is_nonnegative_without_changing_other_interactions():
    path = Path(__file__).resolve().parents[1] / "config/priors/agpd_basic.yaml"
    specs = yaml.safe_load(path.read_text(encoding="utf-8"))["lateral_interaction_priors"]
    assert specs["z_CO_CO_eV"]["lower"] == 0.0
    assert specs["z_CO_CO_eV"]["upper"] > 0.0
    assert specs["z_CO_OH_eV"]["lower"] < 0.0
    assert specs["z_OH_OH_eV"]["lower"] < 0.0
