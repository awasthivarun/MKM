"""PtRu composition plots reproduce the same bounded-slope transform used by the model."""

import numpy as np
import xarray as xr

from mkm.models.ptru import ptru_parameter_names
from mkm.postprocessing.ptru_composition import build_ptru_composition_parameter_trends
from mkm.project_paths import ProjectPaths
from mkm.workflows.ptru_fit import load_ptru_config, resolve_ptru_fit_specification


def test_linear_xru_trends_reconstruct_normalized_beta_slope():
    paths = ProjectPaths.discover(__file__)
    config, priors = load_ptru_config(paths)
    spec = resolve_ptru_fit_specification(
        config, "basic", "CO_ER_LH", all_materials=True, parameterization="linear_xRu",
    )
    variables = {}
    for name in ptru_parameter_names(spec.model_name):
        variables[name] = (("chain", "draw"), np.full((2, 4), 0.4))
        variables[f"{name}_xRu_slope"] = (("chain", "draw"), np.full((2, 4), 0.2))
    posterior = xr.Dataset(variables)
    trends = build_ptru_composition_parameter_trends(
        type("Posterior", (), {"posterior": posterior})(), config, priors, spec, n_grid=5,
    )
    beta = trends.loc[trends["parameter"] == "beta_2_ER"]
    values = beta.loc[np.isclose(beta["xRu"], 0.5)]
    assert len(values) == 1
    lower = float(priors["prior_profiles"]["basic"]["parameters"]["beta_2_ER"]["lower"])
    upper = float(priors["prior_profiles"]["basic"]["parameters"]["beta_2_ER"]["upper"])
    x_reference = float(config["parameterizations"]["linear_xRu"]["x_reference"])
    margin = min(0.4 - lower, upper - 0.4)
    expected = 0.4 + (0.2 * margin / max(abs(x_reference), abs(0.5 - x_reference))) * (0.5 - x_reference)
    np.testing.assert_allclose(values["median"], expected)
