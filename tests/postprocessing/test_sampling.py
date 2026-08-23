import numpy as np
import xarray as xr

from mkm.postprocessing.sampling import build_sampling_datatree, sampling_parameter_names


def test_sampling_parameter_names_include_scalar_noise():
    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), np.ones((2, 3))),
            "b": (("chain", "draw"), np.ones((2, 3))),
            "sigma_ln_rate_material": (("chain", "draw", "material"), np.ones((2, 3, 1))),
            "not_scalar": (("chain", "draw", "point"), np.ones((2, 3, 4))),
        }
    )

    names = sampling_parameter_names(posterior, {"a": {}, "b": {}})

    assert names == ["a", "b", "sigma_ln_rate_material"]


def test_build_sampling_datatree_scalarizes_singleton_dimensions():
    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), np.ones((2, 3))),
            "sigma_ln_rate_material": (("chain", "draw", "material"), np.ones((2, 3, 1))),
        }
    )
    sample_stats = xr.Dataset(
        {
            "energy": (("chain", "draw"), np.ones((2, 3))),
            "depth": (("chain", "draw"), np.full((2, 3), 5)),
            "maxdepth_reached": (("chain", "draw"), np.zeros((2, 3), dtype=bool)),
            "logp": (("chain", "draw"), np.zeros((2, 3))),
        }
    )
    data = xr.DataTree.from_dict({"/posterior": posterior, "/sample_stats": sample_stats})

    result = build_sampling_datatree(data, ["a", "sigma_ln_rate_material"])

    assert result.posterior["a"].dims == ("chain", "draw")
    assert result.posterior["sigma_ln_rate_material"].dims == ("chain", "draw")
    assert "sample_stats" in result.children

    sample_stats_result = result["sample_stats"]

    assert "tree_depth" in sample_stats_result
    assert "reached_max_treedepth" in sample_stats_result
    assert "lp" in sample_stats_result

    assert "depth" not in sample_stats_result
    assert "maxdepth_reached" not in sample_stats_result
    assert "logp" not in sample_stats_result

