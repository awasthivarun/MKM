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


def test_sampling_parameter_names_leave_multimaterial_noise_out_of_plot_names():
    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), np.ones((2, 3))),
            "sigma_ln_rate_material": (("chain", "draw", "material"), np.ones((2, 3, 2))),
        },
        coords={"material": ["Ag10Pd90", "Pd100"]},
    )

    names = sampling_parameter_names(posterior, {"a": {}})

    assert names == ["a"]


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


def test_build_sampling_datatree_expands_multimaterial_noise_for_diagnostics():
    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), np.ones((2, 3))),
            "sigma_ln_rate_material": (("chain", "draw", "material"), np.ones((2, 3, 2))),
            "sigma_ln_rate_setup_material": (("chain", "draw", "material"), np.ones((2, 3, 2))),
            "z_ln_rate_setup": (("chain", "draw", "setup"), np.ones((2, 3, 3))),
        },
        coords={
            "material": ["Ag10Pd90", "Pd100"],
            "setup": ["Ag10_A", "Ag10_B", "Pd_A"],
        },
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

    result = build_sampling_datatree(data, ["a"], include_vector_noise=True)
    posterior_result = result["posterior"].to_dataset()

    assert set(posterior_result.data_vars) == {
        "a",
        "sigma_ln_rate_material[material=Ag10Pd90]",
        "sigma_ln_rate_material[material=Pd100]",
        "sigma_ln_rate_setup_material[material=Ag10Pd90]",
        "sigma_ln_rate_setup_material[material=Pd100]",
        "z_ln_rate_setup[setup=Ag10_A]",
        "z_ln_rate_setup[setup=Ag10_B]",
        "z_ln_rate_setup[setup=Pd_A]",
    }
    assert posterior_result["sigma_ln_rate_material[material=Ag10Pd90]"].dims == ("chain", "draw")
    assert posterior_result["sigma_ln_rate_material[material=Pd100]"].dims == ("chain", "draw")
    assert posterior_result["z_ln_rate_setup[setup=Ag10_A]"].dims == ("chain", "draw")
