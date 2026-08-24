import numpy as np
import pytest

from mkm.postprocessing.diagnostics import (
    build_balance_summary,
    build_noise_summary,
    build_parameter_contraction,
    build_physical_summary,
    flatten_posterior_samples,
    prior_statistics,
)


def test_flatten_posterior_samples_combines_chain_and_draw():
    values = np.arange(24).reshape(2, 3, 4)

    result = flatten_posterior_samples(values)

    assert result.shape == (6, 4)
    np.testing.assert_array_equal(result, values.reshape(6, 4))


def test_prior_statistics_rejects_unsupported_distribution():
    with pytest.raises(ValueError, match="Unsupported prior distribution"):
        prior_statistics({"distribution": "banana"})


def test_build_parameter_contraction_uses_material_model_profile():
    posterior = {
        "x": np.array(
            [
                [0.8, 0.9, 1.0],
                [1.0, 1.1, 1.2],
            ]
        )
    }
    config = {
        "prior_profiles": {
            "Ag10Pd90": {
                "TEST": {
                    "parameters": {
                        "x": {
                            "distribution": "normal",
                            "mu": 0.0,
                            "sigma": 2.0,
                        }
                    }
                }
            }
        }
    }
    result = build_parameter_contraction(
        posterior=posterior,
        config=config,
        material="Ag10Pd90",
        model_name="TEST",
    )

    assert result["parameter"].tolist() == ["x"]
    assert result.loc[0, "prior_sd"] == pytest.approx(2.0)
    assert result.loc[0, "posterior_median"] == pytest.approx(1.0)
    assert result.loc[0, "posterior_hdi95_lower"] <= result.loc[0, "posterior_median"]
    assert result.loc[0, "posterior_hdi95_upper"] >= result.loc[0, "posterior_median"]
    assert result.loc[0, "sd_ratio_posterior_over_prior"] > 0.0


def test_build_noise_summary_handles_material_and_setup_scales():
    posterior = {
        "sigma_ln_rate_material": np.full((2, 3, 1), 0.4),
        "sigma_ln_rate_setup_material": np.full((2, 3, 1), 0.08),
    }

    result = build_noise_summary(posterior)

    assert set(result["variable"]) == {
        "sigma_ln_rate_material",
        "sigma_ln_rate_setup_material",
    }

    material = result[result["variable"] == "sigma_ln_rate_material"].iloc[0]
    setup = result[result["variable"] == "sigma_ln_rate_setup_material"].iloc[0]

    assert material["median"] == pytest.approx(0.4)
    assert setup["median"] == pytest.approx(0.08)
    assert material["hdi95_lower"] == pytest.approx(0.4)
    assert material["hdi95_upper"] == pytest.approx(0.4)


def test_build_physical_summary_detects_out_of_bounds_values():
    posterior = {
        "theta_CO": np.array(
            [
                [[0.2, 0.5], [0.3, 1.01]],
                [[0.4, 0.6], [0.2, 0.7]],
            ]
        )
    }

    result = build_physical_summary(posterior)
    assert result["variable"].tolist() == ["theta_CO"]
    assert result.loc[0, "maximum"] == pytest.approx(1.01)
    assert result.loc[0, "fraction_above_one"] > 0.0
    assert {"median", "hdi95_lower", "hdi95_upper"}.issubset(result.columns)


def test_build_balance_summary_checks_sites_and_pathways():
    posterior = {
        "theta_CO": np.full((2, 3, 4), 0.5),
        "theta_OH_Pd": np.full((2, 3, 4), 0.2),
        "theta_empty_Pd": np.full((2, 3, 4), 0.3),
        "theta_OH_Ag": np.full((2, 3, 4), 0.25),
        "theta_empty_Ag": np.full((2, 3, 4), 0.75),
        "rate_fraction_BF": np.full((2, 3, 4), 0.6),
        "rate_fraction_LH": np.full((2, 3, 4), 0.4),
    }

    result = build_balance_summary(posterior)
    assert set(result["balance"]) == {
        "Pd",
        "Ag",
        "pathway_fraction_sum",
    }

    np.testing.assert_allclose(
        result["max_abs_error"].to_numpy(),
        0.0,
        rtol=0.0,
        atol=1e-15,
    )
