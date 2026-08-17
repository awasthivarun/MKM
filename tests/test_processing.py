import numpy as np

from _CO_Oxidation.context import build_context


def _make_condition(offset):
    E = np.array([0.10, 0.20, 0.30])
    rate = np.array([1.0, 2.0, 3.0]) + offset

    return {
        "truncated_E": E,
        "truncated_rate": rate,
        "truncated_rate_SD": np.full(3, 0.1),
        "truncated_log_rate": np.log(rate),
        "truncated_log_rate_SD": np.full(3, 0.05),
        "truncated_rate_matrix": np.vstack([rate, rate + 100.0]),
        "E": E,
        "rate": rate,
        "rate_SD": np.full(3, 0.1),
        "log_rate": np.log(rate),
        "log_rate_SD": np.full(3, 0.05),
        "alpha": np.full(3, 0.5),
        "alpha_SD": np.full(3, 0.02),
        "E_OH": E,
        "delta_OH": np.full(3, 1.0),
        "delta_OH_SD": np.full(3, 0.1),
    }


def test_process_experimental_data_ordering_and_shapes():
    ctx = build_context("base")

    C_list = np.array([0.25, 1.0])
    P_list = np.array([0.001, 0.01])

    experiments = {
        (0.25, 0.001): _make_condition(0),
        (0.25, 0.01): _make_condition(10),
        (1.0, 0.001): _make_condition(20),
        (1.0, 0.01): _make_condition(30),
    }

    ctx.process_experimental_data(experiments, C_list, P_list)

    # Current canonical ordering:
    # concentration -> pressure -> potential
    expected_E = np.tile([0.10, 0.20, 0.30], 4)

    expected_C = np.repeat(
        [0.25, 0.25, 1.0, 1.0],
        3,
    )

    expected_P = np.repeat(
        [0.001, 0.01, 0.001, 0.01],
        3,
    )

    np.testing.assert_allclose(ctx.E_in, expected_E)
    np.testing.assert_allclose(ctx.C_KOH_in, expected_C)
    np.testing.assert_allclose(ctx.P_CO_in, expected_P)

    # 2 replicate trials x 12 flattened experimental points
    assert ctx.rate_obs_matrix.shape == (2, 12)

    expected_rate_matrix = np.concatenate(
        [
            experiments[(0.25, 0.001)]["truncated_rate_matrix"],
            experiments[(0.25, 0.01)]["truncated_rate_matrix"],
            experiments[(1.0, 0.001)]["truncated_rate_matrix"],
            experiments[(1.0, 0.01)]["truncated_rate_matrix"],
        ],
        axis=1,
    )

    np.testing.assert_allclose(
        ctx.rate_obs_matrix,
        expected_rate_matrix,
    )

import arviz as az

from _CO_Oxidation.config import R, T, F


def test_post_sampling_observables_recover_analytic_orders():
    ctx = build_context("base")

    C_list = np.array([0.25, 1.0])
    P_list = np.array([0.001, 0.01])
    E = np.array([0.10, 0.20, 0.30, 0.40])

    # Analytic model:
    #
    # log(r) = a*log(C) + b*log(P) + c*E + d
    #
    # Therefore:
    # delta_OH = a
    # delta_CO = b
    # alpha = (R*T/F)*c
    a = 0.70
    b = -0.30
    c = 8.0
    d = 2.0

    experiments = {}
    expected_log_rate_blocks = []

    for C in C_list:
        for P in P_list:
            log_rate = (
                a * np.log(C)
                + b * np.log(P)
                + c * E
                + d
            )
            rate = np.exp(log_rate)

            experiments[(C, P)] = {
                "truncated_E": E,
                "truncated_rate": rate,
                "truncated_rate_SD": np.full_like(E, 0.1),
                "truncated_log_rate": log_rate,
                "truncated_log_rate_SD": np.full_like(E, 0.05),
                "truncated_rate_matrix": np.vstack([rate, rate]),
                "E": E,
                "rate": rate,
                "rate_SD": np.full_like(E, 0.1),
                "log_rate": log_rate,
                "log_rate_SD": np.full_like(E, 0.05),
                "alpha": np.full_like(E, (R * T / F) * c),
                "alpha_SD": np.full_like(E, 0.01),
                "E_OH": E,
                "delta_OH": np.full_like(E, a),
                "delta_OH_SD": np.full_like(E, 0.01),
                "E_CO": E,
                "delta_CO": np.full_like(E, b),
                "delta_CO_SD": np.full_like(E, 0.01),
            }

            expected_log_rate_blocks.append(log_rate)

    ctx.process_experimental_data(
        experiments,
        C_list,
        P_list,
    )

    log_rate_flat = np.concatenate(expected_log_rate_blocks)

    # Two chains and three draws, all deliberately identical.
    posterior_log_rate = np.broadcast_to(
        log_rate_flat,
        (2, 3, log_rate_flat.size),
    ).copy()

    trace = az.from_dict(
        posterior={
            "log_rate": posterior_log_rate,
        }
    )

    trace = ctx.add_post_sampling_observables(trace)

    expected_alpha = (R * T / F) * c

    np.testing.assert_allclose(
        trace.posterior["alpha"].values,
        expected_alpha,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        trace.posterior["delta_OH"].values,
        a,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        trace.posterior["delta_CO"].values,
        b,
        rtol=1e-12,
        atol=1e-12,
    )

    # Experimental and model log-rates are identical,
    # so residuals must be zero.
    np.testing.assert_allclose(
        trace.posterior["log_residual"].values,
        0.0,
        atol=1e-12,
    )