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