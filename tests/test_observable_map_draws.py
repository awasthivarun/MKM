import numpy as np
import pandas as pd
import pytest

from mkm.observable_maps import (
    LinearObservableMap,
    evaluate_linear_observable_map,
    evaluate_linear_observable_map_draws,
)


def _make_map():

    outputs = pd.DataFrame(
        {
            "observable_id": [0, 1],
            "label": ["first", "second"],
        }
    )

    terms = pd.DataFrame(
        {
            "observable_id": [0, 0, 1, 1, 1],
            "model_point_id": [0, 1, 1, 2, 3],
            "coefficient": [-1.0, 1.0, 0.5, -1.0, 0.5],
        }
    )

    return LinearObservableMap(outputs=outputs, terms=terms)


def test_draw_evaluator_matches_single_vector_evaluator():

    observable_map = _make_map()

    log_rate = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ]
    )

    result = evaluate_linear_observable_map_draws(log_rate, observable_map)

    expected = np.empty((2, 3, 2))

    for chain in range(2):
        for draw in range(3):
            expected[chain, draw] = evaluate_linear_observable_map(log_rate[chain, draw], observable_map)

    np.testing.assert_allclose(result, expected)
    assert result.shape == (2, 3, 2)


def test_draw_evaluator_rejects_invalid_model_point():

    observable_map = _make_map()
    observable_map.terms.loc[0, "model_point_id"] = 10

    with pytest.raises(ValueError, match="invalid model-point ID"):
        evaluate_linear_observable_map_draws(np.ones((2, 3, 4)), observable_map)