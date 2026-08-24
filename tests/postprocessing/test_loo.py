from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mkm.postprocessing.loo import compute_loo_diagnostics


def test_compute_loo_diagnostics_maps_pointwise_values(monkeypatch):
    observations = pd.DataFrame({"observation_id": [0, 1], "ln_rate": [-1.0, -2.0]})

    fake_loo = SimpleNamespace(
        elpd=-3.0,
        se=0.5,
        p=1.2,
        n_samples=100,
        n_data_points=2,
        good_k=0.7,
        warning=False,
        elpd_i=np.array([-1.0, -2.0]),
        pareto_k=np.array([0.2, 0.8]),
    )

    monkeypatch.setattr("mkm.postprocessing.loo.azs.loo", lambda *args, **kwargs: fake_loo)

    result = compute_loo_diagnostics(
        inference_data=object(),
        observations=observations,
        model_name="TEST",
    )

    assert result.summary.loc[0, "elpd"] == pytest.approx(-3.0)
    assert result.summary.loc[0, "max_pareto_k"] == pytest.approx(0.8)
    assert result.summary.loc[0, "n_pareto_k_above_good_k"] == 1
    assert result.pointwise["elpd_loo"].tolist() == pytest.approx([-1.0, -2.0])
    assert result.pointwise["pareto_k"].tolist() == pytest.approx([0.2, 0.8])