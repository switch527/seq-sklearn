"""Regressor-shell e2e (Phase 6a): point + quantile predict, A17 round trip.

Covers ``BaseSequenceRegressor.predict`` (point and median-of-quantiles),
``predict_quantiles`` (full and ordered-subset), the conformal calibrator
dispatch, and the ``quantiles_`` fit-state through save / load.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.errors import TrainingError
from tests._test_models._dummy_estimator import _DummySequenceRegressor


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel(n_entities: int = 20, n_steps: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(5)
    rows: list[dict[str, object]] = []
    y: list[float] = []
    for e in range(n_entities):
        for t in range(n_steps):
            v = rng.normal()
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "tr": v,
                }
            )
            y.append(float(v))
    return pd.DataFrame(rows), np.asarray(y, dtype=float)


def _reg(task: str, **kw: object) -> _DummySequenceRegressor:
    params: dict[str, object] = {
        "task_type": task,
        "tabular_config": TabularConfigParams(
            time_varying_real_cols=("tr",), lookback=3, min_periods=1, min_periods_predict=1
        ),
        "scheduler": SchedulerParams(name="constant", warmup_steps=0),
        "max_epochs": 2,
        "batch_size": 16,
        "val_fraction": 0.2,
        "cal_fraction": 0.0,
        "precision": "32-true",
        "verbose": False,
        "seed": 42,
    }
    params.update(kw)
    return _DummySequenceRegressor(**params)  # type: ignore[arg-type]


def test_point_regressor_predict_and_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _reg("regression_point").fit(x, y)
    pred = est.predict(x)
    assert pred.ndim == 1
    assert np.isfinite(pred).all()
    assert not hasattr(est, "quantiles_")  # A2: point regressor absent
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceRegressor.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict(x), pred)


def test_conformal_on_undertrained_quantiles_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The trivial dummy's three raw quantile outputs are not monotone;
    # ConformalCalibrator must reject the crossing fold (F9), and the
    # TrainingError propagates out of estimator.fit.
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _reg(
        "regression_quantile",
        quantiles=(0.1, 0.5, 0.9),
        calibration_strategy="conformal",
        cal_fraction=0.2,
    )
    with pytest.raises(TrainingError, match="non-monotone"):
        est.fit(x, y)


def test_quantile_regressor_isotonic_quantile_predict_quantiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _reg(
        "regression_quantile",
        quantiles=(0.1, 0.5, 0.9),
        calibration_strategy="isotonic_quantile",
        cal_fraction=0.2,
    ).fit(x, y)
    assert list(est.quantiles_) == [0.1, 0.5, 0.9]
    full = est.predict_quantiles(x)
    assert full.shape[1] == 3
    subset = est.predict_quantiles(x, quantiles=[0.9, 0.1])
    assert subset.shape[1] == 2
    assert np.array_equal(subset[:, 0], full[:, 2])
    assert np.array_equal(subset[:, 1], full[:, 0])
    point = est.predict(x)  # median (0.5) column
    assert np.array_equal(point, full[:, 1])
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _DummySequenceRegressor.load(tmp_path)  # type: ignore[arg-type]
    assert list(reloaded.quantiles_) == [0.1, 0.5, 0.9]
    assert np.array_equal(reloaded.predict_quantiles(x), full)
