"""Short-entity predict contract (Phase 6a, N1).

Entities below ``min_periods_predict`` get NaN-filled `predict_proba`
rows of the correct shape (never zero / scalar), and exactly one
aggregated ``data.min_periods_predict_breach`` WARNING fires per
``predict`` call regardless of how many entities are below the floor
(requirements F NaN-in-output / short-entity warning + shape).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.logging import Event
from tests._test_models._dummy_estimator import (
    _DummySequenceClassifier,
    _DummySequenceRegressor,
)


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel(spec: list[tuple[int, int]]) -> tuple[pd.DataFrame, np.ndarray]:
    """spec = [(entity_id, n_rows), ...]."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for eid, n in spec:
        label = eid % 2
        for t in range(n):
            rows.append(
                {
                    "id": eid,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "tr": float(label) + 0.2 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y)


def test_short_entity_predict_proba_nan_rows_and_single_breach_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _force_cpu(monkeypatch)
    # min_periods_predict=3: fit on all-above-floor entities.
    x_fit, y_fit = _panel([(e, 6) for e in range(16)])
    est = _DummySequenceClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
            time_varying_real_cols=("tr",),
            lookback=3,
            min_periods=1,
            min_periods_predict=3,
        ),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        max_epochs=1,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(x_fit, y_fit)

    # Predict panel: three below-floor entities (1, 2, 2 rows) + one
    # above-floor (5 rows). Same columns/dtypes so the schema
    # fingerprint matches fit.
    x_pred, _ = _panel([(100, 1), (101, 2), (102, 2), (103, 5)])
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        proba = est.predict_proba(x_pred)

    # entity_id codes are emission order: 100 -> rows[0], 101 -> [1:3],
    # 102 -> [3:5], 103 -> [5:10]. Below-floor: codes 0,1,2 (1,2,2 rows).
    below = np.r_[0:5]
    above = np.r_[5:10]
    assert proba.shape == (10, 2)
    assert np.isnan(proba[below]).all()
    assert np.isfinite(proba[above]).all()
    assert np.allclose(proba[above].sum(axis=1), 1.0)

    breaches = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == Event.DATA_MIN_PERIODS_PREDICT_BREACH
    ]
    assert len(breaches) == 1
    assert breaches[0].payload["count"] == 3


def test_regressor_short_entity_predict_nan_rows_and_single_breach_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Regressor twin of the classifier contract: the _calibrated_matrix
    # below-floor NaN-fill must hold for BOTH predict (point) and
    # predict_quantiles, never zero / shape-corrupted, with one
    # aggregated breach log per call.
    _force_cpu(monkeypatch)
    x_fit, y_fit = _panel([(e, 6) for e in range(16)])
    est = _DummySequenceRegressor(
        task_type="regression_quantile",
        quantiles=(0.1, 0.5, 0.9),
        tabular_config=TabularConfigParams(
            time_varying_real_cols=("tr",),
            lookback=3,
            min_periods=1,
            min_periods_predict=3,
        ),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        max_epochs=1,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(x_fit, y_fit)

    x_pred, _ = _panel([(100, 1), (101, 2), (102, 2), (103, 5)])
    below = np.r_[0:5]
    above = np.r_[5:10]
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        point = est.predict(x_pred)
        quant = est.predict_quantiles(x_pred)

    assert point.shape == (10,)
    assert np.isnan(point[below]).all()
    assert np.isfinite(point[above]).all()
    assert quant.shape == (10, 3)
    assert np.isnan(quant[below]).all()
    assert np.isfinite(quant[above]).all()

    breaches = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == Event.DATA_MIN_PERIODS_PREDICT_BREACH
    ]
    # One aggregated breach per transform; predict + predict_quantiles
    # each transform once.
    assert len(breaches) == 2
    assert all(b.payload["count"] == 3 for b in breaches)
