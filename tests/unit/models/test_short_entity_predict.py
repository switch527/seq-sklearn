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
from seq_sklearn.errors import ConfigError
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


def _cal_panel() -> tuple[pd.DataFrame, np.ndarray]:
    # 6 above-floor entities (6 rows, both labels) + 2 below-floor
    # (2 rows < min_periods_predict=3 but >= min_periods=1, so they
    # survive fit and get a sentinel target in transform).
    return _panel([(e, 6) for e in range(6)] + [(100, 2), (101, 2)])


def _split_diagnostics(est: object, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(below-floor mask, raw cal_idx) for the recomputed split."""
    from seq_sklearn.data.splits import compute_three_way_split, window_time_index

    batch = est.transformer_.transform(x)  # type: ignore[attr-defined]
    eids = batch["entity_id"].cpu().numpy()
    _, _, cal_idx = compute_three_way_split(
        eids,
        window_time_index(eids),
        val_fraction=est.val_fraction,  # type: ignore[attr-defined]
        cal_fraction=est.cal_fraction,  # type: ignore[attr-defined]
        val_split_strategy=est.val_split_strategy,  # type: ignore[attr-defined]
        calibration_set_provided=False,
    )
    return est._below_floor_mask(batch), cal_idx  # type: ignore[attr-defined]


def test_calibration_fold_drops_below_floor_sentinel_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mutation-sensitive guard for the Round-2 `keep` filter: a
    # below-floor entity provably lands in the recomputed cal_idx, so
    # reverting `keep` to `cal_idx` would feed a -1 sentinel target into
    # the calibrator / threshold tuner. This test fails if that happens.
    _force_cpu(monkeypatch)
    x, y = _cal_panel()
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
        cal_fraction=0.5,
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(x, y)

    below, cal_idx = _split_diagnostics(est, x)
    # Non-vacuous preconditions: below-floor windows are genuinely in
    # cal_idx and the UNFILTERED fold would carry the -1 sentinel.
    assert below[cal_idx].any()
    batch = est.transformer_.transform(x)
    assert (batch["target"][cal_idx].numpy() == -1).any()

    raw, targets = est._calibration_fold(est.transformer_, x, None)
    t = targets.numpy()
    assert (t != -1).all()  # sentinel dropped
    assert np.isfinite(t).all()
    assert len(t) == int((~below[cal_idx]).sum())
    assert raw.shape[0] == len(t)


def test_calibration_fold_threshold_tuner_path_no_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The threshold-tuner consumer of _calibration_fold (independent of
    # the calibrator path): the tuned threshold is a valid float, which
    # is only true if the tuner fit on clean (non-sentinel) targets.
    _force_cpu(monkeypatch)
    x, y = _cal_panel()
    est = _DummySequenceClassifier(
        task_type="binary",
        threshold_tuning=True,
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
        cal_fraction=0.5,
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(x, y)

    below, cal_idx = _split_diagnostics(est, x)
    assert below[cal_idx].any()  # non-vacuous
    assert isinstance(est.decision_threshold_, float)
    assert 0.0 <= est.decision_threshold_ <= 1.0
    _, targets = est._calibration_fold(est.transformer_, x, None)
    assert (targets.numpy() != -1).all()


def test_calibration_fold_all_below_floor_raises_configerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Degenerate: a recomputed fold whose every window is below
    # min_periods_predict empties `keep`. The estimator raises a typed
    # ConfigError naming the cause, not a deep calibrator ValueError.
    # Fit on a normal panel, then drive _calibration_fold with an
    # all-short panel (every entity < min_periods_predict=3).
    _force_cpu(monkeypatch)
    x_fit, y_fit = _cal_panel()
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
        cal_fraction=0.5,
        precision="32-true",
        verbose=False,
        seed=42,
    ).fit(x_fit, y_fit)

    x_all_short, _ = _panel([(e, 2) for e in range(8)])
    with pytest.raises(ConfigError, match="calibration fold is empty after dropping"):
        est._calibration_fold(est.transformer_, x_all_short, None)
