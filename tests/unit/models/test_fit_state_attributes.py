"""F1.1 fit-state attribute contract (Phase 6a, N1).

After ``fit`` every F1.1 attribute is present with the documented
shape / dtype; ``decision_threshold_`` / ``quantiles_`` are ABSENT on
the estimators that do not define them (A2); and ``config_`` is the
frozen pydantic model (mutation raises).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from tests._test_models._dummy_estimator import (
    _DummySequenceClassifier,
    _DummySequenceRegressor,
)


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel(label_fn: Callable[[int], object]) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    y: list[object] = []
    for e in range(16):
        lab = label_fn(e)
        for t in range(6):
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "tr": float(hash(lab) % 3) + 0.2 * rng.normal(),
                }
            )
            y.append(lab)
    return pd.DataFrame(rows), np.asarray(y)


def _tab() -> TabularConfigParams:
    return TabularConfigParams(
        time_varying_real_cols=("tr",), lookback=3, min_periods=1, min_periods_predict=1
    )


_COMMON = {
    "scheduler": SchedulerParams(name="constant", warmup_steps=0),
    "max_epochs": 1,
    "batch_size": 16,
    "val_fraction": 0.2,
    "cal_fraction": 0.0,
    "precision": "32-true",
    "verbose": False,
    "seed": 42,
}


def test_classifier_fit_state_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel(lambda e: f"c{e % 2}")
    est = _DummySequenceClassifier(
        task_type="binary",
        tabular_config=_tab(),
        threshold_tuning=True,
        cal_fraction=0.2,
        **{k: v for k, v in _COMMON.items() if k != "cal_fraction"},
    ).fit(x, y)
    assert isinstance(est.classes_, np.ndarray)
    assert est.classes_.ndim == 1
    assert list(est.classes_) == ["c0", "c1"]  # LabelEncoder-sorted
    assert isinstance(est.n_features_in_, int)
    assert est.n_features_in_ == 1  # one declared column (tr)
    assert isinstance(est.feature_names_in_, np.ndarray)
    assert list(est.feature_names_in_) == ["tr"]
    assert est.n_outputs_ == 1
    assert isinstance(est.decision_threshold_, float)  # binary + tuning
    assert not hasattr(est, "quantiles_")


def test_classifier_decision_threshold_absent_without_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel(lambda e: e % 2)
    est = _DummySequenceClassifier(task_type="binary", tabular_config=_tab(), **_COMMON).fit(x, y)
    assert not hasattr(est, "decision_threshold_")  # A2: absent w/o tuning


def test_regressor_quantile_fit_state_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel(lambda e: float(e))
    est = _DummySequenceRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(),
        quantiles=(0.1, 0.5, 0.9),
        **_COMMON,
    ).fit(x, y)
    assert isinstance(est.quantiles_, np.ndarray)
    assert list(est.quantiles_) == [0.1, 0.5, 0.9]
    assert est.n_outputs_ == 1
    assert est.n_features_in_ == 1
    assert not hasattr(est, "classes_")
    assert not hasattr(est, "decision_threshold_")


def test_config_is_frozen_after_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel(lambda e: e % 2)
    est = _DummySequenceClassifier(task_type="binary", tabular_config=_tab(), **_COMMON).fit(x, y)
    with pytest.raises(ValidationError):
        est.config_.batch_size = 999  # frozen pydantic model
