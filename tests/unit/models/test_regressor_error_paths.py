"""The three N1 ``predict_quantiles`` error paths (Phase 6a).

These originate on ``BaseSequenceRegressor`` (not on TFT), so the
concrete instance under test is the smoke ``_DummySequenceRegressor``:

* point-mode regressor -> ``PredictionError`` naming ``predict``;
* pre-fit -> ``NotFittedError`` (catchable as the seq-sklearn AND the
  sklearn exception);
* a requested quantile absent from the fit-time vector -> ``ValueError``
  whose message lists that vector (v1 does not interpolate).
"""

import numpy as np
import pandas as pd
import pytest
import sklearn.exceptions
import torch

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.errors import NotFittedError, PredictionError
from tests._test_models._dummy_estimator import _DummySequenceRegressor


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel(n_entities: int = 16, n_steps: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
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
        "max_epochs": 1,
        "batch_size": 16,
        "val_fraction": 0.2,
        "cal_fraction": 0.0,
        "precision": "32-true",
        "verbose": False,
        "seed": 42,
    }
    params.update(kw)
    return _DummySequenceRegressor(**params)  # type: ignore[arg-type]


def test_predict_quantiles_before_fit_raises_notfitted() -> None:
    reg = _reg("regression_quantile", quantiles=(0.1, 0.5, 0.9))
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "tr": [0.0]})
    with pytest.raises(NotFittedError):
        reg.predict_quantiles(x)
    # Catchable as the sklearn exception too (dual inheritance).
    reg2 = _reg("regression_quantile", quantiles=(0.1, 0.5, 0.9))
    with pytest.raises(sklearn.exceptions.NotFittedError):
        reg2.predict_quantiles(x)


def test_predict_quantiles_on_point_mode_raises_prediction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    reg = _reg("regression_point").fit(x, y)
    with pytest.raises(PredictionError, match="predict"):
        reg.predict_quantiles(x)


def test_predict_quantiles_off_vector_raises_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    reg = _reg("regression_quantile", quantiles=(0.1, 0.5, 0.9)).fit(x, y)
    with pytest.raises(ValueError, match=r"0\.1.*0\.5.*0\.9"):
        reg.predict_quantiles(x, quantiles=[0.99])
