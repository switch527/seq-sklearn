"""Same-process determinism (Phase 6a, N1).

Two ``_DummySequenceClassifier(seed=42)`` fits in one process with no
intervening global-state change produce bit-identical ``predict_proba``.
Exercises the in-process determinism contract independent of the
save / load path; marked ``determinism`` so the integration conftest
snapshots / restores the four N4 globals around it.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from tests._test_models._dummy_estimator import _DummySequenceClassifier


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(20):
        label = e % 2
        for t in range(6):
            rows.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "tr": float(label) + 0.2 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y)


def _estimator() -> _DummySequenceClassifier:
    return _DummySequenceClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
            time_varying_real_cols=("tr",), lookback=3, min_periods=1, min_periods_predict=1
        ),
        scheduler=SchedulerParams(name="constant", warmup_steps=0),
        max_epochs=2,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    )


@pytest.mark.determinism
def test_two_fits_same_seed_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    a = _estimator().fit(x, y).predict_proba(x)
    b = _estimator().fit(x, y).predict_proba(x)
    assert np.array_equal(a, b)
