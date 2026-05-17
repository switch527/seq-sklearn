"""F4 predict-time schema fingerprint contract (Phase 6a, N1).

``_predict_raw`` rejects an ``X`` whose schema (sorted column names +
dtypes + config + task_type) differs from the fit-time fingerprint with
:class:`DataContractError`, BEFORE running the transformer, rather than
silently producing predictions against a drifted schema.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.errors import DataContractError
from tests._test_models._dummy_estimator import _DummySequenceClassifier


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(16):
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
        max_epochs=1,
        batch_size=16,
        val_fraction=0.2,
        cal_fraction=0.0,
        precision="32-true",
        verbose=False,
        seed=42,
    )


def test_predict_schema_fingerprint_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _estimator().fit(x, y)

    # An extra column changes sorted(columns)+dtypes, so the predict-time
    # fingerprint diverges from fit even though every declared column is
    # still present and well-typed.
    drifted = x.assign(extra=1.0)
    with pytest.raises(DataContractError, match="feature_schema_fingerprint mismatch"):
        est.predict_proba(drifted)
