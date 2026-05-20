"""A6.1 recurrent-skeleton contract-lock test (Phase 6b).

An inline no-op concrete subclass fills the three abstract recurrent
hooks plus the Phase-6a ``_build_backbone_head`` and composes with the
``BaseSequenceClassifier`` shell. Running fit / predict / save / load
on a tiny synthetic panel proves the recurrent surface does not break
the v1 estimator contract, the whole point of shipping the skeleton in
v1 before any concrete recurrent model exists.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from seq_sklearn.config.adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models._heads import ClassificationHead
from seq_sklearn.models.recurrent._base import RecurrentSequenceEstimator
from tests._test_models._dummy_estimator import _HIDDEN, _backbone


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


class _NoOpRecurrentClassifier(RecurrentSequenceEstimator, BaseSequenceClassifier):
    """Smallest thing that satisfies both abstract surfaces.

    The recurrent hooks are intentionally trivial: the Phase-6a shell
    never calls them in v1 (they feed a v3 recurrent backbone), so the
    real backbone here is the same pooled-linear stand-in the Phase-6a
    dummy uses. This isolates "does the recurrent ABC compose with the
    shell" from any recurrent math.
    """

    def _init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(batch_size, _HIDDEN, device=device),)

    def _readout(self, hidden_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return hidden_seq[:, -1, :]

    def _bptt_window(self) -> int | None:
        return None

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        return (
            _backbone(transformer, _HIDDEN),
            ClassificationHead(_HIDDEN, self._head_out_dim()),
        )


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


def _estimator() -> _NoOpRecurrentClassifier:
    return _NoOpRecurrentClassifier(
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


def test_recurrent_skeleton_is_abstract() -> None:
    # The bare ABC cannot be instantiated: _build_backbone_head and the
    # three recurrent hooks are all abstract until a concrete fills them.
    with pytest.raises(TypeError):
        RecurrentSequenceEstimator(task_type="binary")  # type: ignore[abstract,call-arg]


def test_noop_recurrent_composes_with_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _estimator().fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (96, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = est.predict(x)
    assert set(np.unique(preds)).issubset({0, 1})


def test_noop_recurrent_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _estimator().fit(x, y)
    before = est.predict_proba(x)
    est.save(tmp_path)  # type: ignore[arg-type]
    reloaded = _NoOpRecurrentClassifier.load(tmp_path)  # type: ignore[arg-type]
    assert np.array_equal(reloaded.predict_proba(x), before)
