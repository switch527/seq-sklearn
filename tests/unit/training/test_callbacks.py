"""Callback unit tests (per architecture A7 / requirements F9 / N4).

Each callback is exercised in isolation. The Phase 4b
``_LightningModule`` does not exist yet, so a minimal real
``pl.LightningModule`` assembled from the Phase 4a dummy building
blocks plays the ``pl_module`` role; the callbacks never call into the
module (they read only ``trainer`` / ``outputs`` / ``checkpoint``), so
a ``MagicMock`` trainer is sufficient. NaN Variant B (Inf into model
weights) is deferred to Phase 7 per the implementation plan; this file
covers Variant A only.
"""

import logging
import random
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule

from seq_sklearn.errors import TrainingError
from seq_sklearn.logging import Event
from seq_sklearn.training.callbacks import (
    EventEmitter,
    GradScalerWatchdog,
    NaNLossGuard,
    RngStateCallback,
)
from tests._test_models._dummy_modules import (
    _DummyBackbone,
    _DummyHead,
    _LossReturningScalar,
)


class _ScaffoldModule(LightningModule):
    """Minimal real LightningModule built from the Phase 4a primitives."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _DummyBackbone()
        self.head = _DummyHead()
        self.loss = _LossReturningScalar()


@pytest.fixture
def module() -> _ScaffoldModule:
    return _ScaffoldModule()


# --- NaNLossGuard (N1 Variant A) ---------------------------------------


def test_nan_guard_three_consecutive_raise_with_batch_idx(
    module: _ScaffoldModule, caplog: pytest.LogCaptureFixture
) -> None:
    guard = NaNLossGuard()
    trainer = MagicMock()
    nan = torch.tensor(float("nan"))

    with caplog.at_level(logging.INFO, logger="seq_sklearn.training"):
        guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=0)
        guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=1)
        with pytest.raises(TrainingError, match=r"3 consecutive NaN"):
            guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=2)

    skip_records = [r for r in caplog.records if r.event == Event.TRAIN_NAN_STEP_SKIPPED]
    assert skip_records
    abort = [r for r in skip_records if r.payload.get("aborting")]
    assert len(abort) == 1
    assert abort[0].payload["batch_idx"] == 2
    assert abort[0].levelno == logging.ERROR


def test_nan_guard_non_nan_resets_counter(module: _ScaffoldModule) -> None:
    guard = NaNLossGuard()
    trainer = MagicMock()
    nan = torch.tensor(float("nan"))
    good = torch.tensor(0.5)

    guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=0)
    guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=1)
    guard.on_train_batch_end(trainer, module, good, {}, batch_idx=2)
    # Counter reset; two more NaNs must not trip the limit.
    guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=3)
    guard.on_train_batch_end(trainer, module, nan, {}, batch_idx=4)


def test_nan_guard_non_tensor_outputs_is_treated_as_non_nan(
    module: _ScaffoldModule,
) -> None:
    """A Mapping / None `outputs` resets the counter (no NaN to read)."""
    guard = NaNLossGuard()
    trainer = MagicMock()
    guard.on_train_batch_end(trainer, module, {"loss": 1.0}, {}, batch_idx=0)
    guard.on_train_batch_end(trainer, module, None, {}, batch_idx=1)


# --- GradScalerWatchdog ------------------------------------------------


def test_grad_scaler_watchdog_no_op_when_plugin_has_no_scaler(
    module: _ScaffoldModule,
) -> None:
    watchdog = GradScalerWatchdog()
    trainer = MagicMock()
    trainer.precision_plugin = object()  # no `scaler` attribute
    watchdog.on_train_batch_end(trainer, module, torch.tensor(0.1), {}, batch_idx=0)


def test_grad_scaler_watchdog_no_op_when_scaler_is_none(
    module: _ScaffoldModule,
) -> None:
    watchdog = GradScalerWatchdog()
    trainer = MagicMock()
    plugin = MagicMock()
    plugin.scaler = None
    trainer.precision_plugin = plugin
    watchdog.on_train_batch_end(trainer, module, torch.tensor(0.1), {}, batch_idx=0)


def test_grad_scaler_watchdog_mock_scaler_decrease(
    module: _ScaffoldModule, caplog: pytest.LogCaptureFixture
) -> None:
    watchdog = GradScalerWatchdog()
    trainer = MagicMock()
    plugin = MagicMock()
    scaler = MagicMock()
    plugin.scaler = scaler
    trainer.precision_plugin = plugin

    scales = iter([8.0, 4.0, 2.0, 1.0])
    scaler.get_scale.side_effect = lambda: next(scales)

    with caplog.at_level(logging.ERROR, logger="seq_sklearn.training"):
        watchdog.on_train_batch_end(trainer, module, None, {}, batch_idx=0)
        watchdog.on_train_batch_end(trainer, module, None, {}, batch_idx=1)
        watchdog.on_train_batch_end(trainer, module, None, {}, batch_idx=2)
        with pytest.raises(TrainingError, match=r"mixed-precision training diverged"):
            watchdog.on_train_batch_end(trainer, module, None, {}, batch_idx=3)

    diverged = [r for r in caplog.records if r.event == Event.TRAIN_MIXED_PRECISION_DIVERGED]
    assert len(diverged) == 1
    assert diverged[0].levelno == logging.ERROR
    assert diverged[0].payload["batch_idx"] == 3


def test_grad_scaler_watchdog_increase_resets(module: _ScaffoldModule) -> None:
    watchdog = GradScalerWatchdog()
    trainer = MagicMock()
    plugin = MagicMock()
    scaler = MagicMock()
    plugin.scaler = scaler
    trainer.precision_plugin = plugin

    scales = iter([8.0, 4.0, 16.0, 8.0, 4.0])
    scaler.get_scale.side_effect = lambda: next(scales)
    for i in range(5):
        watchdog.on_train_batch_end(trainer, module, None, {}, batch_idx=i)


# --- EventEmitter ------------------------------------------------------


def test_event_emitter_record_carries_event_and_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    emitter = EventEmitter()
    with caplog.at_level(logging.INFO, logger="seq_sklearn.training"):
        emitter.emit(Event.TRAIN_EPOCH, epoch=3, train_loss=0.25)

    record = next(r for r in caplog.records if r.event == Event.TRAIN_EPOCH)
    assert record.event == Event.TRAIN_EPOCH.value
    assert record.payload == {"epoch": 3, "train_loss": 0.25}


def test_event_emitter_respects_level(caplog: pytest.LogCaptureFixture) -> None:
    emitter = EventEmitter()
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.training"):
        emitter.emit(Event.TRAIN_EPOCH, level=logging.ERROR, epoch=1)
    record = next(r for r in caplog.records if r.event == Event.TRAIN_EPOCH)
    assert record.levelno == logging.ERROR


# --- RngStateCallback --------------------------------------------------


def _rng_fingerprint() -> tuple[Any, Any, Any]:
    return (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )


def test_rng_state_round_trip_restores_bit_exact() -> None:
    cb = RngStateCallback()
    trainer = MagicMock()
    pl_module = MagicMock()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    checkpoint: dict[str, Any] = {}
    cb.on_save_checkpoint(trainer, pl_module, checkpoint)
    expected = _rng_fingerprint()

    # Perturb every stream so a failed restore is observable.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    cb.on_load_checkpoint(trainer, pl_module, checkpoint)
    assert _rng_fingerprint() == expected


def test_rng_state_load_without_key_is_noop() -> None:
    cb = RngStateCallback()
    cb.on_load_checkpoint(MagicMock(), MagicMock(), {})


def test_rng_state_save_captures_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cb = RngStateCallback()
    sentinel = ["cuda-rng-state"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: sentinel)
    checkpoint: dict[str, Any] = {}
    cb.on_save_checkpoint(MagicMock(), MagicMock(), checkpoint)
    assert checkpoint["seq_sklearn_rng"]["cuda"] == sentinel


def test_rng_state_load_restores_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cb = RngStateCallback()
    captured: dict[str, Any] = {}

    def _record(state: Any) -> None:
        captured["state"] = state

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["s"])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", _record)
    checkpoint: dict[str, Any] = {}
    cb.on_save_checkpoint(MagicMock(), MagicMock(), checkpoint)
    cb.on_load_checkpoint(MagicMock(), MagicMock(), checkpoint)
    assert captured["state"] == ["s"]


def test_rng_state_load_skips_cuda_restore_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cuda state present in checkpoint but no CUDA at load time: skipped."""
    cb = RngStateCallback()
    # Save with CUDA available so checkpoint["...cuda"] is non-None.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["s"])
    checkpoint: dict[str, Any] = {}
    cb.on_save_checkpoint(MagicMock(), MagicMock(), checkpoint)
    # Load with CUDA unavailable: the second `and` operand is False.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cb.on_load_checkpoint(MagicMock(), MagicMock(), checkpoint)
