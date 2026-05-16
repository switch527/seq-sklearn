"""LightningModule hook-surface tests (per architecture A7 / A15).

The ``make_test_module`` fixture composes the Phase 4a ``_Dummy*``
building blocks (no Estimator, no real ``pl.Trainer``); hooks that read
``self.trainer.*`` get a ``MagicMock`` stubbed onto ``module._trainer``
per the A7 fixture pattern. The three named tests pin the A15 delegation
loop and the A7 deferred-prune ordering; the remaining tests close the
100% line / branch gate on the module's forward / optimizer / no-op
surface.
"""

import logging
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import optuna
import pytest
import torch
from torch import Tensor

from seq_sklearn.logging import Event
from seq_sklearn.models._backbone import BackboneOutput
from seq_sklearn.training._lightning_module import _LightningModule
from tests._test_models._dummy_modules import (
    _DummyBackbone,
    _DummyHead,
    _LossReturningScalar,
)


class _EntropyBackbone(_DummyBackbone):
    """``_DummyBackbone`` whose ``compute_training_metrics`` emits one event.

    Overriding the metric method alone (no LightningModule change) is the
    A15 v1 -> v3 cross-family abstraction the delegation-loop test pins.
    """

    def compute_training_metrics(self, output: BackboneOutput) -> dict[str, object]:
        del output
        return {
            "train.var_selection_entropy": {
                "static_entropy": 1.0,
                "temporal_entropy": 0.5,
            }
        }


class _TwoEventBackbone(_DummyBackbone):
    """``_DummyBackbone`` whose ``compute_training_metrics`` emits TWO events.

    Pins the A15 delegation loop against a single-emit / break-after-first
    mutant: a loop that stops after the first entry would drop the second
    payload and fail this test.
    """

    def compute_training_metrics(self, output: BackboneOutput) -> dict[str, object]:
        del output
        return {
            "train.var_selection_entropy": {
                "static_entropy": 1.0,
                "temporal_entropy": 0.5,
            },
            "train.hidden_norm": {"mean_norm": 2.0},
        }


def make_test_module(
    loss: torch.nn.Module | None = None,
    optuna_trial: optuna.trial.BaseTrial | None = None,
    backbone: _DummyBackbone | None = None,
    scheduler_factory: Callable[[Any], dict[str, object]] | None = None,
) -> _LightningModule:
    """Build an isolated module from the Phase 4a primitives (A7 fixture)."""
    mod = _LightningModule(
        backbone=backbone or _DummyBackbone(),
        head=_DummyHead(),
        loss=loss or _LossReturningScalar(),
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler_factory=scheduler_factory,
        optuna_trial=optuna_trial,
    )
    mock_trainer = MagicMock()
    mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
    mock_trainer.current_epoch = 0
    mod._trainer = mock_trainer
    # No real Lightning loop here; self.log would have no results sink.
    # Tests asserting log kwargs override this with a capturing stub.
    mod.log = MagicMock()  # type: ignore[method-assign]
    return mod


def _batch() -> dict[str, Tensor]:
    return {
        "features": torch.randn(2, 4),
        "target": torch.zeros(2, 1),
    }


# --- The three named A7 / A15 deliverable tests ------------------------


def test_on_train_epoch_end_skips_entropy_when_no_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = make_test_module()
    assert module._last_train_output is None

    with caplog.at_level(logging.INFO, logger="seq_sklearn.training"):
        module.on_train_epoch_end()

    assert [r for r in caplog.records if r.event == Event.TRAIN_VAR_SELECTION_ENTROPY] == []
    assert [r for r in caplog.records if r.event == Event.TRAIN_EPOCH] == []


def test_on_train_epoch_end_emits_events_from_compute_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = make_test_module(backbone=_EntropyBackbone())
    module.training_step(_batch(), batch_idx=0)

    with caplog.at_level(logging.INFO, logger="seq_sklearn.training"):
        module.on_train_epoch_end()

    entropy = [r for r in caplog.records if r.event == Event.TRAIN_VAR_SELECTION_ENTROPY]
    assert len(entropy) == 1
    assert entropy[0].payload == {"static_entropy": 1.0, "temporal_entropy": 0.5}
    assert any(r.event == Event.TRAIN_EPOCH for r in caplog.records)


def test_on_train_epoch_end_emits_every_metric_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The A15 loop emits one event per returned key, not just the first.

    A two-key ``compute_training_metrics`` payload makes a
    single-emit / break-after-first mutant fail: BOTH events must be
    present with their own payloads.
    """
    module = make_test_module(backbone=_TwoEventBackbone())
    module.training_step(_batch(), batch_idx=0)

    with caplog.at_level(logging.INFO, logger="seq_sklearn.training"):
        module.on_train_epoch_end()

    entropy = [r for r in caplog.records if r.event == Event.TRAIN_VAR_SELECTION_ENTROPY]
    hidden = [r for r in caplog.records if r.event == Event.TRAIN_HIDDEN_NORM]
    assert len(entropy) == 1
    assert len(hidden) == 1
    assert entropy[0].payload == {"static_entropy": 1.0, "temporal_entropy": 0.5}
    assert hidden[0].payload == {"mean_norm": 2.0}


# --- F9 non-finite-loss skip (training_step owns it; no callback) ------


class _ConstLoss(torch.nn.Module):
    """Loss returning a fixed scalar (NaN / inf / finite) for F9 tests."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(self, *_: object) -> Tensor:
        return torch.tensor(self._value, requires_grad=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_training_step_skips_non_finite_loss(bad: float, caplog: pytest.LogCaptureFixture) -> None:
    # F9: a single non-finite loss skips the step (return None ->
    # Lightning applies no gradient update) and emits the F11 event.
    module = make_test_module(loss=_ConstLoss(bad))
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.training"):
        out = module.training_step(_batch(), batch_idx=0)
    assert out is None
    assert module._last_train_output is None  # not stashed on a skip
    skipped = [r for r in caplog.records if r.event == Event.TRAIN_NAN_STEP_SKIPPED]
    assert len(skipped) == 1
    assert skipped[0].levelno == logging.WARNING
    assert skipped[0].payload["consecutive_nan_count"] == 1
    assert "step" in skipped[0].payload


def test_training_step_three_consecutive_non_finite_aborts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from seq_sklearn.errors import TrainingError

    module = make_test_module(loss=_ConstLoss(float("nan")))
    with caplog.at_level(logging.WARNING, logger="seq_sklearn.training"):
        assert module.training_step(_batch(), batch_idx=0) is None
        assert module.training_step(_batch(), batch_idx=1) is None
        with pytest.raises(TrainingError, match=r"3 consecutive non-finite"):
            module.training_step(_batch(), batch_idx=2)
    abort = [
        r
        for r in caplog.records
        if r.event == Event.TRAIN_NAN_STEP_SKIPPED and r.payload.get("aborting")
    ]
    assert len(abort) == 1
    assert abort[0].levelno == logging.ERROR
    assert abort[0].payload["consecutive_nan_count"] == 3


def test_training_step_finite_resets_consecutive_counter() -> None:
    from seq_sklearn.errors import TrainingError

    module = make_test_module()
    nan_loss = _ConstLoss(float("nan"))
    module.loss = nan_loss
    assert module.training_step(_batch(), batch_idx=0) is None
    assert module.training_step(_batch(), batch_idx=1) is None
    module.loss = _LossReturningScalar()  # finite -> resets the counter
    assert module.training_step(_batch(), batch_idx=2) is not None
    assert module._consecutive_nan == 0
    module.loss = nan_loss
    # Counter was reset: two more NaNs do not trip the 3-abort...
    assert module.training_step(_batch(), batch_idx=3) is None
    assert module.training_step(_batch(), batch_idx=4) is None
    # ...the third consecutive (post-reset) does.
    with pytest.raises(TrainingError):
        module.training_step(_batch(), batch_idx=5)


def test_training_step_finite_stashes_output_and_returns_loss() -> None:
    module = make_test_module()
    out = module.training_step(_batch(), batch_idx=0)
    assert out is not None
    assert module._last_train_output is not None
    assert module._consecutive_nan == 0


def test_on_before_optimizer_step_emits_grad_norm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # F11: train.grad_norm at DEBUG every step, payload step/grad_norm/lr.
    module = make_test_module()
    opt = torch.optim.SGD(module.parameters(), lr=0.05)
    param = next(module.parameters())
    param.grad = torch.ones_like(param)
    with caplog.at_level(logging.DEBUG, logger="seq_sklearn.training"):
        module.on_before_optimizer_step(opt)
    rec = [r for r in caplog.records if r.event == Event.TRAIN_GRAD_NORM]
    assert len(rec) == 1
    assert rec[0].levelno == logging.DEBUG
    assert set(rec[0].payload) == {"step", "grad_norm", "lr"}
    assert rec[0].payload["lr"] == pytest.approx(0.05)
    assert rec[0].payload["grad_norm"] > 0.0


def test_on_train_epoch_end_train_epoch_payload_has_f11_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # F11:1195 train.epoch payload: epoch, train_loss, val_loss, val_metric.
    module = make_test_module()
    module.trainer.callback_metrics = {
        "train_loss": torch.tensor(0.4),
        "val_loss": torch.tensor(0.6),
    }
    module.training_step(_batch(), batch_idx=0)
    with caplog.at_level(logging.INFO, logger="seq_sklearn.training"):
        module.on_train_epoch_end()
    epoch_rec = [r for r in caplog.records if r.event == Event.TRAIN_EPOCH]
    assert len(epoch_rec) == 1
    assert set(epoch_rec[0].payload) == {"epoch", "train_loss", "val_loss", "val_metric"}
    assert epoch_rec[0].payload["train_loss"] == pytest.approx(0.4)
    assert epoch_rec[0].payload["val_loss"] == pytest.approx(0.6)


def test_on_train_epoch_end_deferred_prune_raises_after_logging(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial = optuna.trial.FixedTrial({})
    monkeypatch.setattr(trial, "should_prune", lambda: True)
    module = make_test_module(backbone=_EntropyBackbone(), optuna_trial=trial)
    module.training_step(_batch(), batch_idx=0)

    module.on_validation_epoch_end()
    assert module._pending_prune == (0, 0.5)

    with (
        caplog.at_level(logging.INFO, logger="seq_sklearn.training"),
        pytest.raises(optuna.TrialPruned, match=r"epoch=0 metric=0.5"),
    ):
        module.on_train_epoch_end()

    # train.epoch + entropy fired BEFORE the deferred raise, and the
    # prune slot is cleared so a retry would not double-raise.
    assert any(r.event == Event.TRAIN_EPOCH for r in caplog.records)
    assert any(r.event == Event.TRAIN_VAR_SELECTION_ENTROPY for r in caplog.records)
    assert any(r.event == Event.OPTUNA_TRIAL_PRUNED for r in caplog.records)
    assert module._pending_prune is None


# --- Coverage gate: forward / optimizer / no-op surface ----------------


def test_training_step_stashes_backbone_output() -> None:
    module = make_test_module()
    loss = module.training_step(_batch(), batch_idx=0)
    assert loss is not None
    assert loss.requires_grad
    assert isinstance(module._last_train_output, BackboneOutput)


def test_validation_step_logs_val_metric() -> None:
    module = make_test_module()
    logged: dict[str, object] = {}
    captured_kw: dict[str, object] = {}

    def _log(name: str, value: object, **kw: object) -> None:
        logged[name] = value
        captured_kw.update(kw)

    module.log = _log  # type: ignore[method-assign]
    loss = module.validation_step(_batch(), batch_idx=0)
    assert "val_loss" in logged
    assert torch.equal(logged["val_loss"], loss)  # type: ignore[arg-type]
    # on_step=False / on_epoch=True are load-bearing: they populate
    # trainer.callback_metrics for the prune hook / EarlyStopping /
    # ModelCheckpoint, which read the epoch-aggregated metric.
    assert captured_kw["on_step"] is False
    assert captured_kw["on_epoch"] is True


def test_configure_optimizers_without_scheduler() -> None:
    module = make_test_module()
    cfg = cast(dict[str, object], module.configure_optimizers())
    assert set(cfg) == {"optimizer"}
    assert isinstance(cfg["optimizer"], torch.optim.Optimizer)


def test_configure_optimizers_with_scheduler() -> None:
    def scheduler_factory(opt: torch.optim.Optimizer) -> dict[str, object]:
        return {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0),
            "interval": "epoch",
        }

    module = make_test_module(scheduler_factory=scheduler_factory)
    cfg = cast(dict[str, object], module.configure_optimizers())
    assert set(cfg) == {"optimizer", "lr_scheduler"}


def test_on_validation_epoch_end_no_trial_is_noop() -> None:
    module = make_test_module()
    module.on_validation_epoch_end()
    assert module._pending_prune is None


def test_on_validation_epoch_end_no_metric_is_noop() -> None:
    trial = optuna.trial.FixedTrial({})
    module = make_test_module(optuna_trial=trial)
    module.trainer.callback_metrics = {}
    module.on_validation_epoch_end()
    assert module._pending_prune is None


def test_on_validation_epoch_end_not_pruned_leaves_pending_none() -> None:
    trial = optuna.trial.FixedTrial({})  # should_prune() defaults to False
    module = make_test_module(optuna_trial=trial)
    module.on_validation_epoch_end()
    assert module._pending_prune is None


def test_on_train_epoch_end_no_output_but_pending_prune_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial = optuna.trial.FixedTrial({})
    monkeypatch.setattr(trial, "should_prune", lambda: True)
    module = make_test_module(optuna_trial=trial)
    module.on_validation_epoch_end()
    assert module._last_train_output is None
    with pytest.raises(optuna.TrialPruned):
        module.on_train_epoch_end()


def test_on_train_epoch_end_with_output_no_prune_does_not_raise() -> None:
    module = make_test_module()
    module.training_step(_batch(), batch_idx=0)
    module.on_train_epoch_end()
    assert module._pending_prune is None


def test_bptt_step_is_v1_noop() -> None:
    module = make_test_module()
    out = module._bptt_step(_batch())
    assert out.requires_grad is False
    assert out.item() == 0.0
