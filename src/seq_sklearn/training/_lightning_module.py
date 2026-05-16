"""LightningModule wrapper (per architecture A7 / A15).

:class:`_LightningModule` is the internal pytorch-lightning module the
:class:`seq_sklearn.training.trainer.Trainer` wraps around a backbone /
head / loss triple. It owns the forward + loss path, the optimizer /
scheduler wiring through the curried factories, and the structured-log
emission on epoch boundaries.

Two contracts are load-bearing and pinned by the Phase 4b deliverable
tests:

* The Optuna deferred-prune pattern (A7). ``on_validation_epoch_end``
  fires before ``on_train_epoch_end`` in Lightning 2.6; raising
  ``optuna.TrialPruned`` from the validation hook would skip the
  ``train.epoch`` / entropy structured-log events for the pruned epoch.
  The decision is stashed on ``self._pending_prune`` and re-raised at
  the END of ``on_train_epoch_end`` so logging always runs first.
* The v1 -> v3 cross-family abstraction (A15). ``on_train_epoch_end``
  delegates to ``self.backbone.compute_training_metrics`` and emits one
  structured event per returned ``{event_name: payload}`` entry. The
  module reads only the base :class:`BackboneOutput` fields; it never
  touches family-specific introspection attributes, so v3 recurrent
  backbones add events by overriding ``compute_training_metrics`` alone.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any, cast

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerConfig,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor, nn, optim

from seq_sklearn.errors import TrainingError
from seq_sklearn.logging import Event, emit
from seq_sklearn.models._backbone import BackboneOutput, BaseBackbone

__all__ = ["_LightningModule"]

# F9: three consecutive non-finite (NaN/inf) loss steps abort the run.
_NAN_ABORT_LIMIT = 3


class _LightningModule(pl.LightningModule):
    """Internal Lightning module wrapping a backbone / head / loss triple.

    The constructor is explicit so unit tests build the module from plain
    callables without standing up an Estimator or a real Trainer (A7
    construction-order note). ``automatic_optimization`` stays ``True``;
    v1 ships the encoder-style TFT, so :meth:`_bptt_step` is a no-op and
    truncated BPTT against ``bptt_window`` is a v3 concern.
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        head: nn.Module,
        loss: nn.Module,
        optimizer_factory: Callable[[Iterable[nn.Parameter]], optim.Optimizer],
        scheduler_factory: Callable[[optim.Optimizer], dict[str, object]] | None,
        val_metric_name: str = "val_loss",
        bptt_window: int | None = None,
        optuna_trial: optuna.trial.BaseTrial | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.val_metric_name = val_metric_name
        self.bptt_window = bptt_window
        self._optuna_trial = optuna_trial
        self._pending_prune: tuple[int, float] | None = None
        self._last_train_output: BackboneOutput | None = None
        self._consecutive_nan = 0
        self._logger = logging.getLogger("seq_sklearn.training")
        self.automatic_optimization = True

    def _forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, BackboneOutput]:
        """Run backbone then head, returning ``(loss, backbone_output)``.

        The loss is computed against ``batch["target"]``; the backbone
        output is returned so :meth:`training_step` can stash the most
        recent one for the A15 ``compute_training_metrics`` delegation.
        """
        backbone_out: BackboneOutput = self.backbone(batch)
        logits: Tensor = self.head(backbone_out.representation)
        loss_value: Tensor = self.loss(logits, batch["target"])
        return loss_value, backbone_out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:  # noqa: ARG002
        """Forward + loss; enforce the F9 non-finite-loss skip contract.

        F9 requires a single NaN/inf loss step to skip with no gradient
        update. Under ``automatic_optimization=True`` the only mechanism
        that actually skips backward + ``optimizer.step()`` is returning
        ``None`` from this hook (a post-hoc callback fires after the
        optimizer step and cannot skip it). On a non-finite loss this
        emits ``train.nan_step_skipped`` and returns ``None``; the third
        consecutive non-finite step aborts with ``TrainingError`` per F9.
        A finite step resets the counter and stashes the backbone output
        for the A15 delegation.

        Raises:
            TrainingError: three consecutive non-finite loss steps.
        """
        loss_value, backbone_out = self._forward(batch)
        if not bool(torch.isfinite(loss_value).all()):
            self._consecutive_nan += 1
            emit(
                self._logger,
                Event.TRAIN_NAN_STEP_SKIPPED,
                level=logging.WARNING,
                step=self.global_step,
                consecutive_nan_count=self._consecutive_nan,
            )
            if self._consecutive_nan >= _NAN_ABORT_LIMIT:
                emit(
                    self._logger,
                    Event.TRAIN_NAN_STEP_SKIPPED,
                    level=logging.ERROR,
                    step=self.global_step,
                    consecutive_nan_count=self._consecutive_nan,
                    aborting=True,
                )
                raise TrainingError(
                    f"{_NAN_ABORT_LIMIT} consecutive non-finite loss steps; "
                    "aborting per F9 (try precision='32-true' or a lower "
                    "learning rate)"
                )
            return None
        self._consecutive_nan = 0
        self._last_train_output = backbone_out
        self.log("train_loss", loss_value, on_step=False, on_epoch=True)
        return loss_value

    def on_before_optimizer_step(self, optimizer: optim.Optimizer) -> None:
        """Emit ``train.grad_norm`` (F11) before each optimizer step.

        DEBUG, every step, payload ``step`` / ``grad_norm`` / ``lr``. The
        2-norm is over the post-backward gradients; a NaN/inf step never
        reaches here because :meth:`training_step` returned ``None``.
        """
        grads = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        grad_norm = float(torch.norm(torch.stack(grads), 2)) if grads else 0.0
        emit(
            self._logger,
            Event.TRAIN_GRAD_NORM,
            level=logging.DEBUG,
            step=self.global_step,
            grad_norm=grad_norm,
            lr=float(optimizer.param_groups[0]["lr"]),
        )

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        """Forward + loss; log ``val_metric_name`` for the Optuna / ES hooks.

        ``self.log`` populates ``self.trainer.callback_metrics`` so
        :meth:`on_validation_epoch_end` and Lightning's ``EarlyStopping``
        / ``ModelCheckpoint`` read the same metric.
        """
        loss_value, _ = self._forward(batch)
        self.log(self.val_metric_name, loss_value, on_step=False, on_epoch=True)
        return loss_value

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build the optimizer (and optional scheduler) from the factories."""
        opt = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is None:
            return OptimizerConfig(optimizer=opt)
        sched_dict = self.scheduler_factory(opt)
        return OptimizerLRSchedulerConfig(
            optimizer=opt,
            lr_scheduler=cast(LRSchedulerConfigType, sched_dict),
        )

    def on_validation_epoch_end(self) -> None:
        """Record an Optuna prune decision without raising yet.

        Lightning 2.6 fires this before ``on_train_epoch_end``. Raising
        ``optuna.TrialPruned`` here would skip the train-epoch structured
        logging for the pruned epoch, so the decision is deferred to the
        end of ``on_train_epoch_end``.
        """
        val_metric = self.trainer.callback_metrics.get(self.val_metric_name)
        if self._optuna_trial is not None and val_metric is not None:
            self._optuna_trial.report(val_metric.item(), step=self.current_epoch)
            if self._optuna_trial.should_prune():
                self._pending_prune = (self.current_epoch, val_metric.item())

    def _metric(self, name: str) -> float | None:
        """Read a logged metric from ``callback_metrics`` as a float.

        Returns ``None`` when the metric is absent (e.g. validation has
        not run for this epoch), so the ``train.epoch`` payload carries
        an explicit null rather than a stale or fabricated value.
        """
        value = self.trainer.callback_metrics.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return None

    def on_train_epoch_end(self) -> None:
        """Emit train.epoch + the A15 metric events, then the deferred prune.

        The None-guard short-circuits when no training batch completed
        this epoch (``_last_train_output is None``); a deferred Optuna
        prune still raises afterwards so a pruned no-batch epoch is not
        silently swallowed.

        Raises:
            optuna.TrialPruned: when ``on_validation_epoch_end`` stashed a
                prune decision for this epoch. Raised LAST so the
                structured-log events fire first.
        """
        if self._last_train_output is not None:
            emit(
                self._logger,
                Event.TRAIN_EPOCH,
                epoch=self.current_epoch,
                train_loss=self._metric("train_loss"),
                val_loss=self._metric("val_loss"),
                val_metric=self._metric(self.val_metric_name),
            )
            payloads = self.backbone.compute_training_metrics(self._last_train_output)
            for event_name, payload in payloads.items():
                emit(
                    self._logger,
                    Event(event_name),
                    **cast(dict[str, Any], payload),
                )
        if self._pending_prune is not None:
            epoch, metric = self._pending_prune
            self._pending_prune = None
            emit(
                self._logger,
                Event.OPTUNA_TRIAL_PRUNED,
                epoch=epoch,
                metric=metric,
            )
            raise optuna.TrialPruned(f"epoch={epoch} metric={metric}")

    def _bptt_step(self, batch: dict[str, Tensor]) -> Tensor:  # noqa: ARG002
        """v1 no-op for the encoder-style TFT.

        v3 recurrent models override this for truncated BPTT against
        ``bptt_window``; v1 routes every step through
        :meth:`training_step`, so this returns a detached zero scalar and
        is never on the v1 training path.
        """
        return torch.zeros((), requires_grad=False)
