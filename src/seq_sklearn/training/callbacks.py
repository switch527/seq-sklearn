"""Lightning callbacks (per architecture A7 / requirements F9 / N4).

Four callbacks the Trainer attaches:

* :class:`NaNLossGuard` aborts after three consecutive NaN training
  losses (F9).
* :class:`GradScalerWatchdog` aborts after three consecutive
  ``GradScaler`` scale decreases under mixed precision (F9); a no-op
  when no scaler is present (CPU, bf16-mixed, fp32).
* :class:`EventEmitter` exposes a structured-log ``emit`` from inside
  Lightning hooks (F11).
* :class:`RngStateCallback` round-trips Python / numpy / torch / CUDA
  RNG state through the checkpoint (N4; works around Lightning issue
  #20204).

The callbacks read only the documented Lightning surface; they do not
import the Phase 4b ``_LightningModule``.
"""

import logging
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from seq_sklearn.errors import TrainingError
from seq_sklearn.logging import Event, emit

__all__ = [
    "EventEmitter",
    "GradScalerWatchdog",
    "NaNLossGuard",
    "RngStateCallback",
]

logger = logging.getLogger(__name__)

_RNG_KEY = "seq_sklearn_rng"
_NAN_LIMIT = 3
_SCALE_DECREASE_LIMIT = 3


class EventEmitter(Callback):
    """Structured-log emitter usable from inside Lightning hooks.

    :meth:`emit` routes through the library's structured-log helper on
    the ``seq_sklearn.training`` logger so the ``event`` / ``payload``
    keys land as :class:`logging.LogRecord` attributes (accessible as
    ``record.event`` / ``record.payload`` and captured by ``caplog``
    while ``propagate`` stays ``True``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger("seq_sklearn.training")

    def emit(self, event: Event, level: int = logging.INFO, **payload: Any) -> None:
        """Emit a structured record for ``event`` with ``payload``."""
        emit(self._logger, event, level=level, **payload)


class NaNLossGuard(Callback):
    """Abort training after three consecutive NaN losses (F9).

    Lightning passes the scalar loss returned by ``training_step`` as
    ``outputs`` when ``automatic_optimization=True``. A NaN increments
    an internal counter and emits ``train.nan_step_skipped``; the third
    consecutive NaN raises :class:`TrainingError` with the offending
    ``batch_idx`` in the log payload. A non-NaN step resets the counter.
    """

    def __init__(self) -> None:
        super().__init__()
        self._consecutive = 0
        self._emitter = EventEmitter()

    def on_train_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        """Track consecutive NaN losses; raise on the third."""
        is_nan = isinstance(outputs, torch.Tensor) and bool(torch.isnan(outputs).any())
        if not is_nan:
            self._consecutive = 0
            return
        self._consecutive += 1
        self._emitter.emit(
            Event.TRAIN_NAN_STEP_SKIPPED,
            batch_idx=batch_idx,
            consecutive=self._consecutive,
        )
        if self._consecutive >= _NAN_LIMIT:
            self._emitter.emit(
                Event.TRAIN_NAN_STEP_SKIPPED,
                level=logging.ERROR,
                batch_idx=batch_idx,
                consecutive=self._consecutive,
                aborting=True,
            )
            raise TrainingError(
                f"{_NAN_LIMIT} consecutive NaN training steps; aborting "
                f"per F9 (batch_idx={batch_idx})"
            )


class GradScalerWatchdog(Callback):
    """Abort after three consecutive mixed-precision scale decreases (F9).

    Defensively checks ``hasattr(trainer.precision_plugin, "scaler")``
    and is a no-op when the attribute is absent (CPU, ``bf16-mixed`` on
    CC>=8.0, fp32). When a scaler is present, a decrease in
    ``scaler.get_scale()`` signals an overflow-driven skipped step;
    three in a row raise :class:`TrainingError` and emit
    ``train.mixed_precision_diverged`` at ERROR.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prev_scale: float | None = None
        self._consecutive_decreases = 0
        self._emitter = EventEmitter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: torch.Tensor | Mapping[str, Any] | None,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        """Track the GradScaler scale; raise on three straight decreases."""
        plugin = trainer.precision_plugin
        if not hasattr(plugin, "scaler"):
            return
        scaler = plugin.scaler  # type: ignore[attr-defined]
        if scaler is None:
            return
        scale = float(scaler.get_scale())
        if self._prev_scale is not None and scale < self._prev_scale:
            self._consecutive_decreases += 1
        else:
            self._consecutive_decreases = 0
        self._prev_scale = scale
        if self._consecutive_decreases >= _SCALE_DECREASE_LIMIT:
            self._emitter.emit(
                Event.TRAIN_MIXED_PRECISION_DIVERGED,
                level=logging.ERROR,
                batch_idx=batch_idx,
                scale=scale,
                consecutive_decreases=self._consecutive_decreases,
            )
            raise TrainingError(
                f"{_SCALE_DECREASE_LIMIT} consecutive GradScaler scale "
                f"decreases; mixed-precision training diverged "
                f"(batch_idx={batch_idx})"
            )


class RngStateCallback(Callback):
    """Round-trip RNG state through the checkpoint (N4).

    ``on_save_checkpoint`` snapshots Python / numpy / torch (and, when
    CUDA is available, every device's) RNG state into
    ``checkpoint["seq_sklearn_rng"]``; ``on_load_checkpoint`` restores
    them. Closes the Lightning ``load_from_checkpoint`` RNG gap (issue
    #20204) so a resumed run continues bit-identically.
    """

    def on_save_checkpoint(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        """Snapshot the four RNG sources into the checkpoint."""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }
        checkpoint[_RNG_KEY] = state

    def on_load_checkpoint(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        """Restore the RNG sources captured at save time."""
        state = checkpoint.get(_RNG_KEY)
        if state is None:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        cuda_state = state["cuda"]
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
