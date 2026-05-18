"""Structured logging events (per architecture A15 / requirements F11).

The library emits structured log records via :func:`emit` rather than
print or unstructured strings. Each event carries an :class:`Event` name
plus a payload dict; both land in the :class:`logging.LogRecord` ``extra``
slot so they are accessible as ``record.event`` and ``record.payload``
under standard handlers and pytest's ``caplog``.

The library never configures handlers itself (per F11 / standard library
best practice). Callers attach the handler of their choice.
"""

import logging
from enum import StrEnum
from typing import Any

__all__ = ["Event", "emit"]


class Event(StrEnum):
    """F11 event-name enumeration.

    Every structured log record the library emits carries one of these as
    the ``event`` attribute. Values are dotted strings so they round-trip
    through JSON / Prometheus / OTLP without escaping.
    """

    TRAIN_GRAD_NORM = "train.grad_norm"
    TRAIN_EPOCH = "train.epoch"
    TRAIN_VAR_SELECTION_ENTROPY = "train.var_selection_entropy"
    TRAIN_ATTENTION_ENTROPY = "train.attention_entropy"
    TRAIN_HIDDEN_NORM = "train.hidden_norm"
    TRAIN_NAN_STEP_SKIPPED = "train.nan_step_skipped"
    TRAIN_MIXED_PRECISION_DIVERGED = "train.mixed_precision_diverged"
    CALIBRATION_FIT = "calibration.fit"
    CALIBRATION_SMALL_SET = "calibration.small_set"
    OPTUNA_TRIAL_PRUNED = "optuna.trial_pruned"
    DATA_DUPLICATE_FLOOR_BREACH_COUNT = "data.duplicate_floor_breach_count"
    DATA_UNSEEN_CATEGORIES = "data.unseen_categories"
    HARDWARE_DETECT = "hardware.detect"


def emit(
    logger: logging.Logger,
    event: Event,
    level: int = logging.INFO,
    **payload: Any,
) -> None:
    """Emit a structured log record.

    The record's ``message`` is the event value (e.g. ``"train.epoch"``).
    The ``event`` name and ``payload`` dict are attached via ``extra=`` so
    they appear as attributes on the :class:`logging.LogRecord` under both
    standard handlers and pytest's ``caplog``.

    Example::

        import logging
        logger = logging.getLogger("seq_sklearn.example")
        emit(logger, Event.TRAIN_EPOCH, epoch=1, train_loss=0.5)
        # Produces a record with record.event == "train.epoch" and
        # record.payload == {"epoch": 1, "train_loss": 0.5}.
    """
    logger.log(
        level,
        event.value,
        extra={"event": event.value, "payload": dict(payload)},
    )
