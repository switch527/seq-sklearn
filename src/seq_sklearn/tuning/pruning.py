"""Optuna objective guard (A16).

``study.optimize`` defaults to ``catch=()``: any exception out of the
objective terminates the whole study. A failed *trial* (an illegal
sampled cell, a NaN-loss abort) should prune that trial, not kill the
study; a genuine bug (bad input contract, an unexpected error) should
still fail fast. :func:`optuna_trial_guard` draws that line.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import optuna

from seq_sklearn.errors import ConfigError, TrainingError

__all__ = ["optuna_trial_guard"]

logger = logging.getLogger(__name__)


@contextmanager
def optuna_trial_guard(trial: optuna.trial.BaseTrial) -> Generator[None, None, None]:
    """Wrap a user ``objective(trial)`` body and prune recoverable failures.

    :class:`ConfigError` and :class:`TrainingError` are re-raised as
    :class:`optuna.TrialPruned` with the original message, so the trial
    is recorded as pruned and the study continues.
    :class:`DataContractError`, :class:`KeyboardInterrupt`, and every
    unexpected exception propagate unchanged so the study fails fast on
    genuine bugs.

    Args:
        trial: the active Optuna trial (used for the pruned-trial log
            line; the guard never reports or mutates trial state).

    Raises:
        optuna.TrialPruned: when the wrapped body raises
            :class:`ConfigError` or :class:`TrainingError`.
    """
    try:
        yield
    except (ConfigError, TrainingError) as exc:
        logger.info("Pruning trial %s: %s", getattr(trial, "number", "?"), exc)
        raise optuna.TrialPruned(str(exc)) from exc
