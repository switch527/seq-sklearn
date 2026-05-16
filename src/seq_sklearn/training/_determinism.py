"""Strict-determinism toggle (per requirements N4).

The library, not the caller, configures determinism: the Trainer calls
:func:`enable_strict_mode` when ``precision="32-true"`` and a seed is
set so the bit-identical contract is a library guarantee. The function
is idempotent; a second call is a no-op and emits no warning.
"""

import logging
import os

import torch

__all__ = ["enable_strict_mode"]

logger = logging.getLogger(__name__)

_CUBLAS_DETERMINISTIC = ":4096:8"


def enable_strict_mode() -> None:
    """Force the four N4 determinism side effects.

    Sets ``torch.use_deterministic_algorithms(True, warn_only=False)``,
    ``torch.backends.cudnn.deterministic = True``,
    ``torch.backends.cudnn.benchmark = False``, and exports
    ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` only when it is unset (a
    caller-supplied non-default value is left untouched per N1
    Scenario B). Idempotent: calling again re-asserts the same state
    without emitting any log or warning.
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_DETERMINISTIC
