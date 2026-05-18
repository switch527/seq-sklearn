"""Single source of truth for the F5 validity-matrix field domains.

Tuple constants (immutable, importable from anywhere) covering every
documented value of ``task_type``, ``loss_strategy``,
``imbalance_strategy``, and ``calibration_strategy``. v1.1 task types
(:py:obj:`MULTILABEL` and :py:obj:`REGRESSION_MULTIOUTPUT`) are listed
here so the v1 validator rejects them with a clear "scheduled for v1.1"
message rather than failing later on shape mismatch.
"""

from typing import Final

__all__ = [
    "CALIBRATION_STRATEGIES",
    "CLASSIFICATION_TASK_TYPES",
    "IMBALANCE_STRATEGIES",
    "LOSS_STRATEGIES",
    "REGRESSION_TASK_TYPES",
    "TASK_TYPES",
    "V1_1_TASK_TYPES",
    "V1_TASK_TYPES",
]


TASK_TYPES: Final[tuple[str, ...]] = (
    "binary",
    "multiclass",
    "multilabel",
    "regression_point",
    "regression_quantile",
    "regression_multioutput",
)
"""Every documented ``task_type`` value, v1 + v1.1."""

V1_TASK_TYPES: Final[tuple[str, ...]] = (
    "binary",
    "multiclass",
    "regression_point",
    "regression_quantile",
)
"""Task types supported in v1. v1.1 adds multilabel and regression_multioutput."""

V1_1_TASK_TYPES: Final[tuple[str, ...]] = (
    "multilabel",
    "regression_multioutput",
)
"""Task types scheduled for v1.1. v1 rejects with a clear message."""

CLASSIFICATION_TASK_TYPES: Final[tuple[str, ...]] = (
    "binary",
    "multiclass",
    "multilabel",
)
"""Classification ``task_type`` values, v1 + v1.1. Single source for the
model-family split (the Optuna sampler intersects this with
:py:obj:`V1_TASK_TYPES`)."""

REGRESSION_TASK_TYPES: Final[tuple[str, ...]] = (
    "regression_point",
    "regression_quantile",
    "regression_multioutput",
)
"""Regression ``task_type`` values, v1 + v1.1. Counterpart of
:py:obj:`CLASSIFICATION_TASK_TYPES`; together they partition
:py:obj:`TASK_TYPES`."""

LOSS_STRATEGIES: Final[tuple[str, ...]] = (
    "cross_entropy",
    "focal",
    "mse",
    "mae",
    "huber",
    "pinball",
)
"""Every documented ``loss_strategy`` value."""

IMBALANCE_STRATEGIES: Final[tuple[str, ...]] = (
    "none",
    "class_weighted",
    "oversample_minority",
    "undersample_majority",
)
"""Every documented ``imbalance_strategy`` value."""

CALIBRATION_STRATEGIES: Final[tuple[str, ...]] = (
    "none",
    "temperature",
    "platt",
    "isotonic",
    "conformal",
    "isotonic_quantile",
)
"""Every documented ``calibration_strategy`` value."""
