"""Regression metrics (B5.2).

Primary: `RMSE` (`root_mean_squared_error`).
Secondary: `MAE`, `R²`, MAPE (with three typed skip reasons), pinball
loss at the configured quantile levels.

MAPE pre-checks `y_true` for three pathological inputs per B5.2 and
emits `nan` plus a typed skip-reason string when any is present:

- ``mape_undefined_zero_in_y_true``: any `y_true == 0`. The
  sklearn `mean_absolute_percentage_error` adds an `epsilon` to
  the denominator which silently distorts the GLOBAL number, not
  just the zero rows, so an any-zero skip is the principled
  response.
- ``mape_undefined_nan_in_y_true``: any `np.isnan(y_true)`.
- ``mape_undefined_inf_in_y_true``: any `np.isinf(y_true)`.

The pre-check fires before any sklearn call, so an `inf`/`nan` does
not propagate through the metric.
"""

from typing import Literal

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    r2_score,
    root_mean_squared_error,
)

MapeSkipReason = Literal[
    "mape_undefined_zero_in_y_true",
    "mape_undefined_nan_in_y_true",
    "mape_undefined_inf_in_y_true",
]


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def compute_mape_or_skip(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, MapeSkipReason | None]:
    """Return `(mape_value, skip_reason)`.

    On a pathological input, `mape_value` is `nan` and
    `skip_reason` is one of the three named strings. On a clean
    input, `skip_reason` is `None`.
    """
    y_true_arr = np.asarray(y_true)
    if np.isnan(y_true_arr).any():
        return float("nan"), "mape_undefined_nan_in_y_true"
    if np.isinf(y_true_arr).any():
        return float("nan"), "mape_undefined_inf_in_y_true"
    if (y_true_arr == 0).any():
        return float("nan"), "mape_undefined_zero_in_y_true"
    return float(mean_absolute_percentage_error(y_true, y_pred)), None


def compute_pinball_at_quantiles(
    y_true: np.ndarray, y_quantiles: np.ndarray, *, quantiles: tuple[float, ...]
) -> dict[str, float]:
    """Compute the pinball loss at each configured quantile level.

    `y_quantiles` has shape `(N, len(quantiles))` with the column
    ordering matching `quantiles`. Returns a dict mapping
    ``pinball_q{level}`` to the per-quantile loss; lower is better
    per the `sklearn.metrics.mean_pinball_loss` convention.
    """
    out: dict[str, float] = {}
    for j, q in enumerate(quantiles):
        # The level string uses two decimals so 0.1 -> q0.10.
        out[f"pinball_q{q:.2f}"] = float(mean_pinball_loss(y_true, y_quantiles[:, j], alpha=q))
    return out
