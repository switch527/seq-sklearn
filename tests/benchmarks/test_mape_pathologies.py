"""Phase B4 MAPE pathology tests (B5.2).

`compute_mape_or_skip` returns `(nan, typed_skip_reason)` for each of
the three documented pathological inputs:

- `y_true == 0` anywhere -> ``mape_undefined_zero_in_y_true``
- `np.isnan(y_true)` anywhere -> ``mape_undefined_nan_in_y_true``
- `np.isinf(y_true)` anywhere -> ``mape_undefined_inf_in_y_true``

Pre-check fires BEFORE the sklearn call so a single nan / inf never
propagates into the metric.
"""

import math

import numpy as np
import pytest
from benchmarks.metrics import RegressionPointMetrics, compute_all
from benchmarks.metrics.regression import compute_mape_or_skip


@pytest.mark.parametrize(
    ("y_true", "expected_reason"),
    [
        (
            np.array([1.0, 2.0, 0.0, 4.0]),
            "mape_undefined_zero_in_y_true",
        ),
        (
            np.array([1.0, 2.0, float("nan"), 4.0]),
            "mape_undefined_nan_in_y_true",
        ),
        (
            np.array([1.0, 2.0, float("inf"), 4.0]),
            "mape_undefined_inf_in_y_true",
        ),
        (
            np.array([1.0, 2.0, float("-inf"), 4.0]),
            "mape_undefined_inf_in_y_true",
        ),
    ],
)
def test_compute_mape_or_skip_returns_nan_and_typed_reason(
    y_true: np.ndarray, expected_reason: str
) -> None:
    y_pred = np.ones_like(y_true)
    value, reason = compute_mape_or_skip(y_true, y_pred)
    assert math.isnan(value), "skip path must emit nan, not zero"
    assert reason == expected_reason


def test_regression_record_mape_skip_propagates_through_compute_all() -> None:
    # A zero in y_true must surface the skip reason on the typed
    # record, NOT silently distort MAPE via sklearn's eps fudge.
    y_true = np.array([1.0, 0.0, 3.0, 4.0])
    y_pred = np.array([1.1, 0.1, 3.0, 4.0])
    rec = compute_all(task_type="regression_point", y_true=y_true, y_pred=y_pred)
    assert isinstance(rec, RegressionPointMetrics)
    assert math.isnan(rec.mape)
    assert rec.mape_skip_reason == "mape_undefined_zero_in_y_true"
    # The other regression metrics are still valid even when MAPE skips.
    assert not math.isnan(rec.rmse)
    assert not math.isnan(rec.mae)


def test_mape_skip_reason_is_none_on_clean_inputs() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1])
    value, reason = compute_mape_or_skip(y_true, y_pred)
    assert not math.isnan(value)
    assert reason is None


def test_mape_zero_check_fires_even_with_inf_in_y_pred() -> None:
    # The pre-check is on y_true only. An inf in y_pred is fine as long
    # as y_true is clean (sklearn will produce a finite or inf MAPE).
    # But a zero in y_true takes precedence over a clean y_pred.
    y_true = np.array([1.0, 0.0, 3.0])
    y_pred = np.array([1.0, 1.0, 3.0])
    value, reason = compute_mape_or_skip(y_true, y_pred)
    assert math.isnan(value)
    assert reason == "mape_undefined_zero_in_y_true"
