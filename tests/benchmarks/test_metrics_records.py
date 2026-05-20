"""Phase B4 record-shape tests (B5).

The four typed metric records carry exactly the metrics applicable to
their task type, with the user-facing extension (B5 delta R4) reflected
in suffixed multiclass names (`*_macro_zd0` / `*_weighted_zd0`) and
unsuffixed binary names (`precision_zd0` / `recall_zd0` / `f1_zd0`).
The regression point and quantile records carry the MAPE skip-reason
slot per B5.2.

R1-followup: every `compute_all` `ValueError` raise is exercised so
a refactor that drops one of the guards is caught by the suite.
"""

from typing import cast

import numpy as np
import pytest
from benchmarks.config import TaskType
from benchmarks.metrics import (
    BinaryMetrics,
    MulticlassMetrics,
    RegressionPointMetrics,
    RegressionQuantileMetrics,
    compute_all,
)


def _binary_record() -> BinaryMetrics:
    rec = compute_all(
        task_type="binary",
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 0, 1]),
        y_proba=np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]]),
        pos_label=1,
    )
    assert isinstance(rec, BinaryMetrics)
    return rec


def _multiclass_record() -> MulticlassMetrics:
    rec = compute_all(
        task_type="multiclass",
        y_true=np.array([0, 1, 2, 1]),
        y_pred=np.array([0, 1, 2, 0]),
        y_proba=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.1, 0.8],
                [0.5, 0.4, 0.1],
            ]
        ),
        labels=np.array([0, 1, 2]),
    )
    assert isinstance(rec, MulticlassMetrics)
    return rec


def _regression_point_record() -> RegressionPointMetrics:
    rec = compute_all(
        task_type="regression_point",
        y_true=np.array([1.0, 2.0, 3.0, 4.0]),
        y_pred=np.array([1.1, 1.9, 3.2, 3.7]),
    )
    assert isinstance(rec, RegressionPointMetrics)
    return rec


def _regression_quantile_record() -> RegressionQuantileMetrics:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.7])
    y_quantiles = np.stack([y_pred - 0.5, y_pred, y_pred + 0.5], axis=1)
    rec = compute_all(
        task_type="regression_quantile",
        y_true=y_true,
        y_pred=y_pred,
        y_quantiles=y_quantiles,
        quantiles=(0.1, 0.5, 0.9),
    )
    assert isinstance(rec, RegressionQuantileMetrics)
    return rec


def test_binary_record_has_zd0_names_and_no_bare_names() -> None:
    rec = _binary_record()
    fields = set(type(rec).model_fields.keys())
    # The suffixed binary names from the B5 extension are present.
    assert {"precision_zd0", "recall_zd0", "f1_zd0"}.issubset(fields)
    # The bare names from a naive sklearn passthrough are NOT present:
    # the metric record commits to the convention via the field name.
    assert "precision" not in fields
    assert "recall" not in fields
    assert "f1" not in fields
    # Binary-specific metrics that should be there.
    assert {"log_loss", "roc_auc", "pr_auc", "brier", "ece_q15"}.issubset(fields)
    # The multiclass-specific suffixed names must NOT leak into binary.
    assert "precision_macro_zd0" not in fields
    assert "precision_weighted_zd0" not in fields
    assert "brier_multiclass_mean" not in fields


def test_multiclass_record_has_suffixed_averages_only() -> None:
    rec = _multiclass_record()
    fields = set(type(rec).model_fields.keys())
    # Multiclass carries BOTH macro and weighted, suffixed by name.
    expected_pairs = {
        "precision_macro_zd0",
        "precision_weighted_zd0",
        "recall_macro_zd0",
        "recall_weighted_zd0",
        "f1_macro_zd0",
        "f1_weighted_zd0",
    }
    assert expected_pairs.issubset(fields)
    # Bare names that would be ambiguous between macro and weighted: absent.
    assert "precision" not in fields
    assert "recall" not in fields
    assert "f1" not in fields
    # The binary-specific names must NOT leak into multiclass.
    assert "precision_zd0" not in fields
    assert "recall_zd0" not in fields
    assert "f1_zd0" not in fields
    assert "brier" not in fields  # only brier_multiclass_mean for K>2
    # Multiclass-specific metrics present.
    assert {
        "log_loss",
        "roc_auc_macro_ovr",
        "pr_auc_macro_ovr",
        "brier_multiclass_mean",
        "ece_q15",
    }.issubset(fields)


def test_regression_point_record_carries_mape_and_skip_reason_slot() -> None:
    rec = _regression_point_record()
    fields = set(type(rec).model_fields.keys())
    assert {"rmse", "mae", "r2", "mape", "mape_skip_reason"}.issubset(fields)
    # No leakage of classification fields.
    assert "accuracy" not in fields
    assert "log_loss" not in fields
    # No leakage of the quantile field.
    assert "pinball" not in fields
    # Clean input: skip_reason is None (slot present, value None).
    assert rec.mape_skip_reason is None


def test_regression_quantile_record_carries_pinball_dict() -> None:
    rec = _regression_quantile_record()
    fields = set(type(rec).model_fields.keys())
    assert {"rmse", "mae", "r2", "mape", "mape_skip_reason", "pinball"}.issubset(fields)
    # Pinball dict carries one key per configured quantile level.
    assert set(rec.pinball.keys()) == {
        "pinball_q0.10",
        "pinball_q0.50",
        "pinball_q0.90",
    }
    # No classification leakage.
    assert "accuracy" not in fields


def test_records_reject_extra_fields() -> None:
    # `extra="forbid"` is the contract; a future maintainer cannot
    # silently inject an extra typed field via the constructor path.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # Drives the pydantic `extra="forbid"` validator by hand-feeding
        # a stray kwarg the model never declared. Pyright correctly
        # flags this as a structural type error; the runtime
        # ValidationError IS the assertion.
        BinaryMetrics(
            log_loss=0.1,
            accuracy=0.5,
            precision_zd0=0.5,
            recall_zd0=0.5,
            f1_zd0=0.5,
            roc_auc=0.5,
            pr_auc=0.5,
            balanced_accuracy=0.5,
            mcc=0.0,
            brier=0.1,
            ece_q15=0.0,
            not_a_real_metric=1.0,  # pyright: ignore[reportCallIssue]
        )


# --- R1 / qa-C2: every `compute_all` ValueError raise is exercised. ---


def test_compute_all_binary_missing_y_proba_raises() -> None:
    with pytest.raises(ValueError, match="y_proba"):
        compute_all(
            task_type="binary",
            y_true=np.array([0, 1, 0, 1]),
            y_pred=np.array([0, 1, 0, 1]),
            pos_label=1,
        )


def test_compute_all_binary_missing_pos_label_raises() -> None:
    with pytest.raises(ValueError, match="pos_label"):
        compute_all(
            task_type="binary",
            y_true=np.array([0, 1, 0, 1]),
            y_pred=np.array([0, 1, 0, 1]),
            y_proba=np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
        )


def test_compute_all_multiclass_missing_y_proba_raises() -> None:
    with pytest.raises(ValueError, match="y_proba"):
        compute_all(
            task_type="multiclass",
            y_true=np.array([0, 1, 2]),
            y_pred=np.array([0, 1, 2]),
        )


def test_compute_all_regression_quantile_missing_args_raises() -> None:
    with pytest.raises(ValueError, match="y_quantiles"):
        compute_all(
            task_type="regression_quantile",
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 1.9, 3.0]),
        )


def test_compute_all_unsupported_task_type_raises() -> None:
    # The runtime branch fires only when TaskType is widened without
    # a matching dispatch arm. Cast bypasses pyright's Literal check.
    with pytest.raises(ValueError, match="unsupported task_type"):
        compute_all(
            task_type=cast(TaskType, "not_a_real_task_type"),
            y_true=np.array([1.0]),
            y_pred=np.array([1.0]),
        )


# --- R1 / arch-C1: `pos_label != 1` does not silently invert the
# probability-indexed metrics. ---


def test_binary_pos_label_zero_does_not_invert_indexed_metrics() -> None:
    # Same outcome but with the positive class encoded as 0.
    # y_true=[1,1,1,0,0,0] with pos_label=0 is the mirror of the
    # known-value binary fixture (positive class index switches).
    # If a metric still hardcoded `y_proba[:, 1]`, it would return
    # `1 - correct_value` for ROC-AUC, PR-AUC, and Brier.
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    # column 0 = p(class 0) = p(positive); column 1 = p(class 1).
    y_proba = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.8, 0.2],
        ]
    )
    rec = compute_all(
        task_type="binary",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        pos_label=0,
        labels=np.array([0, 1]),
    )
    assert isinstance(rec, BinaryMetrics)
    # By construction this is the same outcome as the pos_label=1
    # known-value fixture (just with class labels flipped). The
    # pinned values must match exactly.
    assert rec.accuracy == pytest.approx(3 / 6)
    assert rec.precision_zd0 == pytest.approx(1 / 3)
    assert rec.recall_zd0 == pytest.approx(1 / 2)
    assert rec.f1_zd0 == pytest.approx(2 / 5)
    assert rec.roc_auc == pytest.approx(0.75)
    assert rec.pr_auc == pytest.approx(0.75)
    # Brier and ECE must NOT report `1 - x`: the inverted-column bug
    # would surface here as Brier ~0.78 instead of ~0.22.
    expected_brier = (
        (0.9 - 1) ** 2
        + (0.8 - 1) ** 2
        + (0.4 - 1) ** 2
        + (0.3 - 1) ** 2
        + (0.6 - 0) ** 2
        + (0.2 - 0) ** 2
    ) / 6
    assert rec.brier == pytest.approx(expected_brier)


def test_resolve_pos_index_raises_when_pos_label_missing_from_labels() -> None:
    # If the caller passes a labels array that doesn't contain
    # pos_label, the resolver must raise rather than silently
    # picking column 0 (sklearn `np.where` returns an empty array).
    with pytest.raises(ValueError, match="pos_label"):
        compute_all(
            task_type="binary",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_proba=np.array([[0.7, 0.3], [0.2, 0.8]]),
            pos_label="positive_but_string_not_in_labels",
            labels=np.array([0, 1]),
        )


# --- R1 / qa-C3: ECE on empty input returns nan. ---


def test_compute_ece_q15_returns_nan_on_empty_array() -> None:
    from benchmarks.metrics.classification import compute_ece_q15

    out = compute_ece_q15(
        np.array([], dtype=np.int64),
        np.empty((0, 2)),
        n_classes=2,
    )
    import math

    assert math.isnan(out)


def test_compute_ece_q15_multiclass_returns_nan_on_empty_array() -> None:
    # R2 / qa-I4: the n == 0 early return is shared between binary
    # and multiclass entries; the multiclass entry path builds
    # `probs = y_proba.max(axis=1)` and `argmax = y_proba.argmax(axis=1)`
    # before hitting the n == 0 guard. Pins that the guard fires
    # for the multiclass path too.
    from benchmarks.metrics.classification import compute_ece_q15

    out = compute_ece_q15(
        np.array([], dtype=np.int64),
        np.empty((0, 3)),
        n_classes=3,
    )
    import math

    assert math.isnan(out)


# --- R2 / qa-I1, qa-I2: `_resolve_pos_index` fallback path and
# `compute_roc_auc` pos_label=None branch. ---


def test_resolve_pos_index_uses_y_true_when_labels_is_none() -> None:
    # When `labels` is not supplied, `_resolve_pos_index` falls back
    # to `np.unique(y_true)`. With a clean binary fixture this still
    # returns the correct positive column, but if pos_label is not
    # in `y_true` either, the function must still raise.
    with pytest.raises(ValueError, match="pos_label"):
        compute_all(
            task_type="binary",
            y_true=np.array([0, 1, 0, 1]),
            y_pred=np.array([0, 1, 0, 1]),
            y_proba=np.array([[0.7, 0.3], [0.2, 0.8], [0.7, 0.3], [0.2, 0.8]]),
            pos_label=99,  # absent from np.unique(y_true) = [0, 1]
            labels=None,
        )


def test_compute_roc_auc_binary_pos_label_none_uses_y_true_directly() -> None:
    # R2 / qa-I2: `compute_roc_auc`'s direct-call `pos_label=None`
    # branch (the historical-behavior path) is exercised here.
    # `_compute_binary` always supplies pos_label, so this test
    # protects the lower-level API contract.
    from benchmarks.metrics.classification import compute_roc_auc

    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.2, 0.8],
        ]
    )
    out = compute_roc_auc(y_true, y_proba, n_classes=2, pos_label=None, pos_index=1)
    assert out == pytest.approx(0.75)


# --- R2 / qa-I3: regression_quantile path forwards the MAPE skip
# reason from `_compute_regression_quantile`. ---


def test_compute_all_regression_quantile_propagates_mape_skip_reason() -> None:
    y_true = np.array([1.0, 0.0, 3.0, 4.0])
    y_pred = np.array([1.1, 0.1, 3.0, 4.0])
    y_quantiles = np.stack([y_pred - 0.5, y_pred, y_pred + 0.5], axis=1)
    rec = compute_all(
        task_type="regression_quantile",
        y_true=y_true,
        y_pred=y_pred,
        y_quantiles=y_quantiles,
        quantiles=(0.1, 0.5, 0.9),
    )
    assert isinstance(rec, RegressionQuantileMetrics)
    import math

    assert math.isnan(rec.mape)
    assert rec.mape_skip_reason == "mape_undefined_zero_in_y_true"
