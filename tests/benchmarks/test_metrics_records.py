"""Phase B4 record-shape tests (B5).

The four typed metric records carry exactly the metrics applicable to
their task type, with the user-facing extension (B5 delta R4) reflected
in suffixed multiclass names (`*_macro_zd0` / `*_weighted_zd0`) and
unsuffixed binary names (`precision_zd0` / `recall_zd0` / `f1_zd0`).
The regression point and quantile records carry the MAPE skip-reason
slot per B5.2.
"""

import numpy as np
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
    import pytest
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
