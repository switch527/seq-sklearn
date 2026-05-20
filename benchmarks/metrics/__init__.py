"""Benchmark-suite metric layer (B5).

The harness routes every (dataset, model, fold) pair through
:func:`compute_all` and receives a typed :class:`MetricsRecord`
with exactly the metrics applicable to the task type. Missing
fields per task type are absent rather than `nan`; the type itself
documents which metrics apply.

Loss-first per B5: the primary loss is the rank key for the B6.1
leaderboard. Everything else is the secondary table.

The classification metric set carries the user-facing extension
(B5 delta R4): `accuracy_zd0`, `precision_*_zd0`, `recall_*_zd0`,
`f1_*_zd0` (suffixed multiclass averages + binary single value),
plus the strictly-proper Brier and the calibration `ece_q15`. The
regression set carries MAPE with three typed skip reasons per B5.2.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

from benchmarks.config import TaskType
from benchmarks.metrics import classification as _cls
from benchmarks.metrics import regression as _reg
from benchmarks.metrics.regression import MapeSkipReason

__all__ = [
    "BinaryMetrics",
    "MapeSkipReason",
    "MetricsRecord",
    "MulticlassMetrics",
    "RegressionPointMetrics",
    "RegressionQuantileMetrics",
    "compute_all",
]


class _BaseMetrics(BaseModel):
    """Common base for the per-task-type metrics records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BinaryMetrics(_BaseMetrics):
    """Metrics applicable when `task_type == "binary"`."""

    log_loss: float
    accuracy: float
    precision_zd0: float
    recall_zd0: float
    f1_zd0: float
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    mcc: float
    brier: float
    ece_q15: float


class MulticlassMetrics(_BaseMetrics):
    """Metrics applicable when `task_type == "multiclass"`."""

    log_loss: float
    accuracy: float
    precision_macro_zd0: float
    precision_weighted_zd0: float
    recall_macro_zd0: float
    recall_weighted_zd0: float
    f1_macro_zd0: float
    f1_weighted_zd0: float
    roc_auc_macro_ovr: float
    pr_auc_macro_ovr: float
    balanced_accuracy: float
    mcc: float
    brier_multiclass_mean: float
    ece_q15: float


class RegressionPointMetrics(_BaseMetrics):
    """Metrics applicable when `task_type == "regression_point"`."""

    rmse: float
    mae: float
    r2: float
    mape: float
    mape_skip_reason: MapeSkipReason | None


class RegressionQuantileMetrics(_BaseMetrics):
    """Metrics applicable when `task_type == "regression_quantile"`.

    `rmse` / `mae` / `r2` / `mape` are computed against the caller-
    supplied `y_pred` point prediction; `y_quantiles` drives only
    the `pinball` dict. The caller is responsible for selecting the
    median quantile (when one is configured at q=0.5) or any other
    derivation as the point prediction.
    """

    rmse: float
    mae: float
    r2: float
    mape: float
    mape_skip_reason: MapeSkipReason | None
    pinball: dict[str, float]


MetricsRecord = (
    BinaryMetrics | MulticlassMetrics | RegressionPointMetrics | RegressionQuantileMetrics
)


def compute_all(
    *,
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    y_quantiles: np.ndarray | None = None,
    quantiles: tuple[float, ...] | None = None,
    pos_label: int | str | None = None,
    labels: np.ndarray | None = None,
) -> MetricsRecord:
    """Compute every metric applicable to `task_type` and return the
    typed record.

    Raises:
        ValueError: required inputs for the task type are missing
            (e.g., binary task without `pos_label` or `y_proba`).
    """
    if task_type == "binary":
        if y_proba is None:
            raise ValueError("compute_all: binary task requires y_proba")
        if pos_label is None:
            raise ValueError("compute_all: binary task requires pos_label")
        return _compute_binary(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            pos_label=pos_label,
            labels=labels,
        )
    if task_type == "multiclass":
        if y_proba is None:
            raise ValueError("compute_all: multiclass task requires y_proba")
        return _compute_multiclass(y_true=y_true, y_pred=y_pred, y_proba=y_proba, labels=labels)
    if task_type == "regression_point":
        return _compute_regression_point(y_true=y_true, y_pred=y_pred)
    if task_type == "regression_quantile":
        if y_quantiles is None or quantiles is None:
            raise ValueError(
                "compute_all: regression_quantile task requires y_quantiles + quantiles"
            )
        return _compute_regression_quantile(
            y_true=y_true,
            y_pred=y_pred,
            y_quantiles=y_quantiles,
            quantiles=quantiles,
        )
    # pyright catches the exhaustive Literal so the assert is the
    # runtime canary for an enum-widening change.
    raise ValueError(f"compute_all: unsupported task_type={task_type!r}")


def _resolve_pos_index(labels: np.ndarray | None, y_true: np.ndarray, pos_label: int | str) -> int:
    """Return the column of `y_proba` that holds `p(positive_class)`.

    The harness's adapter follows the sklearn convention that the
    `predict_proba` columns are aligned with `classes_` (sorted
    ascending). When the dataset's `positive_label` is the smaller
    of the two classes (e.g., binary `0/1` with `positive_label=0`),
    `y_proba[:, 1]` is `p(negative)`, NOT `p(positive)`, so a naïve
    column-1 index silently inverts ROC-AUC / PR-AUC / Brier / ECE.

    Raises:
        ValueError: `pos_label` is not present in `labels` (or in
            `np.unique(y_true)` when `labels` is not provided).
    """
    label_arr = np.asarray(labels) if labels is not None else np.unique(y_true)
    matches = np.where(label_arr == pos_label)[0]
    if len(matches) == 0:
        raise ValueError(
            f"_resolve_pos_index: pos_label={pos_label!r} not found in "
            f"labels={label_arr.tolist()!r}"
        )
    return int(matches[0])


def _compute_binary(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    pos_label: int | str,
    labels: np.ndarray | None,
) -> BinaryMetrics:
    pos_index = _resolve_pos_index(labels, y_true, pos_label)
    p, r, f = _cls.compute_precision_recall_f1_binary(y_true, y_pred, pos_label=pos_label)
    return BinaryMetrics(
        log_loss=_cls.compute_log_loss(y_true, y_proba, labels=labels),
        accuracy=_cls.compute_accuracy(y_true, y_pred),
        precision_zd0=p,
        recall_zd0=r,
        f1_zd0=f,
        roc_auc=_cls.compute_roc_auc(
            y_true, y_proba, n_classes=2, pos_label=pos_label, pos_index=pos_index
        ),
        pr_auc=_cls.compute_pr_auc(
            y_true, y_proba, n_classes=2, pos_label=pos_label, pos_index=pos_index
        ),
        balanced_accuracy=_cls.compute_balanced_accuracy(y_true, y_pred),
        mcc=_cls.compute_mcc(y_true, y_pred),
        brier=_cls.compute_brier(
            y_true, y_proba, n_classes=2, pos_label=pos_label, pos_index=pos_index
        ),
        ece_q15=_cls.compute_ece_q15(
            y_true, y_proba, n_classes=2, pos_label=pos_label, pos_index=pos_index
        ),
    )


def _compute_multiclass(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: np.ndarray | None,
) -> MulticlassMetrics:
    prf = _cls.compute_precision_recall_f1_multiclass(y_true, y_pred)
    n_classes = int(y_proba.shape[1])
    return MulticlassMetrics(
        log_loss=_cls.compute_log_loss(y_true, y_proba, labels=labels),
        accuracy=_cls.compute_accuracy(y_true, y_pred),
        precision_macro_zd0=prf["precision_macro_zd0"],
        precision_weighted_zd0=prf["precision_weighted_zd0"],
        recall_macro_zd0=prf["recall_macro_zd0"],
        recall_weighted_zd0=prf["recall_weighted_zd0"],
        f1_macro_zd0=prf["f1_macro_zd0"],
        f1_weighted_zd0=prf["f1_weighted_zd0"],
        roc_auc_macro_ovr=_cls.compute_roc_auc(y_true, y_proba, n_classes=n_classes),
        pr_auc_macro_ovr=_cls.compute_pr_auc(y_true, y_proba, n_classes=n_classes),
        balanced_accuracy=_cls.compute_balanced_accuracy(y_true, y_pred),
        mcc=_cls.compute_mcc(y_true, y_pred),
        brier_multiclass_mean=_cls.compute_brier(y_true, y_proba, n_classes=n_classes),
        ece_q15=_cls.compute_ece_q15(y_true, y_proba, n_classes=n_classes),
    )


def _compute_regression_point(*, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionPointMetrics:
    mape, skip = _reg.compute_mape_or_skip(y_true, y_pred)
    return RegressionPointMetrics(
        rmse=_reg.compute_rmse(y_true, y_pred),
        mae=_reg.compute_mae(y_true, y_pred),
        r2=_reg.compute_r2(y_true, y_pred),
        mape=mape,
        mape_skip_reason=skip,
    )


def _compute_regression_quantile(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: tuple[float, ...],
) -> RegressionQuantileMetrics:
    mape, skip = _reg.compute_mape_or_skip(y_true, y_pred)
    pinball = _reg.compute_pinball_at_quantiles(y_true, y_quantiles, quantiles=quantiles)
    return RegressionQuantileMetrics(
        rmse=_reg.compute_rmse(y_true, y_pred),
        mae=_reg.compute_mae(y_true, y_pred),
        r2=_reg.compute_r2(y_true, y_pred),
        mape=mape,
        mape_skip_reason=skip,
        pinball=pinball,
    )
