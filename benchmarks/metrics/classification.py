"""Classification metrics (B5.1).

Every metric is a thin wrapper over an `sklearn.metrics` call (or a
documented formula) so a known-value unit test has a non-arbitrary
oracle. Loss-first per B5: `log_loss` is the primary; everything
else is the secondary table.

Naming convention: each name encodes its convention so a leaderboard
reader cannot mistake one metric for another:

- `precision_zd0` (binary, `average="binary"`, `zero_division=0`)
- `precision_macro_zd0` / `precision_weighted_zd0` (multiclass)
- analog for `recall_*_zd0` and `f1_*_zd0`
- `ece_q15` (expected calibration error, 15 equal-mass quantile bins)

The functions accept the integer-encoded `y_true` / `y_pred` and the
`(N, n_classes)` `y_proba`. The harness calls these per-fold; the
aggregator computes the bootstrap CI per B5.4.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_log_loss(
    y_true: np.ndarray, y_proba: np.ndarray, *, labels: np.ndarray | None = None
) -> float:
    """Primary classification loss per B5.1: natural-log negative-
    log-likelihood, normalized mean. `labels` passed explicitly so
    absent classes do not shift the value."""
    return float(log_loss(y_true, y_proba, labels=labels))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_precision_recall_f1_binary(
    y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int | str
) -> tuple[float, float, float]:
    """Returns `(precision_zd0, recall_zd0, f1_zd0)` for binary.

    `pos_label` is the value the dataset's `positive_label` field
    declares (B2.2). `zero_division=0` encoded as `_zd0` in the
    metric names.
    """
    # sklearn stubs over-restrict `pos_label` (int) and `zero_division`
    # (str), but the runtime accepts `int | str` and `0 | 1 | "warn"`
    # respectively per the sklearn docs. Per-arg ignores keep the
    # public contract honest without widening the stub-typed call sites.
    p = float(
        precision_score(
            y_true,
            y_pred,
            pos_label=pos_label,  # pyright: ignore[reportArgumentType]
            average="binary",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )
    r = float(
        recall_score(
            y_true,
            y_pred,
            pos_label=pos_label,  # pyright: ignore[reportArgumentType]
            average="binary",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )
    f = float(
        f1_score(
            y_true,
            y_pred,
            pos_label=pos_label,  # pyright: ignore[reportArgumentType]
            average="binary",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
    )
    return p, r, f


def compute_precision_recall_f1_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Returns the suffixed names per B5.1:

    `precision_macro_zd0`, `precision_weighted_zd0`,
    `recall_macro_zd0`, `recall_weighted_zd0`,
    `f1_macro_zd0`, `f1_weighted_zd0`.

    `_macro` is unweighted-mean across classes; `_weighted` is
    weighted by class support. Each name is reported under both
    averages so a reader cannot mistake one for the other.
    """
    out: dict[str, float] = {}
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}_zd0"] = float(
            precision_score(
                y_true,
                y_pred,
                average=avg,
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        )
        out[f"recall_{avg}_zd0"] = float(
            recall_score(
                y_true,
                y_pred,
                average=avg,
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        )
        out[f"f1_{avg}_zd0"] = float(
            f1_score(
                y_true,
                y_pred,
                average=avg,
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        )
    return out


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, *, n_classes: int) -> float:
    """Binary: `roc_auc_score(y_true, y_proba[:, 1])`.

    Multiclass: `multi_class="ovr"`, `average="macro"` per B5.1.
    """
    if n_classes == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))


def compute_pr_auc(
    y_true: np.ndarray, y_proba: np.ndarray, *, n_classes: int, pos_label: int | str
) -> float:
    """Average precision (PR-AUC), sklearn step-function definition,
    not trapezoidal (B5.1 / qa-C5).

    Binary: `average_precision_score(y_true, y_proba[:, pos_index],
    pos_label=pos_label)`. Multiclass: one-vs-rest macro average
    across classes.
    """
    if n_classes == 2:
        # `pos_label` is the *value* of the positive class in
        # `y_true`; sklearn passes it through unchanged.
        return float(
            average_precision_score(
                y_true,
                y_proba[:, 1],
                pos_label=pos_label,  # pyright: ignore[reportArgumentType]
            )
        )
    # Multiclass: macro-OVR average per the design.
    scores: list[float] = []
    for k in range(n_classes):
        y_true_k = (y_true == k).astype(np.int64)
        scores.append(float(average_precision_score(y_true_k, y_proba[:, k])))
    return float(np.mean(scores))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray, *, n_classes: int) -> float:
    """Brier score per B5.1.

    Binary: `brier_score_loss(y_true, y_proba[:, 1])`. Multiclass:
    unweighted mean over classes of the one-vs-rest per-class
    `brier_score_loss`, i.e.
    ``mean_k brier_score_loss(y_true == k, y_proba[:, k])``
    (no extra division by K beyond that mean; normalized by N
    inside each call). Pinned by qa-NEW-C1.
    """
    if n_classes == 2:
        return float(brier_score_loss(y_true, y_proba[:, 1]))
    per_class: list[float] = []
    for k in range(n_classes):
        per_class.append(float(brier_score_loss((y_true == k).astype(np.int64), y_proba[:, k])))
    return float(np.mean(per_class))


def compute_ece_q15(y_true: np.ndarray, y_proba: np.ndarray, *, n_classes: int) -> float:
    """Expected calibration error with 15 equal-mass quantile bins
    per B5.1.

    For binary: bin by `y_proba[:, 1]` and per bin compute
    |mean(prob) - frac(y_true == 1)|, weighted by bin size.

    For multiclass: bin by the max-class probability and per bin
    compute |mean(top-prob) - accuracy|, weighted by bin size.
    """
    if n_classes == 2:
        probs = y_proba[:, 1]
        correct = y_true.astype(np.float64)
    else:
        probs = y_proba.max(axis=1)
        argmax = y_proba.argmax(axis=1)
        correct = (argmax == y_true).astype(np.float64)

    n = len(probs)
    if n == 0:
        return float("nan")
    # 15 equal-mass bins via quantile split.
    quantiles = np.linspace(0.0, 1.0, 16)[1:-1]
    bin_edges = np.quantile(probs, quantiles)
    bin_ids = np.digitize(probs, bin_edges)
    ece = 0.0
    for b in range(15):
        mask = bin_ids == b
        if not mask.any():
            continue
        weight = float(mask.sum()) / n
        gap = abs(float(probs[mask].mean()) - float(correct[mask].mean()))
        ece += weight * gap
    return float(ece)
