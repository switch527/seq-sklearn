"""Pairwise diversity statistics + error correlation (B6.2).

The B6 ensemble experiment consumes two classifiers' (or two
regressors') predictions on the same fold and computes the
diversity statistics named in the design's B6.2.4 contract:

- Yule's Q
- Disagreement rate
- Double-fault rate
- Phi (Matthews) correlation

These are computed on the joint correct/incorrect 2x2 table
`N11/N10/N01/N00` of two CLASSIFIERS' hard predictions. The
degenerate-agreement convention is pinned: when the two classifiers
agree perfectly (`N01 = N10 = 0`), Q and Phi are defined as `1.0`
(the limit of both statistics as predictions coincide); when the
denominator collapses for any other reason (e.g., one classifier is
100% correct on the fold), the statistic is `nan` and the caller is
expected to exclude it from aggregation.

For B6.2.2 (prediction agreement) and B6.2.3 (error correlation)
the module exports Pearson + Spearman correlation helpers and
per-sample error functions (classification NLL, regression signed
residual).
"""

import math

import numpy as np

# Joint-counts shape: a 2x2 table where rows index classifier A's
# correctness and columns index classifier B's correctness. The four
# cells (N11, N10, N01, N00) cover (both correct, A correct B wrong,
# A wrong B correct, both wrong) respectively.


def joint_counts(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> tuple[int, int, int, int]:
    """Return `(N11, N10, N01, N00)` for two classifiers on the same
    target vector.

    The hard-prediction convention: a classifier is "correct" on a
    row when `y_pred == y_true`. Multi-class is handled directly
    (each row contributes to exactly one of the four cells).

    Raises:
        ValueError: array length mismatch.
    """
    n = len(y_true)
    if len(y_pred_a) != n or len(y_pred_b) != n:
        raise ValueError(
            f"joint_counts: array length mismatch: y_true={n}, "
            f"y_pred_a={len(y_pred_a)}, y_pred_b={len(y_pred_b)}"
        )
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    return n11, n10, n01, n00


def yule_q(*, n11: int, n10: int, n01: int, n00: int) -> float:
    """Yule's Q on the 2x2 joint-correctness table.

    `Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)`.

    Degenerate convention (B6.2.4): when `N01 == 0 AND N10 == 0`
    (perfect agreement), Q is defined as `1.0` (the limit of Q as
    the two classifiers' predictions coincide); when the denominator
    is otherwise zero, returns `nan`.
    """
    if n01 == 0 and n10 == 0:
        return 1.0
    denom = n11 * n00 + n01 * n10
    if denom == 0:
        return float("nan")
    return (n11 * n00 - n01 * n10) / denom


def phi_coefficient(*, n11: int, n10: int, n01: int, n00: int) -> float:
    """Phi (Matthews) correlation on the 2x2 joint-correctness table.

    `phi = (N11*N00 - N01*N10) / sqrt((N11+N10)(N01+N00)(N11+N01)(N10+N00))`.

    Degenerate convention (B6.2.4): when `N01 == 0 AND N10 == 0`,
    Phi is defined as `1.0`; otherwise when the square-root
    denominator is zero (one classifier is 100% correct or
    incorrect on the fold), returns `nan`.
    """
    if n01 == 0 and n10 == 0:
        return 1.0
    denom_sq = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom_sq == 0:
        return float("nan")
    return (n11 * n00 - n01 * n10) / math.sqrt(denom_sq)


def disagreement_rate(*, n11: int, n10: int, n01: int, n00: int) -> float:
    """Fraction of rows where the two classifiers disagree.

    `disagreement = (N01 + N10) / N`. Always defined for `N > 0`.

    Raises:
        ValueError: `N == 0` (all four cells zero).
    """
    n = n11 + n10 + n01 + n00
    if n == 0:
        raise ValueError("disagreement_rate: empty table (N=0)")
    return (n01 + n10) / n


def double_fault_rate(*, n11: int, n10: int, n01: int, n00: int) -> float:
    """Fraction of rows where BOTH classifiers are wrong.

    `double_fault = N00 / N`. Always defined for `N > 0`.

    Raises:
        ValueError: `N == 0` (all four cells zero).
    """
    n = n11 + n10 + n01 + n00
    if n == 0:
        raise ValueError("double_fault_rate: empty table (N=0)")
    return n00 / n


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two 1-D vectors.

    Returns `nan` when fewer than 2 samples or when either vector
    has zero variance (sklearn's correlation is undefined; the
    `np.corrcoef` warning is suppressed by checking first).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if x.std(ddof=0) == 0.0 or y.std(ddof=0) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D vectors.

    Computed as Pearson on the rank vectors so the degeneracy
    behavior matches `pearson_correlation`. Ties are broken by
    average-rank (sklearn / scipy convention).
    """
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return pearson_correlation(rankdata(x), rankdata(y))


def classification_nll(y_true: np.ndarray, y_proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample negative log-likelihood `-log p(y_true)` (B6.2.3).

    `y_proba` columns are aligned with `labels` (sorted classes
    matching sklearn's `classes_` convention). Probabilities are
    clipped to `[eps, 1.0]` so a literal 0.0 does not produce `inf`.

    Raises:
        ValueError: `y_true` contains a label not in `labels`.
    """
    label_arr = np.asarray(labels)
    eps = 1e-15
    p_true = np.zeros(len(y_true), dtype=np.float64)
    matched = np.zeros(len(y_true), dtype=bool)
    for k, label in enumerate(label_arr):
        m = y_true == label
        p_true[m] = y_proba[m, k]
        matched |= m
    if not matched.all():
        missing = np.unique(np.asarray(y_true)[~matched])
        raise ValueError(
            f"classification_nll: y_true contains labels not in `labels`: {missing.tolist()!r}"
        )
    p_true = np.clip(p_true, eps, 1.0)
    return -np.log(p_true)


def regression_signed_residual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample signed residual `y_true - y_pred` (B6.2.3).

    The pearson correlation of two models' residuals is the direct
    "decorrelated errors" evidence the B6.2 design names.
    """
    return np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
