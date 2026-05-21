"""Phase B6 pairwise-metric known-value tests (B6.2.4).

Pins each diversity statistic + correlation against an inline-
arithmetic oracle so a reader can verify the formula by hand. The
three named degenerate cases from B6.2.4 are pinned separately:

- ``test_q_phi_all_agree_returns_one`` (qa-C6 oracle): two perfectly
  identical classifiers; Q == phi == 1.0 per the limiting-case
  convention.
- ``test_q_phi_half_agree_matches_arithmetic`` (qa-C6 oracle): a
  2x2 table with `N11=N00=2, N10=N01=2`; Q == phi == 0.
- ``test_phi_degenerate_non_perfect_agreement_is_nan`` (qa-NEW-I1):
  a collapsed marginal that is NOT perfect agreement; phi returns
  nan rather than 1.0.
"""

import math

import numpy as np
import pytest
from benchmarks.metrics.pairwise import (
    classification_nll,
    disagreement_rate,
    double_fault_rate,
    joint_counts,
    pearson_correlation,
    phi_coefficient,
    regression_signed_residual,
    spearman_correlation,
    yule_q,
)

# --- joint_counts: structural ------------------------------------------------


def test_joint_counts_basic() -> None:
    # 6 samples; both correct on row 0, A only on row 1, B only on
    # row 2, both wrong on rows 3-5.
    y_true = np.array([1, 1, 1, 1, 1, 1])
    y_pred_a = np.array([1, 1, 0, 0, 0, 0])
    y_pred_b = np.array([1, 0, 1, 0, 0, 0])
    n11, n10, n01, n00 = joint_counts(y_true, y_pred_a, y_pred_b)
    assert n11 == 1
    assert n10 == 1
    assert n01 == 1
    assert n00 == 3


def test_joint_counts_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        joint_counts(np.array([1, 1]), np.array([1]), np.array([1, 1]))


# --- Yule's Q + phi degenerate-agreement convention (B6.2.4) -----------------


def test_q_phi_all_agree_returns_one() -> None:
    # `N01 = N10 = 0` (perfect agreement) -> Q == phi == 1.0 by
    # the limiting-case convention in B6.2.4.
    assert yule_q(n11=5, n10=0, n01=0, n00=5) == 1.0
    assert phi_coefficient(n11=5, n10=0, n01=0, n00=5) == 1.0


def test_q_phi_all_disagree_matches_arithmetic() -> None:
    # `N11 = N00 = 0`; numerator = 0 - N01*N10 = -N01*N10; the Q
    # denominator is N01*N10, so Q = -1.
    # Phi has denom = sqrt(N10 * N01 * N01 * N10) = N01*N10; same
    # result: Phi = -1.
    assert yule_q(n11=0, n10=4, n01=4, n00=0) == pytest.approx(-1.0)
    assert phi_coefficient(n11=0, n10=4, n01=4, n00=0) == pytest.approx(-1.0)


def test_q_phi_half_agree_returns_zero() -> None:
    # Symmetric table: N11 = N00 = 2, N10 = N01 = 2.
    # Q: (4 - 4) / (4 + 4) = 0.
    # Phi: (4 - 4) / sqrt(4*4*4*4) = 0.
    assert yule_q(n11=2, n10=2, n01=2, n00=2) == 0.0
    assert phi_coefficient(n11=2, n10=2, n01=2, n00=2) == 0.0


def test_phi_degenerate_non_perfect_agreement_is_nan() -> None:
    # qa-NEW-I1: a collapsed marginal that is NOT perfect agreement
    # (one of the marginal sums is zero) returns nan for phi.
    # N11+N10 == 0 (A is wrong on every row); not perfect agreement
    # since N01 > 0. Phi denom -> 0; return nan.
    result = phi_coefficient(n11=0, n10=0, n01=3, n00=4)
    assert math.isnan(result)


def test_yule_q_degenerate_non_perfect_agreement_is_nan() -> None:
    # Q denominator: N11*N00 + N01*N10 = 0 happens when either
    # {N11=0 OR N00=0} AND {N01=0 OR N10=0}; specifically when
    # N00 = 0 AND N01 = 0 (model B correct on every row, not
    # perfect-agreement). The function returns nan.
    result = yule_q(n11=3, n10=4, n01=0, n00=0)
    assert math.isnan(result)


def test_q_phi_known_value() -> None:
    # Asymmetric non-degenerate case from a hand-computed table:
    # N11 = 3, N10 = 1, N01 = 2, N00 = 4. Total = 10.
    # Q numerator: 3*4 - 2*1 = 10
    # Q denominator: 3*4 + 2*1 = 14
    # Q = 10/14 = 5/7
    # Phi numerator: 10
    # Phi denominator: sqrt((3+1)(2+4)(3+2)(1+4)) = sqrt(4*6*5*5) = sqrt(600)
    # Phi = 10 / sqrt(600)
    n = {"n11": 3, "n10": 1, "n01": 2, "n00": 4}
    assert yule_q(**n) == pytest.approx(10 / 14)
    assert phi_coefficient(**n) == pytest.approx(10 / math.sqrt(600))


# --- disagreement / double-fault: structural ---------------------------------


def test_disagreement_rate_known_value() -> None:
    # N=10; off-diagonal N10+N01 = 1+2 = 3; rate = 0.3.
    assert disagreement_rate(n11=3, n10=1, n01=2, n00=4) == pytest.approx(0.3)


def test_double_fault_rate_known_value() -> None:
    # N=10; both wrong N00 = 4; rate = 0.4.
    assert double_fault_rate(n11=3, n10=1, n01=2, n00=4) == pytest.approx(0.4)


def test_disagreement_rate_raises_on_empty_table() -> None:
    with pytest.raises(ValueError, match="empty"):
        disagreement_rate(n11=0, n10=0, n01=0, n00=0)


def test_double_fault_rate_raises_on_empty_table() -> None:
    with pytest.raises(ValueError, match="empty"):
        double_fault_rate(n11=0, n10=0, n01=0, n00=0)


# --- Pearson / Spearman ------------------------------------------------------


def test_pearson_perfect_positive() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    assert pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_perfect_negative() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    assert pearson_correlation(x, y) == pytest.approx(-1.0)


def test_pearson_zero_variance_returns_nan() -> None:
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    assert math.isnan(pearson_correlation(x, y))


def test_pearson_too_short_returns_nan() -> None:
    assert math.isnan(pearson_correlation(np.array([1.0]), np.array([2.0])))


def test_spearman_strict_monotone_returns_one() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 200.0, 3000.0, 40000.0])
    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_spearman_with_ties() -> None:
    # average-rank tie-breaking (scipy convention)
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    # ranks: x -> [1, 2.5, 2.5, 4]; y -> [1, 3, 2, 4]
    # pearson on those is deterministic; we just pin >0.
    assert spearman_correlation(x, y) > 0.0


# --- classification_nll ------------------------------------------------------


def test_classification_nll_matches_inline_arithmetic() -> None:
    y_true = np.array([0, 1, 0])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
    labels = np.array([0, 1])
    nll = classification_nll(y_true, y_proba, labels)
    assert nll[0] == pytest.approx(-math.log(0.8))
    assert nll[1] == pytest.approx(-math.log(0.7))
    assert nll[2] == pytest.approx(-math.log(0.5))


def test_classification_nll_clips_zero_probability() -> None:
    # A literal 0.0 probability would give inf NLL; the clip keeps
    # the value finite (and large).
    y_true = np.array([0])
    y_proba = np.array([[0.0, 1.0]])
    labels = np.array([0, 1])
    nll = classification_nll(y_true, y_proba, labels)
    assert math.isfinite(nll[0])
    assert nll[0] > 30.0  # -log(1e-15) ≈ 34.5


def test_classification_nll_rejects_unseen_label() -> None:
    y_true = np.array([0, 1, 2])  # 2 not in labels
    y_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="labels not in `labels`"):
        classification_nll(y_true, y_proba, labels)


# --- regression_signed_residual ----------------------------------------------


def test_regression_signed_residual_is_y_minus_yhat() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.5, 2.5, 2.0])
    res = regression_signed_residual(y_true, y_pred)
    assert res[0] == pytest.approx(0.5)
    assert res[1] == pytest.approx(-0.5)
    assert res[2] == pytest.approx(1.0)
