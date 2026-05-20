"""Phase B4 known-value metric tests (B5).

Three fixtures with the oracle arithmetic spelled out in comments so a
reader can verify each pinned value by hand:

- Binary 6-sample (4/2 class counts) with `pos_label=1`. Exercises
  `average="binary"` precision / recall / F1 and the binary Brier path.
- Multiclass 3-class 4-sample (supports 2/1/1) where `y_pred` never
  emits class 2, so the `zero_division=0` branch fires for class 2's
  precision / recall / F1 and `macro != weighted` falls out of the
  unequal supports.
- Regression 8-sample with three negative `y_true` entries and a
  constant +0.5 residual, so `RMSE`, `MAE`, `MAPE`, and pinball at
  q=0.5 each collapse to an inline-arithmetic formula.
"""

import math

import numpy as np
import pytest
from benchmarks.metrics import (
    BinaryMetrics,
    MulticlassMetrics,
    RegressionPointMetrics,
    compute_all,
)
from benchmarks.metrics.regression import compute_pinball_at_quantiles


def test_binary_known_values_match_inline_arithmetic() -> None:
    # y_true = [0,0,0,0,1,1] (4 negatives, 2 positives), pos_label=1
    # y_pred = [0,0,1,1,0,1]
    # Confusion vs pos_label=1: TP=1 (idx 5), FP=2 (idx 2,3),
    #                           TN=2 (idx 0,1), FN=1 (idx 4)
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
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

    rec = compute_all(
        task_type="binary",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        pos_label=1,
    )
    assert isinstance(rec, BinaryMetrics)

    # accuracy = (TP+TN)/N = (1+2)/6
    assert rec.accuracy == pytest.approx(3 / 6)
    # precision = TP/(TP+FP) = 1/(1+2)
    assert rec.precision_zd0 == pytest.approx(1 / 3)
    # recall    = TP/(TP+FN) = 1/(1+1)
    assert rec.recall_zd0 == pytest.approx(1 / 2)
    # f1 = 2*p*r/(p+r) = 2*(1/3)*(1/2)/(1/3 + 1/2) = (1/3) / (5/6) = 2/5
    assert rec.f1_zd0 == pytest.approx(2 / 5)
    # balanced_accuracy = (TPR+TNR)/2 = (1/2 + 2/4)/2 = 1/2
    assert rec.balanced_accuracy == pytest.approx(0.5)
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    #     = (1*2 - 2*1) / sqrt(3 * 2 * 4 * 3) = 0
    assert rec.mcc == pytest.approx(0.0)

    # log_loss = -(1/N) sum [y*log(p1) + (1-y)*log(1-p1)]
    # = -(1/6) * [log(0.9) + log(0.8) + log(0.4) + log(0.3) + log(0.4) + log(0.8)]
    expected_log_loss = (
        -(
            math.log(0.9)
            + math.log(0.8)
            + math.log(0.4)
            + math.log(0.3)
            + math.log(0.4)
            + math.log(0.8)
        )
        / 6
    )
    assert rec.log_loss == pytest.approx(expected_log_loss)

    # Brier (binary) = (1/N) sum (p1 - y)^2
    expected_brier = (
        (0.1 - 0) ** 2
        + (0.2 - 0) ** 2
        + (0.6 - 0) ** 2
        + (0.7 - 0) ** 2
        + (0.4 - 1) ** 2
        + (0.8 - 1) ** 2
    ) / 6
    assert rec.brier == pytest.approx(expected_brier)

    # ROC-AUC: positives have p1=[0.4, 0.8]; negatives have p1=[0.1, 0.2, 0.6, 0.7].
    # Pairs with positive's score > negative's score:
    #   (0.4 > 0.1)(0.4 > 0.2) (0.4 < 0.6, 0.7)
    #   (0.8 > 0.1, 0.2, 0.6, 0.7) -> 4
    # AUC = 6/8 = 0.75.
    assert rec.roc_auc == pytest.approx(0.75)

    # PR-AUC step-function: AP = sum (R_n - R_{n-1}) * P_n on sorted-by-score:
    #   (p1=0.8, y=1) -> TP=1, FP=0, P=1.0, R=0.5; contrib = 0.5 * 1.0 = 0.5
    #   (p1=0.7, y=0) -> TP=1, FP=1, P=0.5, R=0.5; contrib = 0
    #   (p1=0.6, y=0) -> TP=1, FP=2, P=1/3, R=0.5; contrib = 0
    #   (p1=0.4, y=1) -> TP=2, FP=2, P=0.5, R=1.0; contrib = 0.5 * 0.5 = 0.25
    #   (p1=0.2, y=0) -> R unchanged; contrib = 0
    #   (p1=0.1, y=0) -> R unchanged; contrib = 0
    # AP = 0.5 + 0.25 = 0.75
    assert rec.pr_auc == pytest.approx(0.75)


def test_multiclass_known_values_with_zero_division_and_macro_vs_weighted() -> None:
    # 3-class 4-sample. y_true supports (2, 1, 1); class 2 IS in y_true so
    # ROC-AUC OVR has a positive for each class. y_pred never emits class 2
    # so the zero_division=0 branch fires for class 2 in precision /
    # recall / F1.
    # y_true = [0, 0, 1, 2]; y_pred = [0, 0, 1, 1]
    # Per-class confusion (one-vs-rest):
    #   c=0: TP=2, FP=0, FN=0  -> P=1.0, R=1.0, F1=1.0
    #   c=1: TP=1, FP=1, FN=0  -> P=0.5, R=1.0, F1 = 2*0.5*1.0/1.5 = 2/3
    #   c=2: TP=0, FP=0, FN=1  -> P=0.0 (zero_division=0), R=0.0, F1=0.0
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.5, 0.4, 0.1],
            [0.3, 0.5, 0.2],
            [0.4, 0.5, 0.1],
        ]
    )

    rec = compute_all(
        task_type="multiclass",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=np.array([0, 1, 2]),
    )
    assert isinstance(rec, MulticlassMetrics)

    # accuracy = (TP_0 + TP_1 + TP_2) / N = (2+1+0)/4
    assert rec.accuracy == pytest.approx(3 / 4)

    # precision_macro = (1.0 + 0.5 + 0.0) / 3
    assert rec.precision_macro_zd0 == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    # precision_weighted: support weights (2, 1, 1) / 4
    # = (2*1.0 + 1*0.5 + 1*0.0) / 4 = 2.5 / 4
    assert rec.precision_weighted_zd0 == pytest.approx(2.5 / 4)
    # macro != weighted: this is what test_macro_vs_weighted_distinguishable
    # pins separately, but it must hold here too.
    assert rec.precision_macro_zd0 != pytest.approx(rec.precision_weighted_zd0)

    # recall_macro = (1.0 + 1.0 + 0.0) / 3 = 2/3
    assert rec.recall_macro_zd0 == pytest.approx(2 / 3)
    # recall_weighted = (2*1.0 + 1*1.0 + 1*0.0) / 4 = 3/4
    assert rec.recall_weighted_zd0 == pytest.approx(3 / 4)

    # f1_macro = (1.0 + 2/3 + 0.0) / 3 = 5/9
    assert rec.f1_macro_zd0 == pytest.approx(5 / 9)
    # f1_weighted = (2*1.0 + 1*(2/3) + 1*0.0) / 4 = (2 + 2/3) / 4 = 2/3
    assert rec.f1_weighted_zd0 == pytest.approx(2 / 3)

    # log_loss = -(1/N) sum log(p[y_i])
    # = -(1/4) * [log(0.7) + log(0.5) + log(0.5) + log(0.1)]
    expected_log_loss = -(math.log(0.7) + math.log(0.5) + math.log(0.5) + math.log(0.1)) / 4
    assert rec.log_loss == pytest.approx(expected_log_loss)

    # Brier multiclass mean = mean_k brier_score_loss(y_true == k, y_proba[:, k]).
    # c=0 OVR: y_k=[1,1,0,0], p=[0.7,0.5,0.3,0.4]
    #   mean((1-0.7)^2 + (1-0.5)^2 + (0-0.3)^2 + (0-0.4)^2) / 4 = 0.59/4
    # c=1 OVR: y_k=[0,0,1,0], p=[0.2,0.4,0.5,0.5]
    #   mean(0.04 + 0.16 + 0.25 + 0.25) / 4 = 0.70/4
    # c=2 OVR: y_k=[0,0,0,1], p=[0.1,0.1,0.2,0.1]
    #   mean(0.01 + 0.01 + 0.04 + 0.81) / 4 = 0.87/4
    brier_0 = (0.09 + 0.25 + 0.09 + 0.16) / 4
    brier_1 = (0.04 + 0.16 + 0.25 + 0.25) / 4
    brier_2 = (0.01 + 0.01 + 0.04 + 0.81) / 4
    expected_brier_mean = (brier_0 + brier_1 + brier_2) / 3
    assert rec.brier_multiclass_mean == pytest.approx(expected_brier_mean)

    # balanced_accuracy = macro-recall across all classes in y_true.
    # Classes in y_true are {0, 1, 2}; recalls (1.0, 1.0, 0.0); mean = 2/3.
    assert rec.balanced_accuracy == pytest.approx(2 / 3)


def test_regression_point_known_values_match_inline_arithmetic() -> None:
    # 8-sample. Three negative y_true (-3, -5, -7) per the plan;
    # constant +0.5 residual everywhere so RMSE=MAE=0.5 trivially.
    # y_true never zero so MAPE has a real denominator on every row.
    y_true = np.array([1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0], dtype=np.float64)
    y_pred = y_true + 0.5

    rec = compute_all(task_type="regression_point", y_true=y_true, y_pred=y_pred)
    assert isinstance(rec, RegressionPointMetrics)

    # MAE = mean(|y - p|) = 0.5
    assert rec.mae == pytest.approx(0.5)
    # RMSE = sqrt(mean((y - p)^2)) = sqrt(0.25) = 0.5
    assert rec.rmse == pytest.approx(0.5)

    # R^2 = 1 - SS_res / SS_tot
    # SS_res = 8 * 0.25 = 2
    # y_mean = (1+2-3+4-5+6-7+8)/8 = 6/8 = 0.75
    # SS_tot = sum (y - y_mean)^2
    y_mean = (1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) / 8
    ss_tot = sum((y - y_mean) ** 2 for y in [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0])
    expected_r2 = 1 - 2 / ss_tot
    assert rec.r2 == pytest.approx(expected_r2)

    # MAPE = mean(|y - p| / |y|) = mean(0.5/|y|)
    expected_mape = (
        0.5 / 1 + 0.5 / 2 + 0.5 / 3 + 0.5 / 4 + 0.5 / 5 + 0.5 / 6 + 0.5 / 7 + 0.5 / 8
    ) / 8
    assert rec.mape == pytest.approx(expected_mape)
    assert rec.mape_skip_reason is None


def test_pr_auc_multiclass_ovr_macro_matches_inline_arithmetic() -> None:
    # R1 / qa-I1: the multiclass per-class PR-AUC loop in
    # compute_pr_auc had no oracle test before this. The fixture
    # below produces three distinct per-class APs whose macro mean
    # is hand-computed below.
    #
    # y_true = [0, 0, 1, 1, 2, 2]; y_proba designed so each per-class
    # OVR ranking is non-trivial (a negative outranks at least one
    # positive in classes 0 and 1; class 2's lower positive sits at
    # the bottom of the sort).
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_proba = np.array(
        [
            [0.50, 0.20, 0.30],
            [0.25, 0.42, 0.33],
            [0.45, 0.25, 0.30],
            [0.30, 0.45, 0.25],
            [0.20, 0.70, 0.10],
            [0.60, 0.05, 0.35],
        ]
    )
    # Class 0 OVR ranking (score desc):
    #   (0.6, 0), (0.5, 1), (0.45, 0), (0.3, 0), (0.25, 1), (0.2, 0)
    # AP_0 = (0.5-0)*0.5 + (1-0.5)*(2/5) = 0.25 + 0.2 = 0.45
    #
    # Class 1 OVR ranking:
    #   (0.7, 0), (0.45, 1), (0.42, 0), (0.25, 1), (0.2, 0), (0.05, 0)
    # AP_1 = (0.5-0)*0.5 + (1-0.5)*0.5 = 0.25 + 0.25 = 0.5
    #
    # Class 2 OVR ranking:
    #   (0.35, 1), (0.33, 0), (0.3, 0), (0.3, 0), (0.25, 0), (0.1, 1)
    # AP_2 = (0.5-0)*1 + (1-0.5)*(2/6) = 0.5 + 1/6 = 2/3
    #
    # macro = (0.45 + 0.5 + 2/3)/3 = 97/180
    from benchmarks.metrics.classification import compute_pr_auc

    out = compute_pr_auc(y_true, y_proba, n_classes=3)
    assert out == pytest.approx((0.45 + 0.5 + 2 / 3) / 3)
    assert out == pytest.approx(97 / 180)


def test_pinball_at_q50_matches_inline_formula() -> None:
    # With y - p = -0.5 everywhere and alpha = 0.5,
    # pinball = mean(max(alpha * (y-p), (1-alpha) * (p-y)))
    #        = mean(max(-0.25, 0.25)) = 0.25.
    y_true = np.array([1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0], dtype=np.float64)
    y_pred = y_true + 0.5  # over-predicts by 0.5 on every row
    y_quantiles = y_pred.reshape(-1, 1)
    out = compute_pinball_at_quantiles(y_true, y_quantiles, quantiles=(0.5,))
    assert out == {"pinball_q0.50": pytest.approx(0.25)}
