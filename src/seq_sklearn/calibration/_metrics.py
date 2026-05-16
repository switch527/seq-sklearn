"""Calibration quality scalars for the ``calibration.fit`` payload.

Requirements F11 specifies the ``calibration.fit`` record carries
``pre_ece`` / ``post_ece`` for classifiers and ``pre_coverage`` /
``post_coverage`` for regressors. These helpers compute those scalars
so each calibrator can emit the event from its own ``fit`` (the A9
contract that a calibrator is exercisable standalone, no estimator).
"""

from itertools import pairwise

import torch
from torch import Tensor

__all__ = ["expected_calibration_error", "mean_quantile_coverage"]

_ECE_BINS = 15


def expected_calibration_error(probs: Tensor, y_true: Tensor) -> float:
    """Equal-width binned ECE over the predicted-class confidence.

    ``probs`` is ``(N,)`` positive-class probability for binary or
    ``(N, K)`` per-class probability for multiclass; ``y_true`` is
    ``(N,)`` integer labels. The estimator is the standard
    confidence-vs-accuracy gap binned into :data:`_ECE_BINS` equal-width
    bins on ``[0, 1]`` and weighted by bin population.
    """
    probs = probs.detach().cpu().to(torch.float64)
    y_true = y_true.detach().cpu().reshape(-1)
    if probs.shape[0] == 0:
        raise ValueError("expected_calibration_error: empty calibration fold")
    if probs.ndim == 1:
        confidence = torch.maximum(probs, 1.0 - probs)
        predictions = (probs >= 0.5).long()
    else:
        confidence, predictions = probs.max(dim=1)
    correct = (predictions == y_true.long()).to(torch.float64)
    n = correct.shape[0]
    edges = torch.linspace(0.0, 1.0, _ECE_BINS + 1, dtype=torch.float64)
    ece = 0.0
    for lo, hi in pairwise(edges):
        # Left-open bins except the first so confidence == 1.0 lands in
        # the last bin and every sample is counted exactly once.
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else confidence <= hi
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
    return ece


def mean_quantile_coverage(pred_quantiles: Tensor, y_true: Tensor) -> float:
    """Mean empirical coverage ``P(y <= q_j)`` across quantile columns.

    ``pred_quantiles`` is ``(N, Q)`` predicted quantile values, ``y_true``
    is ``(N,)``. A perfectly calibrated regressor has per-column coverage
    equal to its nominal level; the scalar reported here is the mean of
    those empirical coverages so a single number can sit in the F11
    payload (the per-column comparison is the calibrator's own job).
    """
    pred = pred_quantiles.detach().cpu().to(torch.float64)
    if pred.shape[0] == 0:
        raise ValueError("mean_quantile_coverage: empty calibration fold")
    y = y_true.detach().cpu().reshape(-1, 1).to(torch.float64)
    return float((y <= pred).to(torch.float64).mean())
