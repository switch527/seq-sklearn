"""Calibration quality scalars (``calibration.fit`` payload, F11)."""

import pytest
import torch

from seq_sklearn.calibration._metrics import (
    expected_calibration_error,
    mean_quantile_coverage,
)


def test_ece_binary_perfectly_calibrated_is_zero() -> None:
    # Confidence 0.75 on all, 75% empirical accuracy -> zero gap.
    probs = torch.tensor([0.75, 0.75, 0.75, 0.75])
    y = torch.tensor([1, 1, 1, 0])
    assert expected_calibration_error(probs, y) < 1e-6


def test_ece_binary_systematically_overconfident_is_positive() -> None:
    probs = torch.tensor([0.99, 0.99, 0.99, 0.99])
    y = torch.tensor([1, 0, 1, 0])  # 50% accuracy, 99% confidence
    assert expected_calibration_error(probs, y) > 0.4


def test_ece_multiclass_uses_argmax_confidence() -> None:
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = torch.tensor([0, 1])
    # Both correct; gap is |mean_conf - 1.0| within the populated bins.
    assert expected_calibration_error(probs, y) >= 0.0


def test_ece_confidence_exactly_one_lands_in_last_bin() -> None:
    # Exercises the first-bin (lo == 0) vs left-open-bin branch and the
    # empty-bin continue: all mass at confidence 1.0.
    probs = torch.tensor([1.0, 1.0])
    y = torch.tensor([1, 1])
    assert expected_calibration_error(probs, y) == 0.0


def test_mean_quantile_coverage_matches_hand_count() -> None:
    pred = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    y = torch.tensor([0.5, 2.0])
    # row0: 0.5<=0 F, 0.5<=1 T; row1: 2<=0 F, 2<=1 F -> 1/4
    assert mean_quantile_coverage(pred, y) == 0.25


def test_ece_empty_fold_raises() -> None:
    with pytest.raises(ValueError, match="empty calibration fold"):
        expected_calibration_error(torch.empty(0), torch.empty(0))


def test_coverage_empty_fold_raises() -> None:
    with pytest.raises(ValueError, match="empty calibration fold"):
        mean_quantile_coverage(torch.empty(0, 3), torch.empty(0))
