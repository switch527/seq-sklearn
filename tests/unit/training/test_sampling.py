"""Imbalance-resampling index tests (per requirements F5).

Oversample produces a longer index array than the input; undersample
produces a shorter one; the configured ratios and replacement toggles
are respected; results are deterministic under a seeded generator.
"""

import numpy as np
import pytest

from seq_sklearn.training.sampling import oversample_minority, undersample_majority


def _imbalanced_labels() -> np.ndarray:
    # 90 of class 0, 10 of class 1.
    return np.array([0] * 90 + [1] * 10)


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_oversample_is_longer_and_balances_classes() -> None:
    labels = _imbalanced_labels()
    idx = oversample_minority(labels, _rng(), oversample_ratio=1.0)
    assert idx.shape[0] > labels.shape[0]
    resampled = labels[idx]
    counts = np.bincount(resampled)
    # ratio=1.0 lifts the minority to the majority count (90 each).
    assert counts[0] == 90
    assert counts[1] == 90


def test_oversample_ratio_above_one_scales_target() -> None:
    labels = _imbalanced_labels()
    idx = oversample_minority(labels, _rng(), oversample_ratio=2.0)
    counts = np.bincount(labels[idx])
    assert counts[0] == 180
    assert counts[1] == 180


def test_oversample_without_replacement_caps_at_class_size() -> None:
    labels = _imbalanced_labels()
    idx = oversample_minority(labels, _rng(), oversample_ratio=1.0, replacement=False)
    counts = np.bincount(labels[idx])
    # Minority cannot exceed its own 10 without replacement.
    assert counts[0] == 90
    assert counts[1] == 10


def test_undersample_is_shorter_and_balances_classes() -> None:
    labels = _imbalanced_labels()
    idx = undersample_majority(labels, _rng())
    assert idx.shape[0] < labels.shape[0]
    counts = np.bincount(labels[idx])
    # Both classes cut to the minority count (10 each).
    assert counts[0] == 10
    assert counts[1] == 10


def test_undersample_with_replacement_keeps_minority_count() -> None:
    labels = _imbalanced_labels()
    idx = undersample_majority(labels, _rng(), replacement=True)
    counts = np.bincount(labels[idx])
    assert counts[0] == 10
    assert counts[1] == 10


def test_oversample_is_deterministic_under_same_seed() -> None:
    labels = _imbalanced_labels()
    a = oversample_minority(labels, _rng())
    b = oversample_minority(labels, _rng())
    assert np.array_equal(a, b)


def test_undersample_is_deterministic_under_same_seed() -> None:
    labels = _imbalanced_labels()
    a = undersample_majority(labels, _rng())
    b = undersample_majority(labels, _rng())
    assert np.array_equal(a, b)


def test_oversample_balanced_input_is_a_noop_in_size() -> None:
    labels = np.array([0, 1, 0, 1])
    idx = oversample_minority(labels, _rng(), oversample_ratio=1.0)
    assert idx.shape[0] == labels.shape[0]


def test_oversample_three_class_balances_all_classes() -> None:
    # Exercises the per-class loop body three times, not just two.
    labels = np.array([0] * 100 + [1] * 30 + [2] * 10)
    idx = oversample_minority(labels, _rng(), oversample_ratio=1.0)
    counts = np.bincount(labels[idx])
    assert counts.tolist() == [100, 100, 100]


def test_undersample_three_class_cuts_all_classes() -> None:
    labels = np.array([0] * 100 + [1] * 30 + [2] * 10)
    idx = undersample_majority(labels, _rng())
    counts = np.bincount(labels[idx])
    assert counts.tolist() == [10, 10, 10]


def test_oversample_single_class_is_size_preserving_noop() -> None:
    # Degenerate fold: one class. majority == the only class, target ==
    # its own count, so the result is a same-size shuffled resample.
    labels = np.array([0] * 8)
    idx = oversample_minority(labels, _rng(), oversample_ratio=1.0)
    assert idx.shape[0] == 8
    assert np.array_equal(np.unique(labels[idx]), np.array([0]))


def test_undersample_single_class_is_size_preserving_noop() -> None:
    labels = np.array([0] * 8)
    idx = undersample_majority(labels, _rng())
    assert idx.shape[0] == 8
    assert np.array_equal(np.unique(labels[idx]), np.array([0]))


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_oversample_target_matches_ratio_times_majority(ratio: float) -> None:
    labels = _imbalanced_labels()
    idx = oversample_minority(labels, _rng(), oversample_ratio=ratio)
    counts = np.bincount(labels[idx])
    expected = round(ratio * 90)
    assert counts[0] == expected
    assert counts[1] == expected
