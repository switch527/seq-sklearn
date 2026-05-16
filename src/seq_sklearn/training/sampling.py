"""Imbalance resampling index builders (per requirements F5).

The two sampler-side F5 strategies (``oversample_minority`` /
``undersample_majority``) are realized as pure functions returning a
permuted index array over the training set. The Trainer wraps the array
in a ``torch.utils.data.SubsetRandomSampler`` (or feeds it as the
dataset index) so the DataLoader yields the rebalanced stream; the
resampling math lives here, isolated and unit-testable.

All randomness flows through a caller-supplied ``numpy.random.Generator``
per the project determinism contract (N4): the estimator threads its
single ``seed`` into the generator so two fits with the same seed draw
the same resampled indices.

``binary`` / ``multiclass`` labels are 1-D integer class arrays.
``oversample_ratio`` scales the per-class target relative to the
majority count; ``replacement`` toggles whether minority draws sample
with replacement (the only way to exceed a class's own size).
"""

import logging

import numpy as np

__all__ = ["oversample_minority", "undersample_majority"]

logger = logging.getLogger(__name__)


def _class_indices(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Map each class value to the positions where it occurs.

    Raises:
        ValueError: ``labels`` is empty (no class to resample over).
    """
    if labels.size == 0:
        raise ValueError("labels is empty; nothing to resample")
    return {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}


def oversample_minority(
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    oversample_ratio: float = 1.0,
    replacement: bool = True,
) -> np.ndarray:
    """Return indices oversampling every non-majority class.

    Each class is resampled up to ``round(oversample_ratio *
    majority_count)`` indices. ``oversample_ratio=1.0`` lifts every
    minority class to the majority size (full balance); values > 1.0
    oversample past it. The majority class itself is kept whole and is
    only resampled when ``oversample_ratio != 1.0`` pushes its target
    away from its own count. The returned array is shuffled and is
    longer than ``labels`` whenever any minority class is undersized
    relative to the target.

    ``replacement=False`` caps a class draw at its own size, so a
    minority class cannot reach the target; this matches sklearn-style
    "no-replacement oversampling is a no-op past the class size".

    Raises:
        ValueError: ``labels`` is empty.
    """
    by_class = _class_indices(labels)
    majority = max(len(idx) for idx in by_class.values())
    target = round(oversample_ratio * majority)

    pieces: list[np.ndarray] = []
    for _cls, idx in sorted(by_class.items()):
        if len(idx) == majority and oversample_ratio == 1.0:
            # Majority class at the default ratio: kept whole (each
            # index exactly once), matching the docstring contract and
            # standard oversampling. Resampling it here would feed the
            # Phase 4b sampler duplicated/dropped majority indices.
            draw = idx
        elif replacement:
            draw = rng.choice(idx, size=target, replace=True)
        else:
            take = min(target, len(idx))
            draw = rng.choice(idx, size=take, replace=False)
        pieces.append(draw)

    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out


def undersample_majority(
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    replacement: bool = False,
) -> np.ndarray:
    """Return indices undersampling every non-minority class.

    Every class is cut to the minority class's count, so the result is
    shorter than ``labels`` whenever the data is imbalanced. With
    ``replacement=True`` the per-class draw samples with replacement (it
    still yields ``minority_count`` indices per class, just possibly with
    repeats). The returned array is shuffled.

    Raises:
        ValueError: ``labels`` is empty.
    """
    by_class = _class_indices(labels)
    minority = min(len(idx) for idx in by_class.values())

    pieces: list[np.ndarray] = []
    for _cls, idx in sorted(by_class.items()):
        draw = rng.choice(idx, size=minority, replace=replacement)
        pieces.append(draw)

    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out
