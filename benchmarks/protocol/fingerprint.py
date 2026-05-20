"""Canonical split fingerprint (B8.1).

Every benchmark run records a SHA-256 fingerprint of the
fold-index layout produced by the splitter. The fingerprint is
deterministic in the seed + splitter config + panel shape; an
identical config on the same panel produces the same fingerprint
(qa-C4 stability), and a single-row mutation flips it (qa-NEW-N1
`L_resolved + 1` direction-2 perturbation pin).

The fingerprint is the bridge between the run manifest (Phase B9)
and the reproducibility contract (B8.1): if two runs report the
same fingerprint, the leaderboard rows are directly comparable;
if different, the operator knows the runs split differently.
"""

import hashlib
from collections.abc import Iterable

import numpy as np


def fingerprint_folds(
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> str:
    """SHA-256-hex the canonical serialization of a fold iterator.

    Canonical serialization: per fold, write the sorted train
    positions then the sorted test positions, separated by single
    `\\x1e` (record separator) bytes; folds are separated by
    `\\x1d` (group separator) bytes. The byte stream is the
    deterministic input to the hash, so identical fold layouts
    produce identical fingerprints regardless of in-memory order.
    """
    hasher = hashlib.sha256()
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if fold_i > 0:
            hasher.update(b"\x1d")  # group separator between folds
        hasher.update(np.sort(train_idx.astype(np.int64)).tobytes())
        hasher.update(b"\x1e")  # record separator between train+test
        hasher.update(np.sort(test_idx.astype(np.int64)).tobytes())
    return hasher.hexdigest()
