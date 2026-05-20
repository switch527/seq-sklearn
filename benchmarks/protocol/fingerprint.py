"""Canonical split fingerprint (B8.1).

Every benchmark run records a SHA-256 fingerprint of the
fold-index layout produced by the splitter. Per the design's B8.1
contract (`docs/benchmark_suite_design.md:569-572`):

> The split fingerprint is defined as the SHA-256 of the
> canonical-JSON list of ``(fold_index, sorted(train_idx),
> sorted(test_idx))`` produced by ``EntityTimeSeriesSplit`` for
> that ``(dataset, L_resolved)``.

Canonical JSON: compact separators ``","`` / ``":"`` and ascending
key order. Indices are sorted on the way in, so a caller who passes
unsorted arrays produces the same fingerprint as one who pre-sorts
(qa order-independence pin).

The fingerprint is the bridge between the run manifest (Phase B9)
and the reproducibility contract (B8.1): two runs with the same
fingerprint produced identical fold layouts and the leaderboard
rows are directly comparable.
"""

import hashlib
import json
from collections.abc import Iterable

import numpy as np


def fingerprint_folds(
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> str:
    """SHA-256 hex of the canonical-JSON serialization of a fold
    iterator.

    Per B8.1: ``json.dumps([[fold_index, sorted_train, sorted_test],
    ...], separators=(",", ":"))`` is the canonical form, then
    SHA-256-hex.

    Raises:
        ValueError: ``folds`` is empty. An empty iterator would
            silently return ``sha256("[]")``, which is
            indistinguishable from any other empty-fold run; loud
            failure is the right shape.
    """
    payload: list[list[int | list[int]]] = []
    for fold_index, (train_idx, test_idx) in enumerate(folds):
        payload.append(
            [
                fold_index,
                sorted(int(i) for i in np.asarray(train_idx).tolist()),
                sorted(int(i) for i in np.asarray(test_idx).tolist()),
            ]
        )
    if not payload:
        raise ValueError(
            "fingerprint_folds: no folds produced; check the splitter "
            "configuration (n_splits and the panel's per-entity row "
            "counts)"
        )
    canonical = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
