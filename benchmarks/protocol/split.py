"""Per-entity time-expanding split wrapper (B4.1).

Thin shim over :class:`seq_sklearn.EntityTimeSeriesSplit` (A9.1) that
binds the splitter to a :class:`DatasetSpec`. The splitter reads
``spec.entity_col`` / ``spec.time_col`` so a single function call
produces the canonical per-fold (train_idx, test_idx) iterator the
harness consumes; every adapter family (seq-sklearn, GBM,
TSC) sees the SAME folds and the same `L_resolved`-bound history
overlap.

The wrapper does NOT collapse to `compute_three_way_split`; that
library helper handles the val/cal carve-out inside the deep
estimator's fit. The B4.1a contract makes a separate carve-out
inside CV folds explicit:

- For seq-sklearn adapters: pass `val_split_strategy="time_ordered"`
  so the adapter's own validation carve-out from the train fold
  respects time order (arch-IIb / B4.5 named test 3).
- For GBM and sklearn-passthrough adapters (Phase B3-followup): no
  internal val carve-out; the harness materializes a held-out
  validation block from the train fold via the same time-ordered
  rule when needed.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from benchmarks.config import DatasetSpec
from seq_sklearn import EntityTimeSeriesSplit


def make_splitter(
    spec: DatasetSpec,
    *,
    lookback: int,
    n_splits: int = 5,
    gap: int = 0,
    max_train_size: int | None = None,
) -> EntityTimeSeriesSplit:
    """Build the canonical splitter for a dataset.

    Reads `entity_col` / `time_col` from the spec; uses the resolved
    lookback per `benchmarks.protocol.lookback.resolve_lookback`.
    """
    return EntityTimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        lookback=lookback,
        id_col=spec.entity_col,
        time_col=spec.time_col,
    )


def iter_folds(
    splitter: EntityTimeSeriesSplit, panel: pd.DataFrame
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Forward to ``splitter.split(panel)``, returning
    ``(train_idx, test_idx)`` per fold."""
    yield from splitter.split(panel)
