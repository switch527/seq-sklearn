"""Materialized-dataset record + loader protocol (Phase B1).

A registered dataset is a ``DatasetSpec`` (pydantic, frozen, in
:mod:`benchmarks.config`) plus a ``loader`` callable that produces
a :class:`PanelDataset` from a cache root. The loader signature is
intentionally narrow so a future caller (the experiment driver in
Phase B5+) can invoke any registered dataset uniformly.

Loaders MUST consume the library through its public façade
(``seq_sklearn.__all__``); the no-deep-imports test from Phase B0
catches violations across the whole ``benchmarks/`` tree.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.config import DatasetSpec


@dataclass(frozen=True, slots=True)
class PanelDataset:
    """A loaded benchmark dataset.

    ``panel`` is the F2-contract DataFrame: one row per
    ``(entity_id, period)`` pair, with the schema columns declared
    by the spec. ``y`` is the supervised target, one value per row
    (the estimator picks per-window labels internally based on
    ``lookback`` per F2). ``spec`` is the originating registry
    entry (no copy; identity-equal to the registered spec).

    The dataclass is slotted + frozen so two `PanelDataset` instances
    with the same `(panel, y, spec)` are treated as a single artifact
    by the harness; loaders return one of these and the cache layer
    Parquet-writes the underlying panel directly.
    """

    spec: DatasetSpec
    panel: pd.DataFrame
    y: np.ndarray


# Loaders consume the cache root and return a `PanelDataset`. The
# signature is narrow so the experiment driver can call any loader
# uniformly; loader-side details (network downloads, integrity
# checks, densification) are encapsulated in the loader body.
LoaderCallable = Callable[[Path], PanelDataset]
