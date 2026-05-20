"""Model-adapter registrations (Phase B2).

Importing this package triggers each adapter module's
``register_model(spec)`` call. Phase B2 ships the protocol + the
seq-sklearn adapter pair (TFTClassifier + TFTRegressor) as the
primary deliverable; comparator families (GBMs via sklearn-API,
classical TSC via aeon) land in Phase B2-followup branches
(`benchmark-phase-B2-<family>`), each gated by the adapter-protocol
conformance test so the cumulative roster cannot regress silently.
"""

from benchmarks.adapters import seq_sklearn  # pyright: ignore[reportUnusedImport]
from benchmarks.adapters._base import (
    ProbaUnsupportedError,  # pyright: ignore[reportUnusedImport]
    SeqSklearnAdapter,  # pyright: ignore[reportUnusedImport]
)

__all__ = [
    "ProbaUnsupportedError",
    "SeqSklearnAdapter",
    "seq_sklearn",
]
