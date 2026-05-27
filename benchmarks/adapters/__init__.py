"""Model-adapter registrations (Phase B2 + Phase B10).

Importing this package triggers each adapter module's
``register_model(spec)`` call. Phase B2 shipped the protocol + the
seq-sklearn adapter pair (TFTClassifier + TFTRegressor) as the
primary deliverable. Phase B10 adds the GBM family
(LightGBM / XGBoost / CatBoost via sklearn-API; six model specs
total) per the B2.x roster. The classical TSC family
(aeon-blocked) lands in a future B2-followup branch.
"""

from benchmarks.adapters import (
    gbm,  # pyright: ignore[reportUnusedImport]
    seq_sklearn,  # pyright: ignore[reportUnusedImport]
)
from benchmarks.adapters._base import (
    ProbaUnsupportedError,  # pyright: ignore[reportUnusedImport]
    QuantilesUnsupportedError,  # pyright: ignore[reportUnusedImport]
    SeqSklearnAdapter,  # pyright: ignore[reportUnusedImport]
)

__all__ = [
    "ProbaUnsupportedError",
    "QuantilesUnsupportedError",
    "SeqSklearnAdapter",
    "gbm",
    "seq_sklearn",
]
