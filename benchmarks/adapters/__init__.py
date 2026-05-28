"""Model-adapter registrations (Phase B2 + B10 + B12).

Importing this package triggers each adapter module's
``register_model(spec)`` call. Phase B2 shipped the protocol + the
seq-sklearn adapter pair (TFTClassifier + TFTRegressor) as the
primary deliverable. Phase B10 added the GBM family
(LightGBM / XGBoost / CatBoost via sklearn-API; six model specs
total) per the B2.x roster. Phase B12 adds the classical-TSC
family (ROCKET / MultiRocket / Catch22 via aeon; three classifier
specs). aeon is NOT installed by default; the adapter's
``_check_aeon_available`` raises
:class:`OptionalDependencyMissingError` at fit time and the B5 +
B8 drivers catch it as a typed skip rather than a crash.
"""

from benchmarks.adapters import (
    gbm,  # pyright: ignore[reportUnusedImport]
    seq_sklearn,  # pyright: ignore[reportUnusedImport]
    tsc,  # pyright: ignore[reportUnusedImport]
)
from benchmarks.adapters._base import (
    OptionalDependencyMissingError,  # pyright: ignore[reportUnusedImport]
    ProbaUnsupportedError,  # pyright: ignore[reportUnusedImport]
    QuantilesUnsupportedError,  # pyright: ignore[reportUnusedImport]
    SeqSklearnAdapter,  # pyright: ignore[reportUnusedImport]
)

__all__ = [
    "OptionalDependencyMissingError",
    "ProbaUnsupportedError",
    "QuantilesUnsupportedError",
    "SeqSklearnAdapter",
    "gbm",
    "seq_sklearn",
    "tsc",
]
