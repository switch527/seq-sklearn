"""Exception hierarchy for seq-sklearn (per architecture A10).

All library-raised exceptions inherit from :class:`SeqSklearnError` so
callers can catch the entire library surface with one `except` clause.
:class:`NotFittedError` additionally subclasses
:class:`sklearn.exceptions.NotFittedError` so sklearn's
``check_estimators_unfitted`` and any sklearn-aware caller catches it too.
"""

import sklearn.exceptions as sk_exc

__all__ = [
    "ConfigError",
    "DataContractError",
    "NotFittedError",
    "PredictionError",
    "SeqSklearnError",
    "TrainingError",
]


class SeqSklearnError(Exception):
    """Root of the library exception hierarchy."""


class ConfigError(SeqSklearnError):
    """Configuration validation failure or illegal F5 cross-field combination."""


class DataContractError(SeqSklearnError):
    """Input-data contract violation: missing columns, duplicate (id, time),
    unsupported dtype, tz mixing, NaN-in-features, etc."""


class TrainingError(SeqSklearnError):
    """Training-time failure: NaN loss, mixed-precision divergence,
    non-monotone conformal quantiles, etc."""


class PredictionError(SeqSklearnError):
    """Predict-time failure: shape mismatch, schema-version mismatch on
    load, missing migration path, etc."""


class NotFittedError(SeqSklearnError, sk_exc.NotFittedError):
    """Estimator used before fit.

    MRO order is load-bearing: :class:`SeqSklearnError` is listed first so a
    library-side ``except SeqSklearnError`` catches; sklearn-side
    ``except sklearn.exceptions.NotFittedError`` also catches via the second
    parent. Both contracts hold without preferential treatment in
    ``__str__`` or pickling.
    """
