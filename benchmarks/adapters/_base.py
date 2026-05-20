"""Model-adapter protocol for the benchmark harness (B3.2).

Every model the harness benchmarks is callable through a single
:class:`SeqSklearnAdapter` protocol surface. The protocol is the
narrow contract the experiment driver (Phase B5+) consumes; each
concrete adapter (`benchmarks/adapters/<family>.py`) implements it
and registers itself with `benchmarks.registry.models`.

The protocol intentionally mirrors the sklearn estimator contract:
``fit(panel, y) -> self``, ``predict(panel) -> np.ndarray``, and
the optional ``predict_proba`` / ``predict_quantiles`` methods.
The ``supports_proba`` flag tells the harness whether the
probability-based metrics (log-loss, Brier, ECE, ROC-AUC, PR-AUC)
are computable for this model on this dataset; if ``False``, the
threshold-dependent metrics (accuracy, precision/recall/F1) still
apply, but every consumer of ``predict_proba`` raises
:class:`ProbaUnsupportedError`.

The protocol is checkable at runtime via ``isinstance(obj,
SeqSklearnAdapter)`` (PEP-544 runtime-checkable Protocol).
"""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from benchmarks.config import TaskType


class ProbaUnsupportedError(NotImplementedError):
    """Adapter does not expose calibrated class probabilities (qa-I3
    / B3.2.1).

    Raised by ``predict_proba`` on adapters whose underlying
    estimator does not produce probabilities (e.g. KNN-DTW, some
    classical TSC baselines). The harness inspects
    ``adapter.supports_proba`` first and only routes
    probability-based metrics to ``supports_proba=True`` adapters;
    this typed error covers the path where a caller bypasses that
    inspection.
    """


class QuantilesUnsupportedError(NotImplementedError):
    """Adapter does not expose quantile predictions.

    Raised by ``predict_quantiles`` on classifier adapters and on
    point-regression adapters. The Phase B4 metrics layer inspects
    ``adapter.task_types`` and only routes quantile-based metrics
    (pinball loss at the configured levels) to adapters that admit
    ``regression_quantile``; this typed error covers the bypass
    path.
    """


@runtime_checkable
class SeqSklearnAdapter(Protocol):
    """The benchmark-harness adapter contract.

    Implementations: one per model family. The class attributes
    declare the registry shape; the methods are the experiment-
    driver call surface.
    """

    # Class-level metadata. Read by the harness BEFORE `fit` to
    # decide which (dataset, model) cells are applicable.

    name: str
    """Registry name; must equal the `ModelSpec.name` registered
    via `benchmarks.registry.models.register_model`."""

    family: str
    """One of `seq_sklearn`, `gbm`, `tsc`, `sklearn_passthrough`
    (the `ModelFamily` literal in `benchmarks.config`)."""

    task_types: tuple[TaskType, ...]
    """Task types this adapter supports. The harness skips a
    (dataset, model) cell when `dataset.task_type not in
    adapter.task_types`."""

    supports_proba: bool
    """Whether ``predict_proba`` produces calibrated class
    probabilities; if False, ``predict_proba`` raises
    :class:`ProbaUnsupportedError` and the probability-based
    metrics are skipped on this cell."""

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> "SeqSklearnAdapter":
        """Fit on the F2-contract panel and target. Returns ``self``
        per the sklearn convention."""
        ...

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        """Predict labels (classification) or point values
        (regression) on the F2-contract panel. Returns a 1-D
        ``np.ndarray`` of length ``len(panel)`` (per-row labels;
        the harness aggregates to per-entity if needed)."""
        ...

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        """Class probabilities for classification adapters.

        Returns shape ``(len(panel), n_classes)``. Raises
        :class:`ProbaUnsupportedError` on adapters with
        ``supports_proba=False``.
        """
        ...

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        """Quantile predictions for `regression_quantile` adapters.

        Returns shape ``(len(panel), n_quantiles)`` where the
        ordering matches the adapter's configured quantile levels.
        Raises :class:`QuantilesUnsupportedError` on classifier
        adapters and on point-regression adapters; the harness's
        metric router inspects ``adapter.task_types`` to decide.
        """
        ...
