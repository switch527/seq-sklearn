"""Classical-TSC adapter family (Phase B12 / B2-followup).

The TSC family wraps aeon's three flagship classifiers (ROCKET,
MultiRocket, Catch22) so they appear in the harness alongside the
deep `seq_sklearn` family and the `gbm` family. The adapter
consumes the `raw-mts` data view (`benchmarks/protocol/raw_mts.py`):
a `(n_instances, n_channels, L_resolved)` float32 tensor.

Aeon's API is per-instance: one prediction per entity, regardless
of how many panel rows the entity has. The F2 contract requires
per-row output (`predict(panel)` returns shape `(len(panel),)`).
Per-instance predictions are broadcast back to per-row at the
adapter's `predict` / `predict_proba` surface; every panel row of
an entity carries the same prediction. Below-floor entities
(fewer than `L_resolved` rows) receive `np.nan` per the A9.1
strip convention so the B5 driver's `_strip_below_floor_rows`
machinery surfaces the drop count uniformly.

Aeon is NOT installed by default: aeon 0.11.x pins
`scikit-learn < 1.6` and the library requires `>= 1.6`. The
adapter raises `OptionalDependencyMissingError("aeon")` at fit
time; the B5 + B8 drivers catch this in their narrow per-cell
exception tuple and emit a typed
`skipped_reason="optional_dep_missing: aeon"` row rather than
crashing the run.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, cast

import numpy as np
import pandas as pd

from benchmarks.adapters._base import (
    OptionalDependencyMissingError,
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import DatasetSpec, ModelSpec, TaskType
from benchmarks.protocol import (
    broadcast_per_instance_to_per_row,
    compute_categorical_categories,
    instance_labels,
    panel_to_tensor,
)
from benchmarks.registry.models import register_adapter
from seq_sklearn import NotFittedError


def _check_aeon_available() -> None:
    """Lazy-import aeon; raise `OptionalDependencyMissingError` if
    absent.

    Routed by the B5 + B8 drivers to a typed skip; the run
    continues with the TSC cells marked skipped.
    """
    import importlib.util

    if importlib.util.find_spec("aeon") is None:
        raise OptionalDependencyMissingError("aeon")


@dataclass
class _TSCAdapter:
    """Base adapter for the classical-TSC family.

    Concrete per-classifier subclasses set `name`, `_estimator_factory`,
    and the per-classifier hyperparameter defaults; the base owns
    the panel reshape, the per-instance label projection, and the
    per-row broadcast.
    """

    name: ClassVar[str] = "_TSCAdapter"
    family: ClassVar[str] = "tsc"
    task_types: ClassVar[tuple[TaskType, ...]] = ()
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _estimator_factory: Callable[..., Any] | None = field(default=None, init=False, repr=False)
    _est: Any = field(default=None, init=False, repr=False)
    # Per-instance reshape cache keyed by `id(panel)`; the B5
    # driver calls `predict` then `predict_proba` on the same panel
    # object and the reshape is O(N * channels * L) at fit time and
    # O(N * channels * L) at predict time. Cache halves the
    # predict-time cost.
    _cached_panel_id: int | None = field(default=None, init=False, repr=False)
    _cached_tensor: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_row_to_instance: np.ndarray | None = field(default=None, init=False, repr=False)
    # B39 / D-B12.6 closure: fit-time categorical reference; pins
    # channel layout across fit + predict so unseen-at-predict
    # category values map to all-zero one-hot rather than shifting
    # channel counts. None until fit() runs, which also clears the
    # reshape cache so a re-fit's tensor is rebuilt against the
    # new reference rather than served stale from `_cached_tensor`.
    _fit_categorical_categories: dict[str, tuple[Any, ...]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if type(self) is _TSCAdapter:
            raise TypeError(
                "_TSCAdapter is an abstract base; instantiate one of the "
                "three registered concrete subclasses via "
                "`instantiate_adapter(name, spec=...)` instead."
            )

    def _reshape(self, panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return `(X_3d, panel_row_to_instance)`, cached by `id(panel)`."""
        panel_id = id(panel)
        if (
            self._cached_panel_id == panel_id
            and self._cached_tensor is not None
            and self._cached_row_to_instance is not None
        ):
            return self._cached_tensor, self._cached_row_to_instance
        X_3d, row_to_instance = panel_to_tensor(  # noqa: N806
            panel,
            self.spec,
            categorical_categories=self._fit_categorical_categories,
        )
        self._cached_panel_id = panel_id
        self._cached_tensor = X_3d
        self._cached_row_to_instance = row_to_instance
        return X_3d, row_to_instance

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if self.spec.task_type not in self.task_types:
            raise ValueError(
                f"{self.name} does not support task_type="
                f"{self.spec.task_type!r}; supported: {self.task_types}"
            )
        _check_aeon_available()
        if self._estimator_factory is None:
            raise RuntimeError(
                f"{self.name}: missing `_estimator_factory`; instantiate via a registered factory."
            )
        # B39 / D-B12.6 closure: pin categorical channel layout
        # from the fit-time panel. Clear the reshape cache so the
        # new reference takes effect; otherwise a re-fit on a panel
        # already seen at predict time would return a stale tensor
        # built against the previous reference (or against
        # categorical_categories=None on the pre-B39 path).
        self._fit_categorical_categories = compute_categorical_categories(panel, self.spec)
        self._cached_panel_id = None
        self._cached_tensor = None
        self._cached_row_to_instance = None
        X_3d, row_to_instance = self._reshape(panel)  # noqa: N806
        y_per_instance = instance_labels(np.asarray(y), row_to_instance)
        self._est = self._estimator_factory(**self.hyperparameters)
        self._est.fit(X_3d, y_per_instance)
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(f"{self.name}: predict() called before fit(); call fit() first")
        X_3d, row_to_instance = self._reshape(panel)  # noqa: N806
        per_instance = np.asarray(self._est.predict(X_3d))
        # Broadcast per-instance to per-row; below-floor rows get
        # NaN so the B5 strip surfaces them uniformly.
        return broadcast_per_instance_to_per_row(per_instance, row_to_instance)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(
                f"{self.name}: predict_proba() called before fit(); call fit() first"
            )
        if not self.supports_proba:
            raise ProbaUnsupportedError(
                f"{self.name}: this adapter does not produce class probabilities; "
                "check `supports_proba` before calling"
            )
        proba_fn = getattr(self._est, "predict_proba", None)
        if proba_fn is None:
            raise ProbaUnsupportedError(
                f"{self.name}: underlying aeon estimator has no `predict_proba` method"
            )
        X_3d, row_to_instance = self._reshape(panel)  # noqa: N806
        per_instance_proba = np.asarray(proba_fn(X_3d))
        if per_instance_proba.ndim != 2:
            raise RuntimeError(
                f"{self.name}: predict_proba returned shape "
                f"{per_instance_proba.shape}; expected 2D (n_instances, "
                "n_classes)"
            )
        return broadcast_per_instance_to_per_row(per_instance_proba, row_to_instance)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:  # noqa: ARG002
        raise QuantilesUnsupportedError(
            f"{self.name}: TSC adapters are classification-only at v1 "
            "(D-B12.3); no quantile predictions"
        )


def _make_rocket_classifier(**hyperparameters: Any) -> Any:
    from aeon.classification.convolution_based import (  # pyright: ignore[reportMissingImports]
        RocketClassifier,
    )

    # aeon 1.x renamed `num_kernels` → `n_kernels`; pass through
    # whichever the caller supplies but normalize on the new name.
    if "num_kernels" in hyperparameters and "n_kernels" not in hyperparameters:
        hyperparameters["n_kernels"] = hyperparameters.pop("num_kernels")
    defaults: dict[str, Any] = {
        "n_kernels": 10_000,
        "n_jobs": -1,
    }
    defaults.update(hyperparameters)
    return RocketClassifier(**defaults)


def _make_multirocket_classifier(**hyperparameters: Any) -> Any:
    from aeon.classification.convolution_based import (  # pyright: ignore[reportMissingImports]
        MultiRocketClassifier,
    )

    if "num_kernels" in hyperparameters and "n_kernels" not in hyperparameters:
        hyperparameters["n_kernels"] = hyperparameters.pop("num_kernels")
    defaults: dict[str, Any] = {
        "n_kernels": 10_000,
        "max_dilations_per_kernel": 32,
        "n_jobs": -1,
    }
    defaults.update(hyperparameters)
    return MultiRocketClassifier(**defaults)


def _make_catch22_classifier(**hyperparameters: Any) -> Any:
    from aeon.classification.feature_based import (  # pyright: ignore[reportMissingImports]
        Catch22Classifier,
    )

    defaults: dict[str, Any] = {
        "outlier_norm": False,
        "replace_nans": True,
        "n_jobs": -1,
    }
    defaults.update(hyperparameters)
    return Catch22Classifier(**defaults)


@dataclass
class _RocketClassifierAdapter(_TSCAdapter):
    name: ClassVar[str] = "rocket_classifier"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._estimator_factory = _make_rocket_classifier


@dataclass
class _MultiRocketClassifierAdapter(_TSCAdapter):
    name: ClassVar[str] = "multirocket_classifier"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._estimator_factory = _make_multirocket_classifier


@dataclass
class _Catch22ClassifierAdapter(_TSCAdapter):
    name: ClassVar[str] = "catch22_classifier"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._estimator_factory = _make_catch22_classifier


_REGISTRATIONS: tuple[tuple[type[_TSCAdapter], str], ...] = (
    (
        _RocketClassifierAdapter,
        "ROCKET via aeon; the design's first-named classical-TSC baseline.",
    ),
    (
        _MultiRocketClassifierAdapter,
        "MultiRocket via aeon; the design's second-named classical-TSC baseline.",
    ),
    (
        _Catch22ClassifierAdapter,
        "Catch22 via aeon; the design's feature-based classical-TSC baseline.",
    ),
)


def _make_factory(
    adapter_cls: type[_TSCAdapter],
) -> Callable[..., SeqSklearnAdapter]:
    def _factory(
        *,
        spec: DatasetSpec,
        hyperparameters: dict[str, Any] | None = None,
    ) -> SeqSklearnAdapter:
        return cast(
            SeqSklearnAdapter,
            adapter_cls(spec=spec, hyperparameters=hyperparameters or {}),
        )

    return _factory


for _adapter_cls, _reason in _REGISTRATIONS:
    register_adapter(
        ModelSpec(
            name=_adapter_cls.name,
            family="tsc",
            task_types=_adapter_cls.task_types,
            supports_proba=_adapter_cls.supports_proba,
            reason=_reason,
        ),
        _make_factory(_adapter_cls),
    )
