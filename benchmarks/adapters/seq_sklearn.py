"""seq-sklearn TFT adapter pair (B3.2).

The library's `TFTClassifier` and `TFTRegressor` already implement
the sklearn `fit` / `predict` / `predict_proba` /
`predict_quantiles` contract on the F2-contract panel directly, so
the adapter shape is thin: it builds the `TabularConfigParams` from
the dataset spec, instantiates the estimator with the
adapter-config-driven hyperparameters, and forwards calls.

Both classes consume the library exclusively through its public
v1.0.0 façade (`seq_sklearn.__all__`). This is the design's named
canary for whether the façade is genuinely usable; the no-deep-
imports gate at `tests/benchmarks/test_scaffold.py` would catch
any drift into `seq_sklearn._*`.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
)
from benchmarks.config import DatasetSpec, ModelSpec, TaskType
from benchmarks.registry.models import register_model
from seq_sklearn import (
    NotFittedError,
    SchedulerParams,
    TabularConfigParams,
    TFTClassifier,
    TFTRegressor,
)

# Keys that name the adapter's bind-from-spec contract and the
# library's wiring; user `hyperparameters` must not override these,
# because the adapter's task-type guard and the registry's metadata
# all read from the spec, and a hyperparameter override would
# silently bypass the alignment. The Phase B8 HPO driver feeds
# `suggest_params` output verbatim into `hyperparameters`, so this
# is the exact path that would otherwise hit a confusing run-time
# crash inside the library.
_LOCKED_HYPERPARAMETER_KEYS: frozenset[str] = frozenset(
    {"task_type", "tabular_config", "scheduler", "verbose"}
)


def _check_locked_keys(adapter_name: str, hyperparameters: dict[str, Any]) -> None:
    """Raise `ValueError` when `hyperparameters` overlap with the
    bind-from-spec contract keys."""
    overlap = set(hyperparameters) & _LOCKED_HYPERPARAMETER_KEYS
    if overlap:
        raise ValueError(
            f"{adapter_name}: hyperparameters {sorted(overlap)} are "
            f"bound by the adapter from the dataset spec and cannot be "
            f"overridden; remove them from `hyperparameters`"
        )


# Default scheduler. `SchedulerParams` is a sklearn `BaseEstimator`
# adapter (mutable, not frozen pydantic), so sharing one instance
# across adapters is safe ONLY because every consumer treats it as
# read-only state (`TFTClassifier.__init__` assigns the passed
# object to `self.scheduler` without mutating it). Do not mutate
# this module-level instance in adapter code.
_DEFAULT_SCHEDULER = SchedulerParams(name="cosine_with_warmup", warmup_steps=50)


def _build_tabular_config(spec: DatasetSpec) -> TabularConfigParams:
    """Build the library's `TabularConfigParams` from a `DatasetSpec`.

    Treats every declared feature column as time-varying; static
    features land in Phase B2-followup once the spec carries a
    static-vs-time-varying split (current B2.2 has only the
    real/categorical split). The defaults are conservative: minimum
    one period at fit and predict, lookback from the spec, 10k
    categorical-cardinality cap.
    """
    return TabularConfigParams(
        id_col=spec.entity_col,
        time_col=spec.time_col,
        time_varying_real_cols=tuple(spec.feature_real_cols),
        time_varying_categorical_cols=tuple(spec.feature_categorical_cols),
        static_real_cols=(),
        static_categorical_cols=(),
        lookback=spec.lookback,
        min_periods=1,
        min_periods_predict=1,
        max_categorical_cardinality=10_000,
    )


@dataclass
class SeqSklearnTFTClassifierAdapter:
    """Adapter wrapping :class:`seq_sklearn.TFTClassifier`.

    Constructed with a `DatasetSpec` (carries the panel schema and
    the task type) plus any TFT hyperparameter overrides. The
    estimator is built lazily at `fit` time so the adapter can be
    instantiated cheaply by the harness for protocol introspection.
    Metadata attributes (`name`, `family`, `task_types`,
    `supports_proba`) are `ClassVar` so they cannot be silently
    mutated per-instance and match the protocol's class-attribute
    declarations exactly.
    """

    # Class-level metadata (Protocol contract); pinned ClassVar so
    # neither a stray instance assignment nor a hyperparameter
    # passthrough can drift them from the registered ModelSpec.
    name: ClassVar[str] = "tft_classifier"
    family: ClassVar[str] = "seq_sklearn"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    _est: TFTClassifier | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _check_locked_keys(self.name, self.hyperparameters)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if self.spec.task_type not in self.task_types:
            raise ValueError(
                f"{self.name} does not support task_type="
                f"{self.spec.task_type!r}; supported: {self.task_types}"
            )
        kwargs: dict[str, Any] = {
            "task_type": self.spec.task_type,
            "tabular_config": _build_tabular_config(self.spec),
            "scheduler": _DEFAULT_SCHEDULER,
            "verbose": False,
        }
        kwargs.update(self.hyperparameters)
        self._est = TFTClassifier(**kwargs)
        self._est.fit(panel, y)
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(
                f"{self.name}: predict() called before fit(); call fit() first"
            )
        return self._est.predict(panel)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(
                f"{self.name}: predict_proba() called before fit(); call fit() first"
            )
        return self._est.predict_proba(panel)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:  # noqa: ARG002
        # `panel` matches the protocol signature; classifier
        # adapters do not produce quantile predictions.
        raise QuantilesUnsupportedError(
            f"{self.name}: classifier adapters do not produce "
            f"quantile predictions; the harness's quantile-based "
            f"metrics are only routed to regression_quantile adapters"
        )


@dataclass
class SeqSklearnTFTRegressorAdapter:
    """Adapter wrapping :class:`seq_sklearn.TFTRegressor`.

    Supports both point and quantile regression. ``supports_proba``
    is False (regression has no class probabilities); the harness's
    probability-based metrics are skipped for this family. The
    quantile-prediction path activates when the spec declares
    ``task_type="regression_quantile"``.
    """

    name: ClassVar[str] = "tft_regressor"
    family: ClassVar[str] = "seq_sklearn"
    task_types: ClassVar[tuple[TaskType, ...]] = (
        "regression_point",
        "regression_quantile",
    )
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    _est: TFTRegressor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _check_locked_keys(self.name, self.hyperparameters)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if self.spec.task_type not in self.task_types:
            raise ValueError(
                f"{self.name} does not support task_type="
                f"{self.spec.task_type!r}; supported: {self.task_types}"
            )
        kwargs: dict[str, Any] = {
            "task_type": self.spec.task_type,
            "tabular_config": _build_tabular_config(self.spec),
            "scheduler": _DEFAULT_SCHEDULER,
            "verbose": False,
        }
        kwargs.update(self.hyperparameters)
        self._est = TFTRegressor(**kwargs)
        self._est.fit(panel, y.astype(np.float64))
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(
                f"{self.name}: predict() called before fit(); call fit() first"
            )
        return self._est.predict(panel)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:  # noqa: ARG002
        # `panel` matches the protocol signature; regressors never
        # return probabilities.
        raise ProbaUnsupportedError(
            f"{self.name}: regression adapters do not produce class "
            f"probabilities; check `supports_proba` before calling"
        )

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        """Quantile predictions when fit with `task_type=
        'regression_quantile'`."""
        if self._est is None:
            raise NotFittedError(
                f"{self.name}: predict_quantiles() called before fit()"
            )
        return self._est.predict_quantiles(panel)


# Register the model specs at module-import time per the B3.2.3
# extensibility contract.
_CLASSIFIER_SPEC = ModelSpec(
    name="tft_classifier",
    family="seq_sklearn",
    task_types=("binary", "multiclass"),
    supports_proba=True,
    reason=(
        "v1.0.0 reference classification model; the library's "
        "TFT-with-classification-head adaptation (the design's "
        "primary architectural contribution)"
    ),
)

_REGRESSOR_SPEC = ModelSpec(
    name="tft_regressor",
    family="seq_sklearn",
    task_types=("regression_point", "regression_quantile"),
    supports_proba=False,
    reason=(
        "v1.0.0 reference regression model; the library's TFT "
        "point and quantile regression head"
    ),
)

register_model(_CLASSIFIER_SPEC)
register_model(_REGRESSOR_SPEC)
