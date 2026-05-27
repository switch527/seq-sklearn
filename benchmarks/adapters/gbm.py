"""GBM adapter family (Phase B10 / B2-followup).

Wraps the design's named sklearn-API GBM libraries
(`LightGBM`, `XGBoost`, `CatBoost`) into the harness's
`SeqSklearnAdapter` protocol. Each library exposes a classifier +
regressor; B10 registers six model specs total. Every adapter
shares one pattern:

1. Featurize the input `PanelDataset.panel` via
   `benchmarks.protocol.lag_featurize` (the B4.3 lag-feature
   builder). The featurized X is one row per panel row, with
   `{col}_lag{k}` columns + a `missing_lag_count` warm-up audit
   column.
2. Drop the `entity_col` + `time_col` + the target column from
   the featurized output (the featurizer keeps the panel index
   alignment; the GBM consumes only the lag columns + the audit
   column).
3. Route `fit` / `predict` / `predict_proba` to the underlying
   sklearn-API estimator.

`predict_quantiles` is unsupported on every GBM adapter in v1
(the design's quantile-loss family is the seq-sklearn quantile
regressor; a GBM regressor's quantile output ships in a B10-
followup branch).

Design alignment:

- B2.x roster: "GBM adapter (LightGBM, XGBoost, CatBoost via their
  sklearn-API)" (impl plan §B2). Closed by this module.
- B3-followup `test_lookback_binding.py`: the cross-consumer
  identity invariant the test pins (the splitter, the deep
  adapter, the featurizer, and the MTS reshape all read the same
  `L_resolved`) now has its third consumer; the test lands in
  this branch or a follow-up alongside the TSC adapter.
- B6.2.5 ensemble: the design names a GBM-only ensemble paired
  against GBM+seq for the complementarity-lift Wilcoxon. The
  ensemble driver (B6) is unchanged; this branch only ships the
  adapters that the ensemble pairs over.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, Self, cast

import numpy as np
import pandas as pd

from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import DatasetSpec, ModelSpec, TaskType
from benchmarks.protocol import lag_featurize
from benchmarks.registry.models import register_adapter
from seq_sklearn import NotFittedError

__all__: list[str] = []


logger = logging.getLogger(__name__)


class _SklearnLikeEstimator(Protocol):
    """Narrow Protocol the GBM adapter consumes from each library.

    Captures only the surface the adapter touches (sklearn-API
    `fit` / `predict` / `predict_proba`); avoids importing the
    concrete classes at module scope so a missing optional
    dependency doesn't break the import of `benchmarks/adapters`.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...  # noqa: N803

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...  # noqa: N803


@dataclass
class _GBMAdapter:
    """Generic sklearn-API GBM adapter.

    Driven by a `_estimator_factory` callable that the per-library
    factory closes over; the adapter itself is library-agnostic.
    Subclass-by-composition keeps the adapter's surface (the
    `SeqSklearnAdapter` Protocol contract) on one class so the
    runtime `isinstance` check passes for every GBM variant.

    Attributes set per-instance by the factory:

    - `name`: registered model name (e.g., `"lightgbm_classifier"`).
    - `family`: always `"gbm"`.
    - `task_types`: a single-element tuple (`("binary",
      "multiclass")` for classifiers; `("regression_point",)` for
      regressors). The harness's outer skip-mismatch gate routes
      on this tuple.
    - `supports_proba`: True for classifiers, False for regressors.
    """

    # Sentinel name on the base; concrete subclasses below
    # override with the registered model name. The sentinel lets
    # error messages reference a meaningful identifier when the
    # base is mis-instantiated directly (R1 fix).
    name: ClassVar[str] = "_GBMAdapter"
    family: ClassVar[str] = "gbm"
    task_types: ClassVar[tuple[TaskType, ...]] = ()
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    # `_estimator_factory` and `_classifier` are subclass-bound
    # via `__post_init__` (see the six concrete subclasses below).
    # `init=False` keeps them out of the dataclass's `__init__`
    # signature so a caller cannot accidentally pass them in
    # (R1 arch-I4 / code-N1).
    _estimator_factory: Callable[..., _SklearnLikeEstimator] | None = field(
        default=None, init=False, repr=False
    )
    _classifier: bool = field(default=False, init=False, repr=False)
    _est: _SklearnLikeEstimator | None = field(default=None, init=False, repr=False)
    _feature_columns: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    # Per-instance featurizer cache (R1 arch-I1 close): the B5
    # driver calls `predict` then `predict_proba` then a latency
    # loop of more `predict`s on the same `test_panel` object;
    # the featurizer is pure but its O(N * features * lookback)
    # cost is real on large panels. Cache by `id(panel)` so the
    # second call on the same DataFrame is O(1). The cache is
    # invalidated whenever a different panel object is passed
    # (the harness never mutates panels in-place).
    _cached_panel_id: int | None = field(default=None, init=False, repr=False)
    _cached_featurized: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def _featurize(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Build the flat lag-feature X for the GBM.

        `lag_featurize` returns one row per panel row; the
        harness's metric layer is also one row per panel row, so
        the index alignment between X and the held-out target
        carries through cleanly. Per-instance cache keyed by
        `id(panel)` so repeated calls on the same DataFrame
        (predict + predict_proba + latency-loop predict) skip the
        recomputation.
        """
        panel_id = id(panel)
        if self._cached_panel_id == panel_id and self._cached_featurized is not None:
            return self._cached_featurized
        featurized = lag_featurize(panel, self.spec)
        self._cached_panel_id = panel_id
        self._cached_featurized = featurized
        return featurized

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if self.spec.task_type not in self.task_types:
            raise ValueError(
                f"{self.name} does not support task_type="
                f"{self.spec.task_type!r}; supported: {self.task_types}"
            )
        if self._estimator_factory is None:
            raise RuntimeError(
                f"{self.name}: missing `_estimator_factory`; instantiate via a registered factory."
            )
        x = self._featurize(panel)
        self._feature_columns = tuple(x.columns)
        # `fit` may take library-specific kwargs (e.g.,
        # `categorical_feature=` for LightGBM); B10 keeps the
        # passthrough simple and lets the per-library factory
        # configure these via the hyperparameters dict.
        self._est = self._estimator_factory(**self.hyperparameters)
        self._est.fit(x, np.asarray(y))
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(f"{self.name}: predict() called before fit(); call fit() first")
        x = self._featurize(panel)
        if self._feature_columns is not None:
            x = cast(pd.DataFrame, x[list(self._feature_columns)])
        return np.asarray(self._est.predict(x))

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise NotFittedError(
                f"{self.name}: predict_proba() called before fit(); call fit() first"
            )
        if not self._classifier:
            raise ProbaUnsupportedError(
                f"{self.name}: regression adapters do not produce class probabilities; "
                f"check `supports_proba` before calling"
            )
        x = self._featurize(panel)
        if self._feature_columns is not None:
            x = cast(pd.DataFrame, x[list(self._feature_columns)])
        proba_fn = getattr(self._est, "predict_proba", None)
        if proba_fn is None:
            raise ProbaUnsupportedError(
                f"{self.name}: underlying estimator has no `predict_proba` method"
            )
        return np.asarray(proba_fn(x))

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:  # noqa: ARG002
        # B10 ships point-only GBM regression; quantile loss for
        # GBMs ships in a B10-followup branch alongside the LightGBM
        # `objective="quantile"` + XGBoost quantile-regression
        # wiring.
        raise QuantilesUnsupportedError(
            f"{self.name}: GBM adapters do not produce quantile predictions in v1; "
            f"a B10-followup branch wires LightGBM's `objective='quantile'`"
        )


def _make_lightgbm_classifier_estimator(**hyperparameters: Any) -> _SklearnLikeEstimator:
    import lightgbm as lgb

    defaults: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "verbose": -1,
    }
    defaults.update(hyperparameters)
    return cast(_SklearnLikeEstimator, lgb.LGBMClassifier(**defaults))


def _make_lightgbm_regressor_estimator(**hyperparameters: Any) -> _SklearnLikeEstimator:
    import lightgbm as lgb

    defaults: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "verbose": -1,
    }
    defaults.update(hyperparameters)
    return cast(_SklearnLikeEstimator, lgb.LGBMRegressor(**defaults))


def _make_xgboost_classifier_estimator(**hyperparameters: Any) -> _SklearnLikeEstimator:
    import xgboost as xgb

    defaults: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "tree_method": "hist",
    }
    defaults.update(hyperparameters)
    return cast(_SklearnLikeEstimator, xgb.XGBClassifier(**defaults))


def _make_xgboost_regressor_estimator(**hyperparameters: Any) -> _SklearnLikeEstimator:
    import xgboost as xgb

    defaults: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "tree_method": "hist",
    }
    defaults.update(hyperparameters)
    return cast(_SklearnLikeEstimator, xgb.XGBRegressor(**defaults))


def _make_catboost_classifier_estimator(**hyperparameters: Any) -> _SklearnLikeEstimator:
    import catboost as cb

    defaults: dict[str, Any] = {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    defaults.update(hyperparameters)
    return cast(_SklearnLikeEstimator, cb.CatBoostClassifier(**defaults))


def _make_catboost_regressor_estimator(**hyperparameters: Any) -> _SklearnLikeEstimator:
    import catboost as cb

    defaults: dict[str, Any] = {
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    defaults.update(hyperparameters)
    return cast(_SklearnLikeEstimator, cb.CatBoostRegressor(**defaults))


# Per-(library, task) concrete adapter classes. Each pins its
# `name`, `task_types`, `supports_proba`, and the estimator
# factory the generic base routes through. Explicit subclasses
# (vs dynamic `type()` synthesis) keep the protocol's class-
# attribute declarations visible to pyright + the runtime
# `isinstance(..., SeqSklearnAdapter)` checker.


@dataclass
class _LightGBMClassifierAdapter(_GBMAdapter):
    name: ClassVar[str] = "lightgbm_classifier"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._estimator_factory = _make_lightgbm_classifier_estimator
        self._classifier = True


@dataclass
class _LightGBMRegressorAdapter(_GBMAdapter):
    name: ClassVar[str] = "lightgbm_regressor"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self._estimator_factory = _make_lightgbm_regressor_estimator
        self._classifier = False


@dataclass
class _XGBoostClassifierAdapter(_GBMAdapter):
    name: ClassVar[str] = "xgboost_classifier"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._estimator_factory = _make_xgboost_classifier_estimator
        self._classifier = True


@dataclass
class _XGBoostRegressorAdapter(_GBMAdapter):
    name: ClassVar[str] = "xgboost_regressor"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self._estimator_factory = _make_xgboost_regressor_estimator
        self._classifier = False


@dataclass
class _CatBoostClassifierAdapter(_GBMAdapter):
    name: ClassVar[str] = "catboost_classifier"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self._estimator_factory = _make_catboost_classifier_estimator
        self._classifier = True


@dataclass
class _CatBoostRegressorAdapter(_GBMAdapter):
    name: ClassVar[str] = "catboost_regressor"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self._estimator_factory = _make_catboost_regressor_estimator
        self._classifier = False


# Model spec + factory registration. The reason strings document
# which design-named comparator each adapter closes.
_REGISTRATIONS: tuple[tuple[type[_GBMAdapter], str], ...] = (
    (
        _LightGBMClassifierAdapter,
        "LightGBM via sklearn-API; the design's first-named GBM baseline (B2.x roster).",
    ),
    (
        _LightGBMRegressorAdapter,
        "LightGBM via sklearn-API; point-regression baseline.",
    ),
    (
        _XGBoostClassifierAdapter,
        "XGBoost via sklearn-API; second design-named GBM baseline.",
    ),
    (
        _XGBoostRegressorAdapter,
        "XGBoost via sklearn-API; point-regression baseline.",
    ),
    (
        _CatBoostClassifierAdapter,
        "CatBoost via sklearn-API; third design-named GBM baseline.",
    ),
    (
        _CatBoostRegressorAdapter,
        "CatBoost via sklearn-API; point-regression baseline.",
    ),
)


def _make_factory(
    adapter_cls: type[_GBMAdapter],
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
            family="gbm",
            task_types=_adapter_cls.task_types,
            supports_proba=_adapter_cls.supports_proba,
            reason=_reason,
        ),
        _make_factory(_adapter_cls),
    )
