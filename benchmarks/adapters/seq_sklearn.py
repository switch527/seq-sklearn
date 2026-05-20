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
from typing import Any, Self

import numpy as np
import pandas as pd

from benchmarks.adapters._base import ProbaUnsupportedError
from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.registry.models import register_model
from seq_sklearn import (
    SchedulerParams,
    TabularConfigParams,
    TFTClassifier,
    TFTRegressor,
)

# Default scheduler matches the docs how-to:
# cosine_with_warmup with a short warmup.
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
    """

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    name: str = "tft_classifier"
    family: str = "seq_sklearn"
    task_types: tuple[str, ...] = ("binary", "multiclass")
    supports_proba: bool = True

    _est: TFTClassifier | None = field(default=None, init=False, repr=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if self.spec.task_type not in self.task_types:
            raise ValueError(
                f"SeqSklearnTFTClassifierAdapter does not support "
                f"task_type={self.spec.task_type!r}; supported: "
                f"{self.task_types}"
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
            raise RuntimeError(
                f"{self.name}: predict() called before fit(); call fit() first"
            )
        return self._est.predict(panel)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if self._est is None:
            raise RuntimeError(
                f"{self.name}: predict_proba() called before fit(); call fit() first"
            )
        return self._est.predict_proba(panel)


@dataclass
class SeqSklearnTFTRegressorAdapter:
    """Adapter wrapping :class:`seq_sklearn.TFTRegressor`.

    Supports both point and quantile regression. ``supports_proba``
    is False (regression has no class probabilities); the harness's
    probability-based metrics are skipped for this family.
    """

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    name: str = "tft_regressor"
    family: str = "seq_sklearn"
    task_types: tuple[str, ...] = ("regression_point", "regression_quantile")
    supports_proba: bool = False

    _est: TFTRegressor | None = field(default=None, init=False, repr=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if self.spec.task_type not in self.task_types:
            raise ValueError(
                f"SeqSklearnTFTRegressorAdapter does not support "
                f"task_type={self.spec.task_type!r}; supported: "
                f"{self.task_types}"
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
            raise RuntimeError(
                f"{self.name}: predict() called before fit(); call fit() first"
            )
        return self._est.predict(panel)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:  # noqa: ARG002
        # `panel` matches the protocol signature; this regressor
        # adapter never returns probabilities so the parameter is
        # intentionally unused.
        raise ProbaUnsupportedError(
            f"{self.name}: regression adapters do not produce class "
            f"probabilities; check `supports_proba` before calling"
        )

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        """Quantile predictions when fit with `task_type=
        'regression_quantile'`."""
        if self._est is None:
            raise RuntimeError(
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
