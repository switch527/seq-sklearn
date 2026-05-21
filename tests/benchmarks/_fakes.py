"""Shared fake-adapter machinery for the benchmarks e2e tests.

Both `test_raw_loss_experiment.py` (B5) and `test_ensemble_experiment.py`
(B6) need a synthetic registry of fake adapters + datasets. The
adapter classes + panel builders + registration helpers live here
so the two test files share one implementation.

`register_all_fakes_and_get_panels` is the one-call entry point: it
materializes the three synthetic panels, registers them with the
benchmarks dataset registry, registers the eleven fake adapters
with the benchmarks model + adapter-factory registry, and returns
the panel list. The autouse `isolated_registry` conftest fixture
restores the registry on teardown so the per-test side-effects do
not leak.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import DatasetSpec, ModelSpec, TaskType
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry import (
    register_adapter_factory,
    register_dataset,
    register_model,
)

_ZERO_SHA = "0" * 64


# --- panel builders ----------------------------------------------------------


def make_binary_panel(n_entities: int = 4, n_periods: int = 10) -> PanelDataset:
    """Synthetic 4 entity x 10 period binary classification panel."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for entity in range(n_entities):
        for t in range(n_periods):
            x = float(rng.normal(0.0, 1.0))
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "x_feat": x,
                    "y": int(x > 0),
                }
            )
    panel = pd.DataFrame(rows)
    spec = DatasetSpec(
        name="fake_binary",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/binary.csv",
        integrity_sha256=_ZERO_SHA,
        archive_basename="binary.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x_feat",),
        feature_categorical_cols=(),
        lookback=2,
        positive_label=1,
        citation="synthetic; test only",
    )
    return PanelDataset(spec=spec, panel=panel, y=panel["y"].to_numpy())


def make_regression_panel(n_entities: int = 4, n_periods: int = 10) -> PanelDataset:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1)
    for entity in range(n_entities):
        for t in range(n_periods):
            x = float(rng.normal(0.0, 1.0))
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "x_feat": x,
                    "y": 2.0 * x + 0.5,
                }
            )
    panel = pd.DataFrame(rows)
    spec = DatasetSpec(
        name="fake_regression_point",
        task_type="regression_point",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/reg.csv",
        integrity_sha256=_ZERO_SHA,
        archive_basename="reg.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x_feat",),
        feature_categorical_cols=(),
        lookback=2,
        citation="synthetic; test only",
    )
    return PanelDataset(spec=spec, panel=panel, y=panel["y"].to_numpy())


def make_regression_quantile_panel() -> PanelDataset:
    base = make_regression_panel()
    spec = base.spec.model_copy(
        update={
            "name": "fake_regression_quantile",
            "task_type": "regression_quantile",
            "archive_basename": "regq.csv",
            "source_uri": "https://example.test/regq.csv",
        }
    )
    return PanelDataset(spec=spec, panel=base.panel, y=base.y)


# --- fake adapters -----------------------------------------------------------


@dataclass
class ConstantBinaryAdapter:
    """Predicts the majority class with a constant 0.7/0.3 probability."""

    name: ClassVar[str] = "fake_constant_binary"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _majority: int = field(default=0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel
        if len(y) == 0:
            self._majority = 0
        else:
            values, counts = np.unique(y, return_counts=True)
            self._majority = int(values[counts.argmax()])
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._majority, dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        proba = np.full((len(panel), 2), 0.3)
        proba[:, self._majority] = 0.7
        return proba

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(
            f"{self.name}: classifier does not produce quantile predictions"
        )


@dataclass
class ConstantRegressorAdapter:
    """Predicts the training-mean target on every row."""

    name: ClassVar[str] = "fake_constant_regressor"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _mean: float = field(default=0.0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._mean, dtype=np.float64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise ProbaUnsupportedError(
            f"{self.name}: regression adapter does not produce probabilities"
        )

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(
            f"{self.name}: point-regression adapter does not produce quantiles"
        )


@dataclass
class CrashingAdapter:
    """Always raises at `fit`; exercises the adapter-error skip path."""

    name: ClassVar[str] = "fake_crashing"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        raise RuntimeError("fake_crashing: synthetic adapter failure")

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("not reached")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class QuantileRegressorAdapter:
    """Declares `regression_quantile` so the B5-followup skip fires."""

    name: ClassVar[str] = "fake_quantile_regressor"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = (
        "regression_point",
        "regression_quantile",
    )
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel), dtype=np.float64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise ProbaUnsupportedError(self.name)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(panel), 3), dtype=np.float64)


@dataclass
class NoProbaClassifierAdapter:
    """Classifier-applicable but supports_proba=False (KNN-DTW shape)."""

    name: ClassVar[str] = "fake_no_proba_classifier"
    family: ClassVar[str] = "tsc"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel), dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise ProbaUnsupportedError(self.name)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class NaNProbaAdapter:
    """Returns nan rows in y_proba for the first 2 test rows."""

    name: ClassVar[str] = "fake_nan_proba_classifier"
    family: ClassVar[str] = "seq_sklearn"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _majority: int = field(default=0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel
        if len(y) > 0:
            values, counts = np.unique(y, return_counts=True)
            self._majority = int(values[counts.argmax()])
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._majority, dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        proba = np.full((len(panel), 2), 0.3)
        proba[:, self._majority] = 0.7
        proba[: min(2, len(panel)), :] = np.nan
        return proba

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class HarnessBugAdapter:
    """Raises `ValueError` (a harness-bug signal) at fit time."""

    name: ClassVar[str] = "fake_harness_bug"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        raise ValueError("fake_harness_bug: ValueError must propagate")

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("not reached")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class RuntimeProbaErrorAdapter:
    """Declares supports_proba=True at the spec but raises
    `ProbaUnsupportedError` at runtime."""

    name: ClassVar[str] = "fake_runtime_proba_error"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel), dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise ProbaUnsupportedError(
            f"{self.name}: runtime ProbaUnsupportedError despite supports_proba=True"
        )

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class PredictCrashingAdapter:
    """Fits cleanly but raises `RuntimeError` at `predict`."""

    name: ClassVar[str] = "fake_predict_crashing"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("fake_predict_crashing: synthetic predict-time crash")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class NaNRegressorAdapter:
    """Regression adapter that returns nan in the first 2 y_pred rows."""

    name: ClassVar[str] = "fake_nan_regressor"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _mean: float = field(default=0.0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        out = np.full(len(panel), self._mean, dtype=np.float64)
        out[: min(2, len(panel))] = np.nan
        return out

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise ProbaUnsupportedError(self.name)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


@dataclass
class NotFittedAdapter:
    """Always raises `NotFittedError` at `predict`."""

    name: ClassVar[str] = "fake_not_fitted"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel, y
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        from seq_sklearn import NotFittedError as _NotFittedError

        raise _NotFittedError("fake_not_fitted: synthetic NotFittedError")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


_ADAPTER_CLASSES: tuple[type, ...] = (
    ConstantBinaryAdapter,
    ConstantRegressorAdapter,
    CrashingAdapter,
    QuantileRegressorAdapter,
    NoProbaClassifierAdapter,
    NaNProbaAdapter,
    HarnessBugAdapter,
    RuntimeProbaErrorAdapter,
    PredictCrashingAdapter,
    NaNRegressorAdapter,
    NotFittedAdapter,
)


def _register_fake_datasets(panels: list[PanelDataset]) -> None:
    for panel_dataset in panels:

        def _loader(_root: Path, _captured: PanelDataset = panel_dataset) -> PanelDataset:
            return _captured

        register_dataset(panel_dataset.spec, _loader)


def _register_fake_models() -> None:
    for adapter_cls in _ADAPTER_CLASSES:
        spec = ModelSpec(
            name=adapter_cls.name,
            family=adapter_cls.family,
            task_types=adapter_cls.task_types,
            supports_proba=adapter_cls.supports_proba,
            reason="synthetic; test fixture only",
        )
        register_model(spec)

        def _factory(
            *,
            spec: DatasetSpec,
            hyperparameters: dict[str, Any] | None = None,
            _cls: type = adapter_cls,
        ) -> SeqSklearnAdapter:
            return _cls(spec=spec, hyperparameters=hyperparameters or {})

        register_adapter_factory(adapter_cls.name, _factory)


def register_all_fakes_and_get_panels() -> list[PanelDataset]:
    """Register every synthetic dataset + adapter and return the
    panel list. The autouse `isolated_registry` conftest fixture
    restores the registry on teardown."""
    panels = [
        make_binary_panel(),
        make_regression_panel(),
        make_regression_quantile_panel(),
    ]
    _register_fake_datasets(panels)
    _register_fake_models()
    return panels
