"""Phase B10 GBM adapter protocol-conformance tests (B2-followup).

The six GBM adapters (LightGBM / XGBoost / CatBoost for classifier
and regressor task types) wrap sklearn-API estimators through the
harness's
`SeqSklearnAdapter` Protocol. These tests pin the
runtime-checkable Protocol contract + the registered ModelSpec /
adapter-factory shape + the typed errors.

The e2e fit + predict smoke lives in
`test_gbm_adapter_smoke.py` (marked `slow`); this file is the
fast-only protocol pass.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import DatasetSpec
from benchmarks.registry import get_model, instantiate_adapter

from seq_sklearn import NotFittedError

_GBM_CLASSIFIER_NAMES: tuple[str, ...] = (
    "lightgbm_classifier",
    "xgboost_classifier",
    "catboost_classifier",
)
_GBM_REGRESSOR_NAMES: tuple[str, ...] = (
    "lightgbm_regressor",
    "xgboost_regressor",
    "catboost_regressor",
)
_GBM_ALL_NAMES: tuple[str, ...] = _GBM_CLASSIFIER_NAMES + _GBM_REGRESSOR_NAMES


def _dataset_spec(task_type: str = "binary") -> DatasetSpec:
    return DatasetSpec(
        name="ds_gbm_test",
        task_type=task_type,  # type: ignore[arg-type]
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.com/data.csv",
        integrity_sha256="0" * 64,
        archive_basename="data.csv",
        entity_col="id",
        time_col="t",
        target_col="y",
        feature_real_cols=("x",),
        feature_categorical_cols=(),
        lookback=4,
        observation_cutoff_rule=None,
        densification_policy=None,
        positive_label=1 if task_type == "binary" else None,
        excluded=False,
        exclusion_reason=None,
        citation="GBM 2026",
    )


# --- registry registration ---------------------------------------------------


@pytest.mark.parametrize("name", _GBM_ALL_NAMES)
def test_gbm_adapter_registered_with_correct_family(name: str) -> None:
    spec = get_model(name)
    assert spec.family == "gbm"
    assert spec.name == name


@pytest.mark.parametrize("name", _GBM_CLASSIFIER_NAMES)
def test_gbm_classifier_task_types_and_proba(name: str) -> None:
    spec = get_model(name)
    assert spec.task_types == ("binary", "multiclass")
    assert spec.supports_proba is True


@pytest.mark.parametrize("name", _GBM_REGRESSOR_NAMES)
def test_gbm_regressor_task_types_and_no_proba(name: str) -> None:
    spec = get_model(name)
    assert spec.task_types == ("regression_point",)
    assert spec.supports_proba is False


# --- protocol conformance ----------------------------------------------------


@pytest.mark.parametrize("name", _GBM_ALL_NAMES)
def test_gbm_adapter_satisfies_protocol_at_runtime(name: str) -> None:
    """Runtime-checkable `SeqSklearnAdapter` Protocol: every
    registered GBM adapter must `isinstance`-check against the
    protocol."""
    task = "binary" if name in _GBM_CLASSIFIER_NAMES else "regression_point"
    spec = _dataset_spec(task_type=task)
    adapter = instantiate_adapter(name, spec=spec)
    assert isinstance(adapter, SeqSklearnAdapter)


@pytest.mark.parametrize("name", _GBM_ALL_NAMES)
def test_gbm_adapter_predict_before_fit_raises(name: str) -> None:
    task = "binary" if name in _GBM_CLASSIFIER_NAMES else "regression_point"
    spec = _dataset_spec(task_type=task)
    adapter = instantiate_adapter(name, spec=spec)
    with pytest.raises(NotFittedError, match="called before fit"):
        adapter.predict(pd.DataFrame())


@pytest.mark.parametrize("name", _GBM_CLASSIFIER_NAMES)
def test_gbm_classifier_predict_proba_before_fit_raises(name: str) -> None:
    spec = _dataset_spec(task_type="binary")
    adapter = instantiate_adapter(name, spec=spec)
    with pytest.raises(NotFittedError, match="called before fit"):
        adapter.predict_proba(pd.DataFrame())


@pytest.mark.parametrize("name", _GBM_REGRESSOR_NAMES)
def test_gbm_regressor_predict_proba_pre_fit_raises_not_fitted(name: str) -> None:
    """Pre-fit: `NotFittedError` takes precedence over the typed
    `ProbaUnsupportedError` because the adapter's null `_est` is
    checked first."""
    spec = _dataset_spec(task_type="regression_point")
    adapter = instantiate_adapter(name, spec=spec)
    with pytest.raises(NotFittedError):
        adapter.predict_proba(pd.DataFrame())


@pytest.mark.parametrize("name", _GBM_REGRESSOR_NAMES)
def test_gbm_regressor_predict_proba_post_fit_raises_proba_unsupported(name: str) -> None:
    """R1 qa-CRIT-1 close: AFTER fit (so `_est` is bound), a
    regressor adapter's `predict_proba` raises `ProbaUnsupportedError`
    per the protocol contract. The harness's outer gate routes
    around this normally; the typed error is the defense-in-depth
    backstop for a caller that bypasses the gate."""
    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()
    from benchmarks.registry import get_dataset, get_loader

    spec = get_dataset("fake_regression_point")
    loader = get_loader("fake_regression_point")
    ds = loader(Path("/tmp/cache-b10-r1-proba"))
    # CatBoost uses `iterations`; the other two use `n_estimators`.
    overrides: dict[str, Any] = {"iterations": 10} if "catboost" in name else {"n_estimators": 10}
    adapter = instantiate_adapter(name, spec=spec, hyperparameters=overrides)
    adapter.fit(ds.panel, ds.y.astype(np.float64))
    with pytest.raises(ProbaUnsupportedError, match="do not produce class probabilities"):
        adapter.predict_proba(ds.panel)


@pytest.mark.parametrize("name", _GBM_ALL_NAMES)
def test_gbm_adapter_predict_quantiles_raises_unsupported(name: str) -> None:
    task = "binary" if name in _GBM_CLASSIFIER_NAMES else "regression_point"
    spec = _dataset_spec(task_type=task)
    adapter = instantiate_adapter(name, spec=spec)
    with pytest.raises(QuantilesUnsupportedError, match="do not produce quantile"):
        adapter.predict_quantiles(pd.DataFrame())


# --- hyperparameters forwarding ---------------------------------------------


@pytest.mark.parametrize("name", _GBM_ALL_NAMES)
def test_gbm_adapter_accepts_hyperparameters_dict(name: str) -> None:
    """The factory accepts a hyperparameters dict; an empty dict
    is the default. Each library's factory function applies its
    own defaults so the adapter is usable with no overrides."""
    task = "binary" if name in _GBM_CLASSIFIER_NAMES else "regression_point"
    spec = _dataset_spec(task_type=task)
    # Empty dict is the default-everything path; the underlying
    # estimator builds with the factory's defaults.
    adapter = instantiate_adapter(name, spec=spec, hyperparameters={})
    assert adapter is not None
    # Library-specific overrides via the dict; n_estimators (or
    # iterations) is sampled by EVERY library's HPO space, so we
    # use a value smaller than the default to verify routing.
    smaller: dict[str, Any] = {"iterations": 10} if "catboost" in name else {"n_estimators": 10}
    adapter = instantiate_adapter(name, spec=spec, hyperparameters=smaller)
    assert adapter is not None


# --- task-type filtering -----------------------------------------------------


def test_gbm_classifier_rejects_regression_task_at_fit() -> None:
    """The adapter's `fit` raises `ValueError` when called with a
    spec whose `task_type` is not in the adapter's
    `task_types`. The harness's outer gate normally skips such
    cells (the registry returns the typed reason), but the
    `fit` guard is the defense-in-depth backstop."""
    spec = _dataset_spec(task_type="regression_point")
    adapter = instantiate_adapter("lightgbm_classifier", spec=spec)
    with pytest.raises(ValueError, match="does not support task_type"):
        adapter.fit(pd.DataFrame(), np.array([]))


def test_gbm_regressor_rejects_classification_task_at_fit() -> None:
    spec = _dataset_spec(task_type="binary")
    adapter = instantiate_adapter("lightgbm_regressor", spec=spec)
    with pytest.raises(ValueError, match="does not support task_type"):
        adapter.fit(pd.DataFrame(), np.array([]))


# --- hyperparameters override reach -----------------------------------------


@pytest.mark.parametrize(
    ("name", "kw"),
    [
        ("lightgbm_classifier", "n_estimators"),
        ("xgboost_classifier", "n_estimators"),
        ("catboost_classifier", "iterations"),
    ],
)
def test_gbm_classifier_hyperparameter_override_reaches_underlying_estimator(
    name: str, kw: str
) -> None:
    """R1 qa-CRIT-2 close: the adapter forwards the
    `hyperparameters` dict through to the underlying sklearn-API
    estimator. The protocol-level test only instantiates the
    adapter without reading back the estimator's params, so a
    silent dropped-override could pass that test. Pin that an
    explicit value ACTUALLY reaches the estimator post-fit.

    Uses `7` as the sentinel value (smaller than every factory
    default; no library coerces it; well-formed for both
    `n_estimators` and `iterations`)."""
    from tests.benchmarks._fakes import register_all_fakes_and_get_panels

    register_all_fakes_and_get_panels()
    from benchmarks.registry import get_dataset, get_loader

    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-r1-override"))
    adapter = instantiate_adapter(name, spec=spec, hyperparameters={kw: 7})
    adapter.fit(ds.panel, ds.y)
    # Read back the underlying estimator's resolved parameter; the
    # sklearn-API `get_params()` is the contract every library
    # honors (LightGBM, XGBoost, CatBoost all expose it).
    est = adapter._est  # pyright: ignore[reportAttributeAccessIssue]
    assert est is not None
    params = est.get_params()  # pyright: ignore[reportAttributeAccessIssue]
    assert params[kw] == 7


# --- factory pre-init guard -------------------------------------------------


def test_gbm_adapter_missing_estimator_factory_raises_runtime_error() -> None:
    """R1 qa-IMP close: the `_estimator_factory is None` guard in
    `_GBMAdapter.fit` is the runtime backstop for a future
    subclass that forgets to set the factory in `__post_init__`.
    Pin the typed error so a regression here surfaces explicitly.

    Uses a bare subclass that overrides `__post_init__` to a no-op
    (the base's `__post_init__` blocks direct instantiation per
    R2 arch-I1; a subclass that forgets to call
    `super().__post_init__()` AND forgets to set the factory is
    the realistic regression scenario)."""
    from dataclasses import dataclass
    from typing import ClassVar

    from benchmarks.adapters.gbm import _GBMAdapter  # pyright: ignore[reportPrivateUsage]
    from benchmarks.config import TaskType

    @dataclass
    class _BareGBMAdapter(_GBMAdapter):
        name: ClassVar[str] = "_BareGBMAdapter"
        task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
        supports_proba: ClassVar[bool] = False

        def __post_init__(self) -> None:
            # Deliberately omits the `_estimator_factory` and
            # `_classifier` setup the registered concrete
            # subclasses perform; the base's TypeError guard is
            # bypassed because `type(self) is _BareGBMAdapter`.
            pass

    spec = _dataset_spec(task_type="binary")
    adapter = _BareGBMAdapter(spec=spec)
    with pytest.raises(RuntimeError, match="missing `_estimator_factory`"):
        adapter.fit(pd.DataFrame({"id": [1], "t": [0], "x": [0.0]}), np.array([0]))


def test_gbm_adapter_direct_base_instantiation_raises_type_error() -> None:
    """R2 arch-I1 close: the base `_GBMAdapter` is abstract; the
    `__post_init__` guard raises `TypeError` rather than letting
    a caller construct a half-configured instance that would
    pass the runtime `SeqSklearnAdapter` Protocol check."""
    from benchmarks.adapters.gbm import _GBMAdapter  # pyright: ignore[reportPrivateUsage]

    spec = _dataset_spec(task_type="binary")
    with pytest.raises(TypeError, match="abstract base"):
        _GBMAdapter(spec=spec)
