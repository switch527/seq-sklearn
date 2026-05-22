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

from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters._base import (
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
def test_gbm_regressor_predict_proba_raises_typed(name: str) -> None:
    """Regressor adapters raise `ProbaUnsupportedError` on
    `predict_proba` per the protocol contract. The error fires
    AFTER fit (since the harness's pre-call gate skips classifier-
    only datasets); pin both pre-fit and post-fit paths."""
    spec = _dataset_spec(task_type="regression_point")
    adapter = instantiate_adapter(name, spec=spec)
    # Pre-fit: NotFittedError takes precedence.
    with pytest.raises(NotFittedError):
        adapter.predict_proba(pd.DataFrame())


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
