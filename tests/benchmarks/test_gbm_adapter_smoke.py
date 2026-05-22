"""Phase B10 GBM adapter end-to-end smoke tests.

Each test fits one GBM adapter against the synthetic
`fake_binary` (classifier) or `fake_regression_point` (regressor)
panel and asserts the predict + predict_proba shapes + dtypes
match the protocol contract. Marked `slow` so the per-PR CPU
loop can skip them and the nightly job includes them.

Together with `test_gbm_adapter.py` (the fast protocol pass)
this covers the design's named "every model in the v1
comparator set is callable via the protocol" deliverable test
for the GBM family.
"""

from pathlib import Path

import numpy as np
import pytest
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry import get_dataset, get_loader, instantiate_adapter

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

pytestmark = pytest.mark.slow


@pytest.fixture
def fake_panels() -> list[PanelDataset]:
    return register_all_fakes_and_get_panels()


_GBM_CLASSIFIERS: tuple[str, ...] = (
    "lightgbm_classifier",
    "xgboost_classifier",
    "catboost_classifier",
)
_GBM_REGRESSORS: tuple[str, ...] = (
    "lightgbm_regressor",
    "xgboost_regressor",
    "catboost_regressor",
)


@pytest.mark.parametrize("model_name", _GBM_CLASSIFIERS)
def test_gbm_classifier_adapter_fits_and_predicts_binary(
    fake_panels: list[PanelDataset], model_name: str
) -> None:
    """Smoke: fit + predict + predict_proba on the synthetic binary
    panel; assert shape + dtype + proba sums."""
    del fake_panels
    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-smoke"))
    # Speed knob: smaller GBM keeps the smoke fast.
    overrides: dict[str, object] = (
        {"iterations": 20} if "catboost" in model_name else {"n_estimators": 20}
    )
    adapter = instantiate_adapter(model_name, spec=spec, hyperparameters=overrides)
    adapter.fit(ds.panel, ds.y)
    pred = adapter.predict(ds.panel)
    proba = adapter.predict_proba(ds.panel)
    # Shape: one prediction per panel row.
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(ds.panel),)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(ds.panel), 2)
    # `predict_proba` rows must sum to 1 (within FP tolerance).
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(ds.panel)), atol=1e-5)


@pytest.mark.parametrize("model_name", _GBM_REGRESSORS)
def test_gbm_regressor_adapter_fits_and_predicts_regression_point(
    fake_panels: list[PanelDataset], model_name: str
) -> None:
    """Smoke: fit + predict on the synthetic regression-point panel;
    assert shape + dtype. predict_proba is unsupported per the
    protocol contract."""
    del fake_panels
    spec = get_dataset("fake_regression_point")
    loader = get_loader("fake_regression_point")
    ds = loader(Path("/tmp/cache-b10-smoke"))
    overrides: dict[str, object] = (
        {"iterations": 20} if "catboost" in model_name else {"n_estimators": 20}
    )
    adapter = instantiate_adapter(model_name, spec=spec, hyperparameters=overrides)
    adapter.fit(ds.panel, ds.y.astype(np.float64))
    pred = adapter.predict(ds.panel)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(ds.panel),)
    # Regressor predictions are floats (or at least real-valued).
    assert np.issubdtype(pred.dtype, np.number)


def test_lightgbm_classifier_classes_cache_is_stable_across_predict(
    fake_panels: list[PanelDataset],
) -> None:
    """The adapter caches `classes_` on fit so a subsequent
    `predict_proba` call returns columns in the same order as the
    sklearn convention (`np.unique(y)` ascending). Pin that two
    consecutive `predict_proba` calls produce identical column
    orderings."""
    del fake_panels
    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-stability"))
    adapter = instantiate_adapter(
        "lightgbm_classifier",
        spec=spec,
        hyperparameters={"n_estimators": 20},
    )
    adapter.fit(ds.panel, ds.y)
    proba_1 = adapter.predict_proba(ds.panel)
    proba_2 = adapter.predict_proba(ds.panel)
    # Bitwise identical: the underlying LightGBM is deterministic
    # at the same random_state seed.
    np.testing.assert_array_equal(proba_1, proba_2)


def test_gbm_adapter_uses_featurized_columns_not_raw_panel(
    fake_panels: list[PanelDataset],
) -> None:
    """After `fit` the adapter's `_feature_columns` cache must
    name the lag-feature column names, NOT the raw panel column
    names (entity_col / time_col / target_col). Pinning this
    catches a future refactor that accidentally feeds the raw
    panel through to the GBM."""
    del fake_panels
    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-cols"))
    adapter = instantiate_adapter(
        "lightgbm_classifier",
        spec=spec,
        hyperparameters={"n_estimators": 20},
    )
    adapter.fit(ds.panel, ds.y)
    feature_cols = adapter._feature_columns  # pyright: ignore[reportAttributeAccessIssue]
    assert feature_cols is not None
    # The raw panel's identity/time/target columns must NOT
    # appear in the GBM's feature set.
    assert spec.entity_col not in feature_cols
    assert spec.time_col not in feature_cols
    assert spec.target_col not in feature_cols
    # The lag columns + the warm-up audit column must appear.
    assert any(col.endswith("_lag0") for col in feature_cols)
    assert "missing_lag_count" in feature_cols
