"""Phase B-followup GBM native-categorical-handling tests.

The GBM adapter routes the spec's
``feature_categorical_cols`` into each library's native
categorical-handling parameter rather than letting the
booster treat the int-encoded lag columns as ordinal
numerics. The tests pin:

- LightGBM receives ``categorical_feature=[<cat_lag_cols>]``
  at ``fit`` (covers all ``{cat}_lag{k}`` lag positions).
- CatBoost receives ``cat_features=[<cat_lag_cols>]`` at
  the estimator constructor.
- XGBoost receives ``enable_categorical=True`` at the
  estimator constructor AND every categorical lag column
  arrives at the booster with ``pd.Categorical`` dtype.
- Specs with empty ``feature_categorical_cols`` route
  through unchanged (no library-specific kwarg, no dtype
  conversion).

The categorical-lag-column naming convention is
``{cat_col}_lag{k}``; the adapter rebuilds the list from
the featurized X column order via
``_GBMAdapter._categorical_lag_columns``.
"""

from typing import Any

import numpy as np
import pandas as pd
from benchmarks.adapters.gbm import (
    _CatBoostClassifierAdapter,
    _LightGBMClassifierAdapter,
    _XGBoostClassifierAdapter,
)
from benchmarks.config import DatasetSpec


def _spec_with_categoricals() -> DatasetSpec:
    return DatasetSpec(
        name="cat_smoke",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="mixed",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("real_0",),
        feature_categorical_cols=("cat_0", "cat_1"),
        lookback=3,
        positive_label=1,
        citation="native-cat test",
    )


def _spec_without_categoricals() -> DatasetSpec:
    return DatasetSpec(
        name="numeric_smoke",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("real_0", "real_1"),
        feature_categorical_cols=(),
        lookback=3,
        positive_label=1,
        citation="numeric test",
    )


def _panel_with_categoricals(
    n_entities: int = 8, n_periods: int = 12
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for entity in range(n_entities):
        for t in range(n_periods):
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "real_0": float(rng.normal()),
                    "cat_0": rng.choice(["a", "b", "c"]),
                    "cat_1": int(rng.integers(0, 4)),
                    "y": int(entity % 2),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, panel["y"].to_numpy()


def _panel_numeric(n_entities: int = 8, n_periods: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for entity in range(n_entities):
        for t in range(n_periods):
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "real_0": float(rng.normal()),
                    "real_1": float(rng.normal()),
                    "y": int(entity % 2),
                }
            )
    panel = pd.DataFrame(rows)
    return panel, panel["y"].to_numpy()


# =============================================================================
# Helpers
# =============================================================================


def _expected_cat_lag_cols(cat_cols: tuple[str, ...], lookback: int) -> set[str]:
    return {f"{c}_lag{k}" for c in cat_cols for k in range(lookback)}


# =============================================================================
# LightGBM
# =============================================================================


def test_lightgbm_passes_categorical_feature_at_fit_when_cats_present() -> None:
    spec = _spec_with_categoricals()
    panel, y = _panel_with_categoricals()
    captured: dict[str, Any] = {}

    def _spy_factory(**kwargs: Any) -> Any:
        import lightgbm as lgb

        est = lgb.LGBMClassifier(n_estimators=5, verbose=-1, **kwargs)
        orig_fit = est.fit

        def fit(X: pd.DataFrame, y: np.ndarray, **fit_kwargs: Any) -> Any:  # noqa: N803
            captured["fit_kwargs"] = fit_kwargs
            captured["X_dtypes"] = dict(X.dtypes)
            return orig_fit(X, y, **fit_kwargs)

        est.fit = fit  # type: ignore[method-assign]
        return est

    adapter = _LightGBMClassifierAdapter(spec=spec)
    adapter._estimator_factory = _spy_factory
    adapter.fit(panel, y)

    assert "categorical_feature" in captured["fit_kwargs"]
    cat_kw = captured["fit_kwargs"]["categorical_feature"]
    assert set(cat_kw) == _expected_cat_lag_cols(spec.feature_categorical_cols, spec.lookback)


def test_lightgbm_omits_categorical_feature_when_no_cats() -> None:
    spec = _spec_without_categoricals()
    panel, y = _panel_numeric()
    captured: dict[str, Any] = {}

    def _spy_factory(**kwargs: Any) -> Any:
        import lightgbm as lgb

        est = lgb.LGBMClassifier(n_estimators=5, verbose=-1, **kwargs)
        orig_fit = est.fit

        def fit(X: pd.DataFrame, y: np.ndarray, **fit_kwargs: Any) -> Any:  # noqa: N803
            captured["fit_kwargs"] = fit_kwargs
            return orig_fit(X, y, **fit_kwargs)

        est.fit = fit  # type: ignore[method-assign]
        return est

    adapter = _LightGBMClassifierAdapter(spec=spec)
    adapter._estimator_factory = _spy_factory
    adapter.fit(panel, y)
    assert "categorical_feature" not in captured["fit_kwargs"]


# =============================================================================
# CatBoost
# =============================================================================


def test_catboost_passes_cat_features_at_init_when_cats_present() -> None:
    spec = _spec_with_categoricals()
    panel, y = _panel_with_categoricals()
    captured: dict[str, Any] = {}

    def _spy_factory(**kwargs: Any) -> Any:
        import catboost as cb

        captured["init_kwargs"] = dict(kwargs)
        return cb.CatBoostClassifier(
            iterations=5, verbose=False, allow_writing_files=False, **kwargs
        )

    adapter = _CatBoostClassifierAdapter(spec=spec)
    adapter._estimator_factory = _spy_factory
    adapter.fit(panel, y)

    assert "cat_features" in captured["init_kwargs"]
    cat_kw = captured["init_kwargs"]["cat_features"]
    assert set(cat_kw) == _expected_cat_lag_cols(spec.feature_categorical_cols, spec.lookback)


def test_catboost_omits_cat_features_when_no_cats() -> None:
    spec = _spec_without_categoricals()
    panel, y = _panel_numeric()
    captured: dict[str, Any] = {}

    def _spy_factory(**kwargs: Any) -> Any:
        import catboost as cb

        captured["init_kwargs"] = dict(kwargs)
        return cb.CatBoostClassifier(
            iterations=5, verbose=False, allow_writing_files=False, **kwargs
        )

    adapter = _CatBoostClassifierAdapter(spec=spec)
    adapter._estimator_factory = _spy_factory
    adapter.fit(panel, y)
    assert "cat_features" not in captured["init_kwargs"]


# =============================================================================
# XGBoost
# =============================================================================


def test_xgboost_sets_enable_categorical_and_pandas_categorical_dtype() -> None:
    spec = _spec_with_categoricals()
    panel, y = _panel_with_categoricals()
    captured: dict[str, Any] = {}

    def _spy_factory(**kwargs: Any) -> Any:
        import xgboost as xgb

        captured["init_kwargs"] = dict(kwargs)
        est = xgb.XGBClassifier(n_estimators=5, verbosity=0, tree_method="hist", **kwargs)
        orig_fit = est.fit

        def fit(X: pd.DataFrame, y: np.ndarray, **fit_kwargs: Any) -> Any:  # noqa: N803
            captured["X_dtypes"] = dict(X.dtypes)
            return orig_fit(X, y, **fit_kwargs)

        est.fit = fit  # type: ignore[method-assign]
        return est

    adapter = _XGBoostClassifierAdapter(spec=spec)
    adapter._estimator_factory = _spy_factory
    adapter.fit(panel, y)

    assert captured["init_kwargs"].get("enable_categorical") is True
    cat_lag_cols = _expected_cat_lag_cols(spec.feature_categorical_cols, spec.lookback)
    for col in cat_lag_cols:
        assert isinstance(captured["X_dtypes"][col], pd.CategoricalDtype), (
            f"XGBoost column {col!r} must arrive as pd.CategoricalDtype; "
            f"got {captured['X_dtypes'][col]}"
        )


def test_xgboost_skips_enable_categorical_and_dtype_when_no_cats() -> None:
    spec = _spec_without_categoricals()
    panel, y = _panel_numeric()
    captured: dict[str, Any] = {}

    def _spy_factory(**kwargs: Any) -> Any:
        import xgboost as xgb

        captured["init_kwargs"] = dict(kwargs)
        est = xgb.XGBClassifier(n_estimators=5, verbosity=0, tree_method="hist", **kwargs)
        orig_fit = est.fit

        def fit(X: pd.DataFrame, y: np.ndarray, **fit_kwargs: Any) -> Any:  # noqa: N803
            captured["X_dtypes"] = dict(X.dtypes)
            return orig_fit(X, y, **fit_kwargs)

        est.fit = fit  # type: ignore[method-assign]
        return est

    adapter = _XGBoostClassifierAdapter(spec=spec)
    adapter._estimator_factory = _spy_factory
    adapter.fit(panel, y)
    assert "enable_categorical" not in captured["init_kwargs"]
    # Every column is numeric; no pd.CategoricalDtype anywhere.
    assert not any(isinstance(d, pd.CategoricalDtype) for d in captured["X_dtypes"].values())


# =============================================================================
# Categorical-lag-column rebuild helper
# =============================================================================


def test_categorical_lag_columns_helper_matches_naming_convention() -> None:
    """``_categorical_lag_columns`` reads the featurized X column
    list and returns all ``{cat}_lag{k}`` entries for every cat
    column in the spec. The helper is the single source of truth
    for which columns each library's native cat-handling kwarg
    addresses."""
    spec = _spec_with_categoricals()
    panel, y = _panel_with_categoricals()
    adapter = _LightGBMClassifierAdapter(spec=spec)
    # Reach inside via the protected helper; the test is the
    # contract pin for the naming convention.
    x = adapter._featurize(panel)
    cat_lag_cols = adapter._categorical_lag_columns(x)
    assert set(cat_lag_cols) == _expected_cat_lag_cols(spec.feature_categorical_cols, spec.lookback)
    # Real-only spec returns an empty list.
    numeric_spec = _spec_without_categoricals()
    numeric_panel, _ = _panel_numeric()
    numeric_adapter = _LightGBMClassifierAdapter(spec=numeric_spec)
    numeric_x = numeric_adapter._featurize(numeric_panel)
    assert numeric_adapter._categorical_lag_columns(numeric_x) == []
    del y
