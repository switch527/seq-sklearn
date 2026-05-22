"""Phase B10 GBM family HPO tests.

Pin the per-library sampler dispatch, the search-space size
disclosure (B6.4.0), and the typed errors for unregistered
model names.
"""

from typing import Any

import optuna
import pytest
from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.hpo import HPO_REGISTRY, get_hpo_space
from benchmarks.hpo.gbm import sample_gbm_hyperparameters


def _dataset_spec(task_type: str = "binary") -> DatasetSpec:
    return DatasetSpec(
        name="ds_hpo_gbm",
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
        citation="HPO 2026",
    )


def _model_spec(name: str, task_types: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="gbm",
        task_types=task_types,  # type: ignore[arg-type]
        supports_proba="classifier" in name,
        reason="test",
    )


# --- registry ----------------------------------------------------------------


def test_gbm_family_registered_in_hpo_registry() -> None:
    assert "gbm" in HPO_REGISTRY
    space, sampler = get_hpo_space("gbm")
    assert space.family == "gbm"
    assert space.search_space_size > 0
    assert callable(sampler)


def test_gbm_search_space_size_is_disclosed_dimensionality() -> None:
    """B6.4.0 disclosure: the registered HPOSpace names the
    sampling dimensionality the report quotes."""
    space, _ = get_hpo_space("gbm")
    assert space.search_space_size == 6


def test_gbm_space_description_names_libraries() -> None:
    space, _ = get_hpo_space("gbm")
    desc = space.description
    assert "LightGBM" in desc or "lightgbm" in desc.lower()
    assert "XGBoost" in desc or "xgboost" in desc.lower()
    assert "CatBoost" in desc or "catboost" in desc.lower()


# --- per-library samplers ----------------------------------------------------


def _sample_once(name: str, task_types: tuple[str, ...]) -> dict[str, Any]:
    model_spec = _model_spec(name, task_types)
    ds_spec = _dataset_spec(task_type="binary" if "classifier" in name else "regression_point")
    study = optuna.create_study()
    trial = study.ask()
    return sample_gbm_hyperparameters(trial, model_spec, ds_spec)


def test_sample_gbm_lightgbm_keys() -> None:
    kwargs = _sample_once("lightgbm_classifier", ("binary", "multiclass"))
    assert "n_estimators" in kwargs
    assert "learning_rate" in kwargs
    assert "num_leaves" in kwargs
    assert "min_child_samples" in kwargs
    assert "reg_alpha" in kwargs
    assert "reg_lambda" in kwargs


def test_sample_gbm_xgboost_keys() -> None:
    kwargs = _sample_once("xgboost_classifier", ("binary", "multiclass"))
    assert "n_estimators" in kwargs
    assert "learning_rate" in kwargs
    assert "max_depth" in kwargs
    assert "min_child_weight" in kwargs
    assert "reg_alpha" in kwargs
    assert "reg_lambda" in kwargs


def test_sample_gbm_catboost_keys() -> None:
    kwargs = _sample_once("catboost_classifier", ("binary", "multiclass"))
    assert "iterations" in kwargs
    assert "learning_rate" in kwargs
    assert "depth" in kwargs
    assert "l2_leaf_reg" in kwargs
    assert "random_strength" in kwargs


def test_sample_gbm_regressors_use_same_library_search_space() -> None:
    """Per-library samplers are task-agnostic. The classifier and
    regressor for the same library produce the same kwarg key
    set (parameter values differ per trial but the names are
    library-bound, not task-bound)."""
    for cls, reg in (
        ("lightgbm_classifier", "lightgbm_regressor"),
        ("xgboost_classifier", "xgboost_regressor"),
        ("catboost_classifier", "catboost_regressor"),
    ):
        cls_kwargs = _sample_once(cls, ("binary", "multiclass"))
        reg_kwargs = _sample_once(reg, ("regression_point",))
        assert set(cls_kwargs) == set(reg_kwargs), (
            f"{cls} and {reg} sampled different kwarg key sets: "
            f"{sorted(set(cls_kwargs) ^ set(reg_kwargs))}"
        )


def test_sample_gbm_rejects_non_gbm_family() -> None:
    model_spec = ModelSpec(
        name="tft_classifier",
        family="seq_sklearn",
        task_types=("binary",),
        supports_proba=True,
        reason="test",
    )
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="must be 'gbm'"):
        sample_gbm_hyperparameters(trial, model_spec, _dataset_spec())


def test_sample_gbm_rejects_unknown_model_name() -> None:
    model_spec = ModelSpec(
        name="gbm_unknown",
        family="gbm",
        task_types=("binary",),
        supports_proba=True,
        reason="test",
    )
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(KeyError, match="no sampler entry"):
        sample_gbm_hyperparameters(trial, model_spec, _dataset_spec())
