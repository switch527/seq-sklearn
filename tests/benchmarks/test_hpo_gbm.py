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
    # R1 arch-CRIT-1 close: `bagging_temperature` is the sixth
    # CatBoost param that aligns the sampled dimensionality with
    # the registered `search_space_size=6` disclosure.
    assert "bagging_temperature" in kwargs


def test_sample_gbm_all_libraries_emit_six_searched_params() -> None:
    """R1 arch-CRIT-1 close: the disclosed `search_space_size=6`
    must hold per library, not just for the family. Pin that
    each library's sampler returns a dict of exactly 6 keys."""
    for name, task_types in (
        ("lightgbm_classifier", ("binary", "multiclass")),
        ("xgboost_classifier", ("binary", "multiclass")),
        ("catboost_classifier", ("binary", "multiclass")),
    ):
        kwargs = _sample_once(name, task_types)
        assert len(kwargs) == 6, (
            f"{name} emits {len(kwargs)} params; expected 6 for "
            f"_GBM_SEARCH_SPACE_SIZE alignment: got {sorted(kwargs)}"
        )


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


@pytest.mark.parametrize(
    ("library_pair", "fixed_params"),
    [
        (
            ("lightgbm_classifier", "lightgbm_regressor"),
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
            },
        ),
        (
            ("xgboost_classifier", "xgboost_regressor"),
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 1.0,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
            },
        ),
        (
            ("catboost_classifier", "catboost_regressor"),
            {
                "iterations": 100,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "bagging_temperature": 0.5,
            },
        ),
    ],
)
def test_sample_gbm_same_library_classifier_regressor_produce_identical_values_on_fixed_trial(
    library_pair: tuple[str, str], fixed_params: dict[str, Any]
) -> None:
    """R1 qa-IMP close: the same-library sampler is task-agnostic
    (the seq_sklearn comparator's F5 closure makes its sampler
    task-dependent; GBMs do not). Pin VALUE identity under a
    fixed trial (not just key-set equality) so a future refactor
    that accidentally wires the regressor to a different sampler
    function with the same keys would fail."""
    cls_name, reg_name = library_pair
    cls_spec = _model_spec(cls_name, ("binary", "multiclass"))
    reg_spec = _model_spec(reg_name, ("regression_point",))
    cls_ds = _dataset_spec(task_type="binary")
    reg_ds = _dataset_spec(task_type="regression_point")
    cls_trial = optuna.trial.FixedTrial(fixed_params)
    reg_trial = optuna.trial.FixedTrial(fixed_params)
    cls_kwargs = sample_gbm_hyperparameters(cls_trial, cls_spec, cls_ds)
    reg_kwargs = sample_gbm_hyperparameters(reg_trial, reg_spec, reg_ds)
    assert cls_kwargs == reg_kwargs


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
