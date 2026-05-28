"""Phase B12 TSC HPO suggest tests.

The HPO suggest functions are pure pydantic + Optuna trials with
no aeon import, so they exercise without aeon installed.
"""

from typing import Any

import optuna
import pytest
from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.hpo import get_hpo_space
from benchmarks.hpo._base import HPOFamilyNotRegisteredError
from benchmarks.hpo.tsc import sample_tsc_hyperparameters


def _binary_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        name="ds_tsc_hpo",
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
        feature_real_cols=("x1",),
        feature_categorical_cols=(),
        lookback=3,
        positive_label=1,
        citation="tsc hpo test",
    )


def _model_spec(name: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="tsc",
        task_types=("binary", "multiclass"),
        supports_proba=True,
        reason="tsc hpo test",
    )


def _run_one_trial(name: str) -> dict[str, Any]:
    study = optuna.create_study(direction="minimize")
    sampled: dict[str, Any] = {}

    def _objective(trial: optuna.trial.BaseTrial) -> float:
        nonlocal sampled
        sampled = sample_tsc_hyperparameters(trial, _model_spec(name), _binary_dataset_spec())
        return 0.0

    study.optimize(_objective, n_trials=1)
    return sampled


# --- per-classifier shape ---------------------------------------------------


def test_rocket_sampler_returns_num_kernels_only() -> None:
    sampled = _run_one_trial("rocket_classifier")
    assert set(sampled) == {"num_kernels"}
    assert 500 <= sampled["num_kernels"] <= 20_000


def test_multirocket_sampler_returns_two_keys() -> None:
    sampled = _run_one_trial("multirocket_classifier")
    assert set(sampled) == {"num_kernels", "max_dilations_per_kernel"}
    assert 500 <= sampled["num_kernels"] <= 20_000
    assert 8 <= sampled["max_dilations_per_kernel"] <= 64


def test_catch22_sampler_returns_two_booleans() -> None:
    sampled = _run_one_trial("catch22_classifier")
    assert set(sampled) == {"outlier_norm", "replace_nans"}
    assert isinstance(sampled["outlier_norm"], bool)
    assert isinstance(sampled["replace_nans"], bool)


# --- family lookup + parity disclosure --------------------------------------


def test_get_hpo_space_returns_registered_tsc_space() -> None:
    """The TSC module's side-effect import registers a search space
    so the HPO uplift driver picks it up without changes."""
    space, sampler = get_hpo_space("tsc")
    assert space.family == "tsc"
    assert space.search_space_size == 2
    assert callable(sampler)


def test_get_hpo_space_unknown_family_raises_typed() -> None:
    """Sanity: the family lookup still raises on an unregistered
    family even though TSC is now registered."""
    with pytest.raises(HPOFamilyNotRegisteredError):
        get_hpo_space("sklearn_passthrough")


# --- error paths ------------------------------------------------------------


def test_sample_tsc_rejects_wrong_family() -> None:
    spec = ModelSpec(
        name="rocket_classifier",
        family="gbm",  # deliberately wrong
        task_types=("binary",),
        supports_proba=True,
        reason="wrong-family test",
    )
    study = optuna.create_study(direction="minimize")

    def _objective(trial: optuna.trial.BaseTrial) -> float:
        with pytest.raises(ValueError, match="must be 'tsc'"):
            sample_tsc_hyperparameters(trial, spec, _binary_dataset_spec())
        return 0.0

    study.optimize(_objective, n_trials=1)


def test_sample_tsc_unknown_model_name_raises_keyerror() -> None:
    spec = ModelSpec(
        name="not_a_real_tsc_model",
        family="tsc",
        task_types=("binary",),
        supports_proba=True,
        reason="unknown-name test",
    )
    study = optuna.create_study(direction="minimize")

    def _objective(trial: optuna.trial.BaseTrial) -> float:
        with pytest.raises(KeyError, match="no sampler entry"):
            sample_tsc_hyperparameters(trial, spec, _binary_dataset_spec())
        return 0.0

    study.optimize(_objective, n_trials=1)
