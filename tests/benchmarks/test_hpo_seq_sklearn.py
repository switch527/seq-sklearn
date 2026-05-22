"""Phase B8 seq_sklearn HPO search-space tests.

`benchmarks/hpo/seq_sklearn.py` wraps the library's
`suggest_params` into a kwargs dict the registered adapter
consumes. These tests pin:

- The registry is populated for `family="seq_sklearn"` after the
  package's side-effect import.
- A single sample produces a dict that contains the
  STABLE-field keys (learning rate, dropout, model shape) and
  pins `task_type` to the dataset spec (overriding the library's
  sampled value).
- `tabular_config` and `advanced` are stripped (the adapter owns
  the former from the spec channel; the latter is empty in v1).
- Calling the sampler against a non-seq_sklearn family raises a
  typed error.
"""

from typing import Any

import optuna
import pytest
from benchmarks.config import DatasetSpec, ModelSpec
from benchmarks.hpo import HPO_REGISTRY, HPOFamilyNotRegisteredError, get_hpo_space
from benchmarks.hpo.seq_sklearn import sample_seq_sklearn_hyperparameters


def _dataset_spec(task_type: str = "binary") -> DatasetSpec:
    return DatasetSpec(
        name="ds_b8",
        task_type=task_type,
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
        lookback=8,
        observation_cutoff_rule=None,
        densification_policy=None,
        positive_label=1 if task_type == "binary" else None,
        excluded=False,
        exclusion_reason=None,
        citation="Test 2026",
    )


def _model_spec(name: str = "tft_classifier") -> ModelSpec:
    return ModelSpec(
        name=name,
        family="seq_sklearn",
        task_types=("binary", "multiclass"),
        supports_proba=True,
        reason="test",
    )


# --- registration ------------------------------------------------------------


def test_seq_sklearn_family_registered_after_import() -> None:
    assert "seq_sklearn" in HPO_REGISTRY
    space, sampler = get_hpo_space("seq_sklearn")
    assert space.family == "seq_sklearn"
    assert space.search_space_size > 0
    assert callable(sampler)


def test_seq_sklearn_space_description_is_non_empty() -> None:
    space, _ = get_hpo_space("seq_sklearn")
    assert len(space.description) > 0


# --- sample_seq_sklearn_hyperparameters ---------------------------------


def test_sample_seq_sklearn_returns_adapter_friendly_kwargs() -> None:
    spec = _dataset_spec(task_type="binary")
    model_spec = _model_spec("tft_classifier")
    study = optuna.create_study()
    trial = study.ask()
    kwargs = sample_seq_sklearn_hyperparameters(trial, model_spec, spec)
    # F5-sampled sub-configs in adapter-Params form.
    assert "loss" in kwargs
    assert "sampler" in kwargs
    assert "optimizer" in kwargs
    assert "calibration_strategy" in kwargs
    # TFT model-shape fields.
    assert "hidden_size" in kwargs
    assert "attention_heads" in kwargs
    assert "dropout" in kwargs
    assert "variable_selection_dropout" in kwargs
    assert "prediction_readout" in kwargs


def test_sample_seq_sklearn_strips_adapter_locked_keys() -> None:
    """Adapter binds `task_type`, `tabular_config`, `scheduler`,
    `verbose` from the dataset spec / harness defaults; the
    wrapper MUST strip them so the adapter's locked-keys guard
    does not reject the hyperparameters dict at construction."""
    spec = _dataset_spec(task_type="binary")
    model_spec = _model_spec("tft_classifier")
    study = optuna.create_study()
    trial = study.ask()
    kwargs = sample_seq_sklearn_hyperparameters(trial, model_spec, spec)
    assert "task_type" not in kwargs
    assert "tabular_config" not in kwargs
    assert "scheduler" not in kwargs
    assert "verbose" not in kwargs
    assert "advanced" not in kwargs


def test_sample_seq_sklearn_omits_quantiles_on_non_quantile_task() -> None:
    """`quantiles` is only meaningful on `regression_quantile`; the
    wrapper omits the key entirely on every other task so the
    parent validator's no-quantiles-on-non-quantile invariant is
    not tripped by a stray `None`."""
    spec = _dataset_spec(task_type="binary")
    model_spec = _model_spec("tft_classifier")
    study = optuna.create_study()
    trial = study.ask()
    kwargs = sample_seq_sklearn_hyperparameters(trial, model_spec, spec)
    assert "quantiles" not in kwargs


def test_sample_seq_sklearn_rejects_non_seq_sklearn_family() -> None:
    spec = _dataset_spec()
    bad_model = ModelSpec(
        name="some_baseline",
        family="gbm",
        task_types=("binary",),
        supports_proba=True,
        reason="test",
    )
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="must be 'seq_sklearn'"):
        sample_seq_sklearn_hyperparameters(trial, bad_model, spec)


def test_sample_seq_sklearn_rejects_unknown_model_name() -> None:
    spec = _dataset_spec()
    bad_model = ModelSpec(
        name="tft_unknown",  # not in _MODEL_CLASS_BY_NAME
        family="seq_sklearn",
        task_types=("binary",),
        supports_proba=True,
        reason="test",
    )
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(KeyError, match="has no class entry"):
        sample_seq_sklearn_hyperparameters(trial, bad_model, spec)


def test_sample_seq_sklearn_includes_quantiles_on_regression_quantile_task() -> None:
    """Positive arm of the quantile branch: when the dataset task
    IS regression_quantile, the wrapper must include the
    `quantiles` kwarg in the returned dict (the adapter validator
    requires it for the regression_quantile cell). qa-I2."""
    spec = _dataset_spec(task_type="regression_quantile")
    # `tft_regressor` is the seq_sklearn family's regression model.
    model_spec = ModelSpec(
        name="tft_regressor",
        family="seq_sklearn",
        task_types=("regression_point", "regression_quantile"),
        supports_proba=False,
        reason="test",
    )
    study = optuna.create_study()
    trial = study.ask()
    kwargs = sample_seq_sklearn_hyperparameters(trial, model_spec, spec)
    assert "quantiles" in kwargs
    assert kwargs["quantiles"] is not None


# --- registry typed errors ---------------------------------------------------


def test_get_hpo_space_unknown_family_raises_typed() -> None:
    with pytest.raises(HPOFamilyNotRegisteredError, match="no HPO search space"):
        get_hpo_space("tsc")  # not registered until B8-followup


def test_hpo_registration_is_frozen_dataclass() -> None:
    """qa-N3 + B9 arch-I5: `HPORegistration` is the single-record
    replacement for the prior dual-dict registry. The frozen
    contract is the guarantee that no caller can mutate the
    pair (space, sampler) after registration."""
    import dataclasses

    from benchmarks.hpo._base import HPORegistration

    def _sampler(trial: object, model_spec: object, dataset_spec: object) -> dict[str, Any]:
        del trial, model_spec, dataset_spec
        return {}

    from benchmarks.hpo._base import HPOSpace as _HPOSpace

    space = _HPOSpace(family="tsc", search_space_size=2, description="x")
    registration = HPORegistration(space=space, sampler=_sampler)  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        registration.space = space  # pyright: ignore[reportAttributeAccessIssue]


def test_register_hpo_space_conflict_raises() -> None:
    """qa-I3: registering a second `HPOSpace` for the same family
    with a different `search_space_size` (or sampler) must raise
    `ValueError`. Idempotent re-registration of an identical
    record is allowed; the conflict guard catches typos in
    downstream HPO modules."""
    from benchmarks.hpo._base import HPOSpace as _HPOSpace
    from benchmarks.hpo._base import register_hpo_space

    def _sampler1(trial: object, model_spec: object, dataset_spec: object) -> dict[str, Any]:
        del trial, model_spec, dataset_spec
        return {}

    def _sampler2(trial: object, model_spec: object, dataset_spec: object) -> dict[str, Any]:
        del trial, model_spec, dataset_spec
        return {}

    space_a = _HPOSpace(family="tsc", search_space_size=3, description="a")
    space_b = _HPOSpace(family="tsc", search_space_size=5, description="b")
    register_hpo_space(space_a, _sampler1)  # type: ignore[arg-type]
    # Same record + same sampler is idempotent; no raise.
    register_hpo_space(space_a, _sampler1)  # type: ignore[arg-type]
    # Different record (or sampler) raises.
    with pytest.raises(ValueError, match="already registered"):
        register_hpo_space(space_b, _sampler1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already registered"):
        register_hpo_space(space_a, _sampler2)  # type: ignore[arg-type]
