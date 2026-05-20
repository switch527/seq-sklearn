"""Phase B2 adapter-protocol conformance tests.

Walks every registered model's adapter implementation and asserts:

- The adapter satisfies the runtime-checkable
  :class:`SeqSklearnAdapter` protocol.
- The `name` / `family` / `task_types` / `supports_proba` class
  attributes match the spec registered under the same name.
- `predict_proba` raises :class:`ProbaUnsupportedError` on adapters
  whose `supports_proba=False`.

The contract test is intentionally protocol-only (no `fit` is
exercised); end-to-end fit smoke for the seq-sklearn adapter is in
`tests/benchmarks/test_seq_sklearn_adapter_smoke.py` and is marked
`slow`.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.adapters.seq_sklearn import (
    SeqSklearnTFTClassifierAdapter,
    SeqSklearnTFTRegressorAdapter,
)
from benchmarks.config import DatasetSpec
from benchmarks.registry import get_model, list_models

from seq_sklearn import NotFittedError


def _make_minimal_dataset_spec(
    *, name: str = "ds", task_type: str = "regression_point"
) -> DatasetSpec:
    """Construct a `DatasetSpec` with the minimal field set required
    by the adapters; used for cheap protocol-conformance checks."""
    kw: dict[str, Any] = {
        "name": name,
        "task_type": task_type,
        "access_tier": "OPEN",
        "size_tier": "small",
        "balance": "balanced",
        "modality": "numeric",
        "source_uri": "https://example.com/data.csv",
        "integrity_sha256": "0" * 64,
        "archive_basename": "data.csv",
        "entity_col": "entity_id",
        "time_col": "cycle",
        "target_col": "y",
        "feature_real_cols": ("x",),
        "feature_categorical_cols": (),
        "lookback": 4,
        "observation_cutoff_rule": None,
        "densification_policy": None,
        "positive_label": 1 if task_type == "binary" else None,
        "excluded": False,
        "exclusion_reason": None,
        "citation": "Acme 2020",
    }
    return DatasetSpec(**kw)


def test_tft_classifier_adapter_satisfies_protocol() -> None:
    spec = _make_minimal_dataset_spec(task_type="binary")
    adapter = SeqSklearnTFTClassifierAdapter(spec=spec)
    assert isinstance(adapter, SeqSklearnAdapter)


def test_tft_regressor_adapter_satisfies_protocol() -> None:
    spec = _make_minimal_dataset_spec(task_type="regression_point")
    adapter = SeqSklearnTFTRegressorAdapter(spec=spec)
    assert isinstance(adapter, SeqSklearnAdapter)


def test_tft_classifier_adapter_metadata_matches_registered_spec() -> None:
    """The class-level adapter metadata must match the `ModelSpec`
    registered under the same name. A drift between them would let
    the harness select an adapter for a task type the spec rejects
    (or vice versa)."""
    spec = _make_minimal_dataset_spec(task_type="binary")
    adapter = SeqSklearnTFTClassifierAdapter(spec=spec)
    registered = get_model(adapter.name)
    assert adapter.family == registered.family
    assert adapter.task_types == registered.task_types
    assert adapter.supports_proba == registered.supports_proba


def test_tft_regressor_adapter_metadata_matches_registered_spec() -> None:
    spec = _make_minimal_dataset_spec(task_type="regression_point")
    adapter = SeqSklearnTFTRegressorAdapter(spec=spec)
    registered = get_model(adapter.name)
    assert adapter.family == registered.family
    assert adapter.task_types == registered.task_types
    assert adapter.supports_proba == registered.supports_proba


def test_tft_regressor_adapter_predict_proba_raises_typed() -> None:
    """`supports_proba=False` adapters must raise the typed
    `ProbaUnsupportedError`, not a generic exception."""
    spec = _make_minimal_dataset_spec(task_type="regression_point")
    adapter = SeqSklearnTFTRegressorAdapter(spec=spec)
    with pytest.raises(ProbaUnsupportedError, match="probabilities"):
        adapter.predict_proba(pd.DataFrame({"entity_id": [1], "cycle": [1]}))


def test_tft_classifier_adapter_predict_before_fit_raises_typed() -> None:
    """arch-I3: the adapter raises the library's typed
    `NotFittedError` (re-exported from `seq_sklearn`) so a caller
    can `except NotFittedError` across every adapter family."""
    spec = _make_minimal_dataset_spec(task_type="binary")
    adapter = SeqSklearnTFTClassifierAdapter(spec=spec)
    with pytest.raises(NotFittedError, match="before fit"):
        adapter.predict(pd.DataFrame())
    with pytest.raises(NotFittedError, match="before fit"):
        adapter.predict_proba(pd.DataFrame())


def test_tft_regressor_adapter_predict_before_fit_raises_typed() -> None:
    """Mirror of the classifier test for the regressor; closes
    the qa-I1 symmetric-coverage gap."""
    spec = _make_minimal_dataset_spec(task_type="regression_point")
    adapter = SeqSklearnTFTRegressorAdapter(spec=spec)
    with pytest.raises(NotFittedError, match="before fit"):
        adapter.predict(pd.DataFrame())


def test_tft_classifier_adapter_rejects_unsupported_task_type() -> None:
    """The adapter explicitly rejects a task type it does not
    support, rather than passing it through to the underlying
    estimator and getting a confusing downstream error."""
    spec = _make_minimal_dataset_spec(task_type="regression_point")
    adapter = SeqSklearnTFTClassifierAdapter(spec=spec)
    with pytest.raises(ValueError, match="does not support"):
        adapter.fit(
            pd.DataFrame({"entity_id": [1], "cycle": [1], "x": [0.0]}),
            np.array([0.5]),
        )


def test_tft_regressor_adapter_rejects_classification_task_type() -> None:
    spec = _make_minimal_dataset_spec(task_type="binary")
    adapter = SeqSklearnTFTRegressorAdapter(spec=spec)
    with pytest.raises(ValueError, match="does not support"):
        adapter.fit(
            pd.DataFrame({"entity_id": [1], "cycle": [1], "x": [0.0]}),
            np.array([0]),
        )


def test_tft_regressor_adapter_predict_quantiles_before_fit_raises_typed() -> None:
    spec = _make_minimal_dataset_spec(task_type="regression_quantile")
    adapter = SeqSklearnTFTRegressorAdapter(spec=spec)
    with pytest.raises(NotFittedError, match="before fit"):
        adapter.predict_quantiles(pd.DataFrame())


def test_tft_classifier_adapter_predict_quantiles_raises_typed() -> None:
    """code-C1 / arch-I1: classifier adapters raise the typed
    `QuantilesUnsupportedError` when asked for quantile
    predictions, mirroring the `predict_proba` pattern."""
    spec = _make_minimal_dataset_spec(task_type="binary")
    adapter = SeqSklearnTFTClassifierAdapter(spec=spec)
    with pytest.raises(QuantilesUnsupportedError, match="quantile"):
        adapter.predict_quantiles(pd.DataFrame({"entity_id": [1], "cycle": [1]}))


def test_quantiles_unsupported_error_is_subclass_of_not_implemented() -> None:
    """Parallel to qa-I3 / `ProbaUnsupportedError`: callers catching
    `NotImplementedError` also catch the typed subclass."""
    assert issubclass(QuantilesUnsupportedError, NotImplementedError)


@pytest.mark.parametrize(
    ("adapter_cls", "task_type"),
    [
        (SeqSklearnTFTClassifierAdapter, "binary"),
        (SeqSklearnTFTRegressorAdapter, "regression_point"),
    ],
)
def test_adapter_rejects_locked_hyperparameter_keys(
    adapter_cls: type, task_type: str
) -> None:
    """arch-C1: `hyperparameters` cannot override the
    bind-from-spec contract keys (`task_type`, `tabular_config`,
    `scheduler`, `verbose`). The check fires at `__post_init__`,
    not at `fit`, so a Phase B8 HPO driver that pipes
    `suggest_params` output verbatim still sees a clean
    construction-time error."""
    spec = _make_minimal_dataset_spec(task_type=task_type)
    for locked_key in ("task_type", "tabular_config", "scheduler", "verbose"):
        with pytest.raises(ValueError, match=locked_key):
            adapter_cls(spec=spec, hyperparameters={locked_key: "garbage"})


def test_tft_classifier_adapter_metadata_is_classvar_not_instance() -> None:
    """arch-I2 / code-C2 / arch-I4: the metadata
    (`name`, `family`, `task_types`, `supports_proba`) lives on the
    class, not the instance, so it cannot drift per-adapter-instance.
    The class attributes also satisfy the Protocol's declared
    `tuple[TaskType, ...]` shape (not the widened `tuple[str, ...]`).
    """
    spec = _make_minimal_dataset_spec(task_type="binary")
    adapter = SeqSklearnTFTClassifierAdapter(spec=spec)
    # Reading from class and instance both work; the value is the
    # same object.
    assert (
        SeqSklearnTFTClassifierAdapter.name
        == adapter.name
        == "tft_classifier"
    )
    assert SeqSklearnTFTClassifierAdapter.task_types == ("binary", "multiclass")


def test_every_registered_seq_sklearn_model_has_a_reason() -> None:
    """qa-N1 / B3.2.3: every model spec carries a non-empty
    `reason`; the harness shows this when listing applicable cells.
    """
    for name in list_models():
        spec = get_model(name)
        if spec.family != "seq_sklearn":
            continue
        assert spec.reason.strip(), (
            f"model {name!r}: empty `reason` field"
        )


def test_proba_unsupported_error_is_subclass_of_not_implemented() -> None:
    """qa-I3: ProbaUnsupportedError inherits from NotImplementedError
    so existing `except NotImplementedError` clauses also catch it."""
    assert issubclass(ProbaUnsupportedError, NotImplementedError)
