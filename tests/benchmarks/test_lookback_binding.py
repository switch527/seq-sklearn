"""Phase B10 / B3-followup: cross-consumer lookback binding.

`L_resolved` is the single source of truth bound to:
1. The splitter (`benchmarks.protocol.split.make_splitter`) (A9.1).
2. The seq_sklearn adapter's `tabular_config.lookback`.
3. The GBM / sklearn-passthrough featurizer
   (`benchmarks.protocol.featurize.lag_featurize`).
4. (B3-followup) The raw-MTS reshape for TSC adapters;
   aeon-blocked and not in B10's scope.

Before B10 the test had only two live consumers (splitter + deep
adapter); the GBM adapter (B10) is the third, so the cross-
consumer identity invariant now has enough surface to pin
meaningfully. The TSC consumer arrives with its B3-followup
branch.
"""

from pathlib import Path

import numpy as np
import pytest
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.protocol import lag_featurize, resolve_lookback
from benchmarks.registry import get_dataset, get_loader, instantiate_adapter

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


@pytest.fixture
def fake_panels() -> list[PanelDataset]:
    return register_all_fakes_and_get_panels()


def test_resolve_lookback_returns_spec_lookback_by_default() -> None:
    """Sanity baseline: the resolver returns `spec.lookback`
    verbatim when no override is supplied."""
    spec = DatasetSpec(
        name="ds_lb_test",
        task_type="binary",
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
        lookback=7,
        observation_cutoff_rule=None,
        densification_policy=None,
        positive_label=1,
        excluded=False,
        exclusion_reason=None,
        citation="LB 2026",
    )
    assert resolve_lookback(spec) == 7


def test_lookback_binds_featurizer_and_gbm_adapter_to_same_value(
    fake_panels: list[PanelDataset],
) -> None:
    """The featurizer's column count is bound to `resolve_lookback`;
    the GBM adapter's `_feature_columns` after fit must reflect
    the same lookback (one column per feature times L + the
    warm-up audit column). Pin the identity so a future driver-side
    override of L would propagate through both consumers
    consistently."""
    del fake_panels
    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-lb"))
    lookback = resolve_lookback(spec)
    # Direct featurizer call: one column per feature times the
    # resolved lookback, plus the `missing_lag_count` column.
    x = lag_featurize(ds.panel, spec)
    n_features = len(spec.feature_real_cols) + len(spec.feature_categorical_cols)
    assert x.shape[1] == n_features * lookback + 1
    # The GBM adapter's `_feature_columns` after fit must match
    # the featurizer's column set exactly (the adapter routes
    # through `lag_featurize` at fit time).
    adapter = instantiate_adapter(
        "lightgbm_classifier",
        spec=spec,
        hyperparameters={"n_estimators": 10},
    )
    adapter.fit(ds.panel, ds.y)
    feature_cols = adapter._feature_columns  # pyright: ignore[reportAttributeAccessIssue]
    assert feature_cols is not None
    assert set(feature_cols) == set(x.columns)


def test_lookback_perturbation_changes_featurizer_column_count(
    fake_panels: list[PanelDataset],
) -> None:
    """A `lookback_override` argument to the featurizer changes
    the column count by exactly `delta * n_features`. The
    invariant pins that the consumer-side override is the only
    way to deviate from `spec.lookback`; the resolver enforces
    the typing."""
    del fake_panels
    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-lb-override"))
    # Anchor the default and the override against `spec.lookback`
    # directly; `resolve_lookback` already returns `spec.lookback`
    # when no override is supplied (pinned above).
    x_default = lag_featurize(ds.panel, spec)
    x_override = lag_featurize(ds.panel, spec, lookback_override=spec.lookback + 1)
    n_features = len(spec.feature_real_cols) + len(spec.feature_categorical_cols)
    assert x_override.shape[1] - x_default.shape[1] == n_features


def test_seq_sklearn_adapter_binds_tabular_config_lookback_to_spec(
    fake_panels: list[PanelDataset],
) -> None:
    """R1 arch-I3 / qa-NIT close: consumer 2 of the
    cross-consumer identity invariant is the seq_sklearn deep
    adapter. The adapter constructs `TabularConfigParams(...,
    lookback=spec.lookback, ...)` in
    `benchmarks/adapters/seq_sklearn.py::_build_tabular_config`,
    so a regression that hardcoded a different lookback would
    silently produce a mismatched feature surface vs the
    splitter + GBM featurizer. Pin the identity directly."""
    del fake_panels
    from benchmarks.adapters.seq_sklearn import (
        _build_tabular_config,  # pyright: ignore[reportPrivateUsage]
    )

    spec = get_dataset("fake_binary")
    tab = _build_tabular_config(spec)
    assert tab.lookback == resolve_lookback(spec)
    assert tab.lookback == spec.lookback


def test_gbm_predict_proba_after_lookback_aware_featurize_aligns_with_panel(
    fake_panels: list[PanelDataset],
) -> None:
    """The metric layer's index alignment between predictions and
    the held-out target relies on the featurizer + predict
    chain returning one row per panel row. The seq_sklearn
    adapter's deep model honors this; pin that the GBM adapter
    does too via the featurizer."""
    del fake_panels
    spec = get_dataset("fake_binary")
    loader = get_loader("fake_binary")
    ds = loader(Path("/tmp/cache-b10-lb-align"))
    adapter = instantiate_adapter(
        "lightgbm_classifier",
        spec=spec,
        hyperparameters={"n_estimators": 10},
    )
    adapter.fit(ds.panel, ds.y)
    proba = adapter.predict_proba(ds.panel)
    pred = adapter.predict(ds.panel)
    assert proba.shape == (len(ds.panel), 2)
    assert pred.shape == (len(ds.panel),)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(ds.panel)), atol=1e-5)
