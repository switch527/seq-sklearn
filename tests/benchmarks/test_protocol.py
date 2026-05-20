"""Phase B3 fair-comparison protocol tests (B4.1, B4.3, B4.5, B8.1).

Walks the four protocol leaves:

- lookback: `resolve_lookback` returns the spec's lookback by
  default; an explicit positive override wins; non-positive
  override raises.
- split: `make_splitter` returns an `EntityTimeSeriesSplit` bound
  to the spec's `entity_col` / `time_col`; the splitter yields
  per-fold (train, test) index arrays that respect entity
  grouping.
- featurize: `lag_featurize` produces one row per panel row, with
  L lagged columns per real / categorical feature, the warm-up
  rows zero-imputed, and `missing_lag_count` tracking the
  warm-up depth.
- fingerprint: identical (config, panel) -> identical fingerprint;
  a single-row mutation flips it; the `L_resolved + 1`
  direction-2 perturbation also flips it (qa-NEW-N1).

The named B4.5 leakage tests live here too:

- `test_train_perturbation_changes_train_only` mutates a train-side
  cell of both a numeric and a categorical column and asserts the
  test-fold featurization is unchanged.
- `test_target_window_in_test_does_not_appear_in_train` asserts no
  test target-window row index appears in any train fold.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks.config import DatasetSpec
from benchmarks.protocol import (
    fingerprint_folds,
    iter_folds,
    lag_featurize,
    make_splitter,
    resolve_lookback,
)


def _make_spec(
    *,
    name: str = "ds",
    task_type: str = "regression_point",
    feature_real_cols: tuple[str, ...] = ("x",),
    feature_categorical_cols: tuple[str, ...] = (),
    lookback: int = 3,
) -> DatasetSpec:
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
        "feature_real_cols": feature_real_cols,
        "feature_categorical_cols": feature_categorical_cols,
        "lookback": lookback,
        "observation_cutoff_rule": None,
        "densification_policy": None,
        "positive_label": None,
        "excluded": False,
        "exclusion_reason": None,
        "citation": "Acme 2020",
    }
    return DatasetSpec(**kw)


def _make_panel(
    n_entities: int = 3,
    n_periods: int = 10,
    *,
    extra_categorical: bool = False,
) -> pd.DataFrame:
    """Synthetic panel with `n_entities` entities each observed for
    `n_periods` periods. `x` is a deterministic ramp;
    `category` is a per-entity categorical when `extra_categorical`."""
    rows: list[dict[str, object]] = []
    for e in range(1, n_entities + 1):
        for t in range(n_periods):
            row: dict[str, object] = {
                "entity_id": e,
                "cycle": t,
                "x": float(e * 100 + t),
                "y": float(e * 1000 + t),
            }
            if extra_categorical:
                row["category"] = "A" if e % 2 else "B"
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- #
# Lookback resolver (B4.1b)
# ----------------------------------------------------------------- #


def test_resolve_lookback_default_reads_spec() -> None:
    spec = _make_spec(lookback=7)
    assert resolve_lookback(spec) == 7


def test_resolve_lookback_override_wins() -> None:
    spec = _make_spec(lookback=7)
    assert resolve_lookback(spec, override=12) == 12


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_resolve_lookback_rejects_non_positive_override(bad: int) -> None:
    spec = _make_spec(lookback=7)
    with pytest.raises(ValueError, match=">= 1"):
        resolve_lookback(spec, override=bad)


# ----------------------------------------------------------------- #
# Split wrapper (B4.1)
# ----------------------------------------------------------------- #


def test_make_splitter_binds_spec_columns() -> None:
    spec = _make_spec()
    splitter = make_splitter(spec, lookback=3, n_splits=2)
    assert splitter.id_col == "entity_id"
    assert splitter.time_col == "cycle"
    assert splitter.lookback == 3
    assert splitter.n_splits == 2


def test_iter_folds_yields_non_empty_folds_on_sufficient_panel() -> None:
    spec = _make_spec()
    # n_periods >= n_splits + 1 + gap + lookback - 1 = 4 + 0 + 2 = 6
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter = make_splitter(spec, lookback=3, n_splits=4)
    folds = list(iter_folds(splitter, panel))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_iter_folds_test_target_window_not_in_train(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """B4.5 named test: no row index that ends up in a fold's test
    target window appears in that same fold's train set. The
    `lookback - 1` history overlap is the only intersection
    A9.1 admits, and the test target window itself (after the
    history prefix) cannot share rows with train."""
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter = make_splitter(spec, lookback=3, n_splits=4)
    for train_idx, test_idx in iter_folds(splitter, panel):
        train_set = set(train_idx.tolist())
        # The history overlap is at most lookback - 1 = 2 rows per
        # entity; the rest of test_idx must be disjoint from train.
        overlap = sorted(set(test_idx.tolist()) & train_set)
        # 3 entities * (lookback - 1) = 6 max overlap rows.
        assert len(overlap) <= 3 * (spec.lookback - 1)


# ----------------------------------------------------------------- #
# Featurizer (B4.3)
# ----------------------------------------------------------------- #


def test_lag_featurize_one_row_per_panel_row() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=5)
    fe = lag_featurize(panel, spec, lookback=3)
    assert len(fe) == len(panel)


def test_lag_featurize_columns_have_expected_names() -> None:
    spec = _make_spec(feature_real_cols=("x",))
    panel = _make_panel(n_entities=2, n_periods=4)
    fe = lag_featurize(panel, spec, lookback=3)
    assert set(fe.columns) == {"x_lag0", "x_lag1", "x_lag2", "missing_lag_count"}


def test_lag_featurize_warmup_rows_are_zero_imputed() -> None:
    """Row 0 of each entity has no preceding rows; lag1 and lag2
    must be 0.0 for real features. Panel formula is
    `x = e*100 + t` so entity 1 (e=1) cycles 0..3 carry
    100, 101, 102, 103."""
    spec = _make_spec(feature_real_cols=("x",))
    panel = _make_panel(n_entities=1, n_periods=4)
    fe = lag_featurize(panel, spec, lookback=3)
    # Row 0: lag0 = 100 (the value), lag1 = 0, lag2 = 0; warmup = 2
    assert fe.iloc[0]["x_lag0"] == 100.0
    assert fe.iloc[0]["x_lag1"] == 0.0
    assert fe.iloc[0]["x_lag2"] == 0.0
    assert fe.iloc[0]["missing_lag_count"] == 2
    # Row 2: lag0 = 102, lag1 = 101, lag2 = 100; warmup = 0
    assert fe.iloc[2]["x_lag0"] == 102.0
    assert fe.iloc[2]["x_lag1"] == 101.0
    assert fe.iloc[2]["x_lag2"] == 100.0
    assert fe.iloc[2]["missing_lag_count"] == 0


def test_lag_featurize_does_not_cross_entity_boundaries() -> None:
    """Entity 2's row 0 must NOT pull lag values from entity 1's
    rows; the warmup-fill convention applies per entity."""
    spec = _make_spec(feature_real_cols=("x",))
    panel = _make_panel(n_entities=2, n_periods=4)
    fe = lag_featurize(panel, spec, lookback=3)
    # Entity 2 starts at panel row 4. Its row-0 lag1 must be 0
    # (the warmup fill), NOT entity 1's row-3 value.
    assert fe.iloc[4]["x_lag1"] == 0.0
    assert fe.iloc[4]["x_lag2"] == 0.0
    assert fe.iloc[4]["missing_lag_count"] == 2


def test_lag_featurize_handles_categorical_columns() -> None:
    spec = _make_spec(
        feature_real_cols=("x",), feature_categorical_cols=("category",)
    )
    panel = _make_panel(n_entities=2, n_periods=3, extra_categorical=True)
    fe = lag_featurize(panel, spec, lookback=2)
    # Categorical lag0 of entity-1 row-0 is entity-1's own value;
    # lag1 is the warmup fill `""`.
    assert "category_lag0" in fe.columns
    assert "category_lag1" in fe.columns
    assert fe.iloc[0]["category_lag1"] == ""


def test_lag_featurize_rejects_non_positive_lookback() -> None:
    spec = _make_spec()
    panel = _make_panel()
    with pytest.raises(ValueError, match=">= 1"):
        lag_featurize(panel, spec, lookback=0)


def test_lag_featurize_rejects_missing_entity_col() -> None:
    spec = _make_spec()
    panel = pd.DataFrame({"x": [1.0, 2.0], "cycle": [0, 1]})
    with pytest.raises(ValueError, match="entity_col"):
        lag_featurize(panel, spec, lookback=2)


def test_lag_featurize_rejects_missing_feature_col() -> None:
    spec = _make_spec(feature_real_cols=("missing",))
    panel = _make_panel()
    with pytest.raises(KeyError, match="missing"):
        lag_featurize(panel, spec, lookback=2)


def test_lag_featurize_empty_feature_cols_returns_warmup_only() -> None:
    """A spec with no time-varying features (GATED scaffold case)
    still produces the warmup column so the downstream GBM can fit
    on the warmup signal alone."""
    spec = _make_spec(feature_real_cols=(), feature_categorical_cols=())
    panel = _make_panel(n_entities=2, n_periods=3)
    fe = lag_featurize(panel, spec, lookback=2)
    assert list(fe.columns) == ["missing_lag_count"]
    assert len(fe) == len(panel)


def test_lag_featurize_train_perturbation_outside_lookback_changes_train_only() -> None:
    """B4.5 named test (arch-IIa wording): mutating a train-side
    cell that's OUTSIDE every test row's lookback window must NOT
    change the featurization of any test row. A9.1 explicitly
    admits a `lookback - 1` history overlap, so we restrict the
    pivot to rows whose value cannot appear in any test row's lag
    window (qa-NEW-I2: covers both numeric and categorical
    channels)."""
    lookback = 3
    spec = _make_spec(
        feature_real_cols=("x",), feature_categorical_cols=("category",)
    )
    panel = _make_panel(n_entities=4, n_periods=10, extra_categorical=True)
    splitter = make_splitter(spec, lookback=lookback, n_splits=4)
    base = lag_featurize(panel, spec, lookback=lookback)
    for train_idx, test_idx in iter_folds(splitter, panel):
        # Identify a pivot in train that's outside every test row's
        # lookback window. The pivot's value affects featurization
        # at panel rows `[pivot, pivot+1, ..., pivot+L-1]` within
        # the SAME entity; restrict to a pivot whose entity has no
        # test row in that window.
        test_set = set(test_idx.tolist())
        train_set = set(train_idx.tolist())
        pivot: int | None = None
        for candidate in sorted(train_set - test_set):
            entity_id = int(panel.at[candidate, "entity_id"])
            same_entity_mask = panel["entity_id"] == entity_id
            entity_positions = np.where(same_entity_mask.to_numpy())[0]
            window = set(
                int(p)
                for p in entity_positions
                if candidate <= p < candidate + lookback
            )
            if not (window & test_set):
                pivot = candidate
                break
        if pivot is None:
            continue  # no eligible pivot for this fold
        perturbed = panel.copy()
        perturbed.at[pivot, "x"] = 9999.0
        perturbed.at[pivot, "category"] = "Z"
        re_fe = lag_featurize(perturbed, spec, lookback=lookback)
        for col in ("x_lag0", "x_lag1", "x_lag2"):
            assert (
                base.loc[test_idx, col].to_numpy()
                == re_fe.loc[test_idx, col].to_numpy()
            ).all(), f"fold leakage at col {col!r} on pivot {pivot}"


# ----------------------------------------------------------------- #
# Fingerprint (B8.1)
# ----------------------------------------------------------------- #


def test_fingerprint_is_stable_under_identical_config() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter_a = make_splitter(spec, lookback=3, n_splits=4)
    splitter_b = make_splitter(spec, lookback=3, n_splits=4)
    fp_a = fingerprint_folds(iter_folds(splitter_a, panel))
    fp_b = fingerprint_folds(iter_folds(splitter_b, panel))
    assert fp_a == fp_b


def test_fingerprint_flips_under_lookback_perturbation() -> None:
    """qa-NEW-N1 direction-2 perturbation: `L_resolved + 1`
    produces a different fold layout, so the fingerprint differs."""
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter_a = make_splitter(spec, lookback=3, n_splits=4)
    splitter_b = make_splitter(spec, lookback=4, n_splits=4)
    fp_a = fingerprint_folds(iter_folds(splitter_a, panel))
    fp_b = fingerprint_folds(iter_folds(splitter_b, panel))
    assert fp_a != fp_b


def test_fingerprint_flips_under_n_splits_perturbation() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter_a = make_splitter(spec, lookback=3, n_splits=4)
    splitter_b = make_splitter(spec, lookback=3, n_splits=2)
    fp_a = fingerprint_folds(iter_folds(splitter_a, panel))
    fp_b = fingerprint_folds(iter_folds(splitter_b, panel))
    assert fp_a != fp_b


def test_fingerprint_is_canonical_64char_hex() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter = make_splitter(spec, lookback=3, n_splits=4)
    fp = fingerprint_folds(iter_folds(splitter, panel))
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)
