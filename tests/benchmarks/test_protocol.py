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
    folds = list(splitter.split(panel))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_iter_folds_test_target_window_is_history_only_overlap() -> None:
    """B4.5 named test (strengthened per qa-I3): the only rows
    appearing in both train and test for a given fold must be the
    last `lookback - 1` rows of each entity's train segment (the
    A9.1 history overlap). Any other intersection is a leak.

    The assertion walks per-entity: each overlapping row's cycle
    must be strictly less than the minimum cycle of that entity's
    non-overlapping test rows.
    """
    lookback = 3
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter = make_splitter(spec, lookback=lookback, n_splits=4)
    for train_idx, test_idx in splitter.split(panel):
        test_set = set(test_idx.tolist())
        train_set = set(train_idx.tolist())
        overlap = test_set & train_set
        # Per-entity: every overlap row's cycle is strictly less
        # than the entity's first non-overlap test cycle.
        for entity_id in panel["entity_id"].unique():
            entity_mask = (panel["entity_id"] == entity_id).to_numpy()
            entity_positions = np.where(entity_mask)[0]
            entity_overlap = sorted(set(entity_positions) & overlap)
            entity_test_only = sorted(set(entity_positions) & (test_set - overlap))
            if not entity_overlap or not entity_test_only:
                continue
            max_overlap_cycle = panel.iloc[entity_overlap]["cycle"].max()
            min_test_only_cycle = panel.iloc[entity_test_only]["cycle"].min()
            assert max_overlap_cycle < min_test_only_cycle, (
                f"entity {entity_id}: overlap row at cycle "
                f"{max_overlap_cycle} is NOT a history-prefix row "
                f"(non-overlap test starts at cycle {min_test_only_cycle})"
            )


def test_iter_folds_short_entity_emits_user_warning() -> None:
    """The library's `EntityTimeSeriesSplit` emits an aggregated
    `UserWarning` when an entity has too few rows to participate in
    a split. Pin the warning shape so the protocol can rely on it."""
    spec = _make_spec()
    # Entity 1 has 5 rows; entity 2 has 4 rows. min_rows for
    # n_splits=4, lookback=3 is 4 + 1 + 0 + 2 = 7, so both
    # entities are dropped.
    panel = pd.DataFrame(
        {
            "entity_id": [1] * 5 + [2] * 4,
            "cycle": list(range(5)) + list(range(4)),
            "x": [1.0] * 9,
            "y": [0.0] * 9,
        }
    )
    splitter = make_splitter(spec, lookback=3, n_splits=4)
    with pytest.warns(UserWarning, match="dropped"):
        list(splitter.split(panel))


# ----------------------------------------------------------------- #
# Featurizer (B4.3)
# ----------------------------------------------------------------- #


def test_lag_featurize_one_row_per_panel_row() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=5)
    fe = lag_featurize(panel, spec, lookback_override=3)
    assert len(fe) == len(panel)


def test_lag_featurize_columns_have_expected_names() -> None:
    spec = _make_spec(feature_real_cols=("x",))
    panel = _make_panel(n_entities=2, n_periods=4)
    fe = lag_featurize(panel, spec, lookback_override=3)
    assert set(fe.columns) == {"x_lag0", "x_lag1", "x_lag2", "missing_lag_count"}


def test_lag_featurize_warmup_rows_are_zero_imputed() -> None:
    """Row 0 of each entity has no preceding rows; lag1 and lag2
    must be 0.0 for real features. Panel formula is
    `x = e*100 + t` so entity 1 (e=1) cycles 0..3 carry
    100, 101, 102, 103."""
    spec = _make_spec(feature_real_cols=("x",))
    panel = _make_panel(n_entities=1, n_periods=4)
    fe = lag_featurize(panel, spec, lookback_override=3)
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
    fe = lag_featurize(panel, spec, lookback_override=3)
    # Entity 2 starts at panel row 4. Its row-0 lag1 must be 0
    # (the warmup fill), NOT entity 1's row-3 value.
    assert fe.iloc[4]["x_lag1"] == 0.0
    assert fe.iloc[4]["x_lag2"] == 0.0
    assert fe.iloc[4]["missing_lag_count"] == 2


def test_lag_featurize_handles_categorical_columns() -> None:
    spec = _make_spec(feature_real_cols=("x",), feature_categorical_cols=("category",))
    panel = _make_panel(n_entities=2, n_periods=3, extra_categorical=True)
    fe = lag_featurize(panel, spec, lookback_override=2)
    # Categorical columns are int-encoded via pandas category codes
    # (sorted-unique order). Lag0 of entity-1 row-0 is entity-1's
    # own code; lag1 is the leading-window sentinel `-1`. Encoded
    # column dtype is int64 so LightGBM/XGBoost/CatBoost accept it
    # without an extra cast.
    assert "category_lag0" in fe.columns
    assert "category_lag1" in fe.columns
    assert fe.iloc[0]["category_lag1"] == -1
    assert fe["category_lag0"].dtype == np.int64
    assert fe["category_lag1"].dtype == np.int64


def test_lag_featurize_rejects_non_positive_lookback() -> None:
    spec = _make_spec()
    panel = _make_panel()
    with pytest.raises(ValueError, match=">= 1"):
        lag_featurize(panel, spec, lookback_override=0)


def test_lag_featurize_rejects_missing_entity_col() -> None:
    spec = _make_spec()
    panel = pd.DataFrame({"x": [1.0, 2.0], "cycle": [0, 1]})
    with pytest.raises(ValueError, match="entity_col"):
        lag_featurize(panel, spec, lookback_override=2)


def test_lag_featurize_rejects_missing_feature_col() -> None:
    spec = _make_spec(feature_real_cols=("missing",))
    panel = _make_panel()
    with pytest.raises(KeyError, match="missing"):
        lag_featurize(panel, spec, lookback_override=2)


def test_lag_featurize_empty_feature_cols_returns_warmup_only() -> None:
    """A spec with no time-varying features (GATED scaffold case)
    still produces the warmup column so the downstream GBM can fit
    on the warmup signal alone."""
    spec = _make_spec(feature_real_cols=(), feature_categorical_cols=())
    panel = _make_panel(n_entities=2, n_periods=3)
    fe = lag_featurize(panel, spec, lookback_override=2)
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
    spec = _make_spec(feature_real_cols=("x",), feature_categorical_cols=("category",))
    panel = _make_panel(n_entities=4, n_periods=10, extra_categorical=True)
    splitter = make_splitter(spec, lookback=lookback, n_splits=4)
    base = lag_featurize(panel, spec, lookback_override=lookback)
    for train_idx, test_idx in splitter.split(panel):
        # Identify a pivot in train that's outside every test row's
        # lookback window. The pivot's value affects featurization
        # at panel rows `[pivot, pivot+1, ..., pivot+L-1]` within
        # the SAME entity; restrict to a pivot whose entity has no
        # test row in that window.
        # Note: the window check below uses panel POSITIONS
        # (entity_positions) which is correct only because
        # `_make_panel` stores same-entity rows consecutively.
        test_set = set(test_idx.tolist())
        train_set = set(train_idx.tolist())
        pivot: int | None = None
        for candidate in sorted(train_set - test_set):
            entity_id = int(panel.at[candidate, "entity_id"])
            same_entity_mask = panel["entity_id"] == entity_id
            entity_positions = np.where(same_entity_mask.to_numpy())[0]
            window = set(int(p) for p in entity_positions if candidate <= p < candidate + lookback)
            if not (window & test_set):
                pivot = candidate
                break
        if pivot is None:
            continue  # no eligible pivot for this fold
        perturbed = panel.copy()
        perturbed.at[pivot, "x"] = 9999.0
        perturbed.at[pivot, "category"] = "Z"
        re_fe = lag_featurize(perturbed, spec, lookback_override=lookback)
        # Numeric AND categorical channels (qa-NEW-I2: cover the
        # categorical leakage path too).
        for col in (
            "x_lag0",
            "x_lag1",
            "x_lag2",
            "category_lag0",
            "category_lag1",
            "category_lag2",
        ):
            assert (
                base.loc[test_idx, col].to_numpy() == re_fe.loc[test_idx, col].to_numpy()
            ).all(), f"fold leakage at col {col!r} on pivot {pivot}"


# ----------------------------------------------------------------- #
# Fingerprint (B8.1)
# ----------------------------------------------------------------- #


def test_fingerprint_is_stable_under_identical_config() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter_a = make_splitter(spec, lookback=3, n_splits=4)
    splitter_b = make_splitter(spec, lookback=3, n_splits=4)
    fp_a = fingerprint_folds(splitter_a.split(panel))
    fp_b = fingerprint_folds(splitter_b.split(panel))
    assert fp_a == fp_b


def test_fingerprint_flips_under_lookback_perturbation() -> None:
    """qa-NEW-N1 direction-2 perturbation: `L_resolved + 1`
    produces a different fold layout, so the fingerprint differs."""
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter_a = make_splitter(spec, lookback=3, n_splits=4)
    splitter_b = make_splitter(spec, lookback=4, n_splits=4)
    fp_a = fingerprint_folds(splitter_a.split(panel))
    fp_b = fingerprint_folds(splitter_b.split(panel))
    assert fp_a != fp_b


def test_fingerprint_flips_under_n_splits_perturbation() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter_a = make_splitter(spec, lookback=3, n_splits=4)
    splitter_b = make_splitter(spec, lookback=3, n_splits=2)
    fp_a = fingerprint_folds(splitter_a.split(panel))
    fp_b = fingerprint_folds(splitter_b.split(panel))
    assert fp_a != fp_b


def test_fingerprint_is_canonical_64char_hex() -> None:
    spec = _make_spec()
    panel = _make_panel(n_entities=3, n_periods=20)
    splitter = make_splitter(spec, lookback=3, n_splits=4)
    fp = fingerprint_folds(splitter.split(panel))
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_is_order_independent_on_unsorted_indices() -> None:
    """qa-C1 mutation kill: passing a fold whose train/test arrays
    are in reverse memory order must produce the same fingerprint
    as the same fold with sorted arrays. The canonical-JSON
    serialization sorts on the way in, so the two are equivalent."""
    sorted_folds = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
    ]
    reversed_folds = [
        (np.array([3, 2, 1, 0]), np.array([5, 4])),
        (np.array([5, 4, 3, 2, 1, 0]), np.array([7, 6])),
    ]
    assert fingerprint_folds(sorted_folds) == fingerprint_folds(reversed_folds)


def test_fingerprint_raises_on_empty_fold_iterator() -> None:
    """arch-I2 / code-N1: an empty iterator yields no folds; the
    function raises rather than silently returning a sha256 of the
    empty list (which would be indistinguishable from any other
    empty run)."""
    with pytest.raises(ValueError, match="no folds produced"):
        fingerprint_folds(iter([]))


# ----------------------------------------------------------------- #
# Featurizer order precondition (code-I2 / arch-I3)
# ----------------------------------------------------------------- #


def test_lag_featurize_rejects_out_of_order_panel() -> None:
    """The featurizer's `groupby().shift(k)` is row-positional; a
    panel whose rows are not in non-decreasing per-entity time order
    would produce silently wrong lags. The function raises typed at
    the module boundary."""
    spec = _make_spec()
    panel = _make_panel(n_entities=2, n_periods=4)
    # Swap two consecutive rows within entity 1 so its cycle order
    # is [0, 1, 3, 2].
    shuffled = panel.copy()
    a_idx = shuffled.index[2]
    b_idx = shuffled.index[3]
    shuffled.loc[a_idx, "cycle"], shuffled.loc[b_idx, "cycle"] = (
        int(shuffled.loc[b_idx, "cycle"]),
        int(shuffled.loc[a_idx, "cycle"]),
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        lag_featurize(shuffled, spec, lookback_override=2)


def test_lag_featurize_rejects_missing_time_col() -> None:
    spec = _make_spec()
    panel = _make_panel().drop(columns=["cycle"])
    with pytest.raises(ValueError, match="time_col"):
        lag_featurize(panel, spec, lookback_override=2)


def test_lag_featurize_default_lookback_reads_spec() -> None:
    """B4.1b binding: when `lookback_override` is None, the
    featurizer reads `spec.lookback` so the deep model and the GBM
    consume the same L."""
    spec = _make_spec(lookback=2)
    panel = _make_panel(n_entities=2, n_periods=4)
    fe = lag_featurize(panel, spec)  # no override
    assert set(fe.columns) == {"x_lag0", "x_lag1", "missing_lag_count"}
