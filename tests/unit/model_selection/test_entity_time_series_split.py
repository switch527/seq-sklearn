"""Tests for the entity-aware time-series CV splitter (A9.1)."""

import pandas as pd
import pytest

from seq_sklearn.model_selection.split import EntityTimeSeriesSplit


def _panel(n_entities: int, per_entity: int) -> pd.DataFrame:
    rows = []
    for entity in range(n_entities):
        for t in range(per_entity):
            rows.append(
                {
                    "id": entity,
                    "time": pd.Timestamp("2020-01-01") + pd.offsets.Day(t),
                    "v": float(t),
                }
            )
    return pd.DataFrame(rows)


class _EstimatorStub:
    def __init__(self, lookback: int) -> None:
        self._lookback = lookback

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"tabular_config__lookback": self._lookback}


class _AttrEstimatorStub:
    class _Cfg:
        lookback = 9

    tabular_config = _Cfg()


class _PlainAttrStub:
    lookback = 4


class _GetParamsWithoutKeyStub:
    """get_params is callable but lacks the lookback key; falls to attr."""

    lookback = 6

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"n_splits": 5}


def test_get_n_splits_returns_n_splits() -> None:
    assert EntityTimeSeriesSplit(n_splits=4).get_n_splits() == 4


def test_produces_n_splits() -> None:
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=2)
    panel = _panel(2, 40)
    splits = list(splitter.split(panel))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_train_test_overlap_exactly_lookback_minus_one_per_entity() -> None:
    lookback = 4
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=lookback)
    panel = _panel(2, 60)
    for train_idx, test_idx in splitter.split(panel):
        ids = panel["id"].to_numpy()
        for entity in (0, 1):
            train_e = {i for i in train_idx if ids[i] == entity}
            test_e = {i for i in test_idx if ids[i] == entity}
            assert len(train_e & test_e) == lookback - 1


def test_lookback_one_means_no_overlap() -> None:
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=1)
    panel = _panel(1, 40)
    for train_idx, test_idx in splitter.split(panel):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_gap_measured_in_windows() -> None:
    panel = _panel(1, 60)
    no_gap = EntityTimeSeriesSplit(n_splits=3, gap=0, lookback=2)
    with_gap = EntityTimeSeriesSplit(n_splits=3, gap=5, lookback=2)
    ng = list(no_gap.split(panel))
    wg = list(with_gap.split(panel))
    for (tr_ng, _), (tr_wg, _) in zip(ng, wg, strict=True):
        assert len(tr_wg) == len(tr_ng) - 5


def test_lookback_one_gap_nonzero() -> None:
    panel = _panel(1, 60)
    no_gap = EntityTimeSeriesSplit(n_splits=3, gap=0, lookback=1)
    with_gap = EntityTimeSeriesSplit(n_splits=3, gap=4, lookback=1)
    ng = list(no_gap.split(panel))
    wg = list(with_gap.split(panel))
    for (tr_ng, _), (tr_wg, te_wg) in zip(ng, wg, strict=True):
        # lookback=1 -> no train/test overlap; gap trims gap windows
        # off the train tail.
        assert set(tr_wg).isdisjoint(set(te_wg))
        assert len(tr_wg) == len(tr_ng) - 4


def test_max_train_size_caps_per_entity() -> None:
    panel = _panel(2, 60)
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=2, max_train_size=7)
    for train_idx, _ in splitter.split(panel):
        ids = panel["id"].to_numpy()
        for entity in (0, 1):
            count = sum(1 for i in train_idx if ids[i] == entity)
            assert count <= 7


def test_gap_and_max_train_size_together() -> None:
    panel = _panel(2, 60)
    splitter = EntityTimeSeriesSplit(n_splits=3, gap=3, lookback=2, max_train_size=5)
    for train_idx, _ in splitter.split(panel):
        ids = panel["id"].to_numpy()
        for entity in (0, 1):
            count = sum(1 for i in train_idx if ids[i] == entity)
            # gap trims the train tail first, then max_train_size caps
            # the remainder to its trailing 5 rows per entity.
            assert count <= 5


def test_short_entities_dropped_with_one_aggregated_warning() -> None:
    rows = []
    # Two long entities, three short ones.
    for entity in range(2):
        for t in range(50):
            rows.append(
                {"id": entity, "time": pd.Timestamp("2020-01-01") + pd.offsets.Day(t), "v": 1.0}
            )
    for entity in range(2, 5):
        for t in range(2):
            rows.append(
                {"id": entity, "time": pd.Timestamp("2020-01-01") + pd.offsets.Day(t), "v": 1.0}
            )
    panel = pd.DataFrame(rows)
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=4)
    with pytest.warns(UserWarning, match="were dropped from split") as records:
        splits = list(splitter.split(panel))
    # One aggregated warning per split() iteration -> n_splits warnings.
    assert len(records) == 3
    for _, test_idx in splits:
        ids = panel["id"].to_numpy()
        assert set(ids[i] for i in test_idx).issubset({0, 1})


def test_from_estimator_get_params() -> None:
    splitter = EntityTimeSeriesSplit.from_estimator(_EstimatorStub(7), n_splits=4)
    assert splitter.lookback == 7
    assert splitter.n_splits == 4


def test_from_estimator_attr_fallback() -> None:
    splitter = EntityTimeSeriesSplit.from_estimator(_AttrEstimatorStub())
    assert splitter.lookback == 9


def test_from_estimator_plain_lookback_attr() -> None:
    splitter = EntityTimeSeriesSplit.from_estimator(_PlainAttrStub())
    assert splitter.lookback == 4


def test_from_estimator_get_params_missing_key_falls_to_attr() -> None:
    splitter = EntityTimeSeriesSplit.from_estimator(_GetParamsWithoutKeyStub())
    assert splitter.lookback == 6


def test_from_estimator_undiscoverable_raises() -> None:
    with pytest.raises(ValueError, match="could not discover lookback"):
        EntityTimeSeriesSplit.from_estimator(object())


def test_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        EntityTimeSeriesSplit(n_splits=1)
    with pytest.raises(ValueError, match="gap"):
        EntityTimeSeriesSplit(gap=-1)
    with pytest.raises(ValueError, match="lookback"):
        EntityTimeSeriesSplit(lookback=0)
    with pytest.raises(ValueError, match="max_train_size"):
        EntityTimeSeriesSplit(max_train_size=0)


def test_missing_columns_raise() -> None:
    splitter = EntityTimeSeriesSplit(n_splits=2, lookback=2)
    with pytest.raises(ValueError, match="id_col"):
        list(splitter.split(pd.DataFrame({"time": [1, 2]})))
    with pytest.raises(ValueError, match="time_col"):
        list(splitter.split(pd.DataFrame({"id": [1, 2]})))


def test_custom_id_time_cols() -> None:
    panel = _panel(1, 40).rename(columns={"id": "cust", "time": "ts"})
    splitter = EntityTimeSeriesSplit(n_splits=2, lookback=2, id_col="cust", time_col="ts")
    splits = list(splitter.split(panel))
    assert len(splits) == 2


def test_unsorted_input_is_sorted_per_entity() -> None:
    panel = _panel(1, 40).sample(frac=1.0, random_state=0).reset_index(drop=True)
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=3)
    times = panel["time"].to_numpy()
    for train_idx, test_idx in splitter.split(panel):
        train_times = times[train_idx]
        # Train indices, read in yield order, are time-ascending per entity.
        assert list(train_times) == sorted(train_times)
        assert len(test_idx) > 0


def test_all_entities_dropped_yields_empty() -> None:
    rows = [
        {"id": e, "time": pd.Timestamp("2020-01-01") + pd.offsets.Day(t), "v": 1.0}
        for e in range(3)
        for t in range(2)
    ]
    panel = pd.DataFrame(rows)
    splitter = EntityTimeSeriesSplit(n_splits=3, lookback=10)
    with pytest.warns(UserWarning, match="were dropped"):
        splits = list(splitter.split(panel))
    for train_idx, test_idx in splits:
        assert train_idx.size == 0
        assert test_idx.size == 0


def test_gap_wider_than_train_segment_empties_train_not_leaks() -> None:
    # n_splits=2 -> 3 segments; min_rows = 2+1+4+1-1 = 7, so a 7-row
    # entity is NOT dropped. np.array_split(7, 3) -> [3, 2, 2]; split 0's
    # train is segment 0 (3 rows) and gap=4 > 3. The negative-slice bug
    # kept a 2-row prefix (rows inside the gap window leaking into
    # train); the clamp must empty it instead.
    panel = _panel(1, 7)
    splitter = EntityTimeSeriesSplit(n_splits=2, gap=4, lookback=1)
    splits = list(splitter.split(panel))
    train0, test0 = splits[0]
    assert train0.size == 0
    assert test0.size > 0
    for train_idx, test_idx in splits:
        # lookback=1 -> no history-only overlap; folds must be disjoint.
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
