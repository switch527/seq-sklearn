"""Tests for the three-way split function (A5 / F2)."""

import numpy as np
import pytest

from seq_sklearn.data.splits import compute_three_way_split
from seq_sklearn.errors import ConfigError


def _panel(n_entities: int, per_entity: int) -> tuple[np.ndarray, np.ndarray]:
    entity_ids = np.repeat(np.arange(n_entities), per_entity)
    window_time_index = np.tile(np.arange(per_entity), n_entities)
    return entity_ids, window_time_index


def test_pairwise_intersections_empty_on_id_time_tuples() -> None:
    entity_ids, wti = _panel(3, 10)
    train, val, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.2,
        cal_fraction=0.2,
        val_split_strategy="time_ordered",
        calibration_set_provided=False,
    )

    def tuples(idx: np.ndarray) -> set[tuple[int, int]]:
        return {(int(entity_ids[i]), int(wti[i])) for i in idx}

    t, v, c = tuples(train), tuples(val), tuples(cal)
    assert t & v == set()
    assert t & c == set()
    assert v & c == set()
    assert len(t | v | c) == len(entity_ids)


def test_cal_is_exactly_last_fraction_per_entity() -> None:
    entity_ids, wti = _panel(2, 10)
    _, _, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.1,
        cal_fraction=0.3,
        val_split_strategy="time_ordered",
        calibration_set_provided=False,
    )
    for entity in (0, 1):
        cal_for_entity = sorted(int(wti[i]) for i in cal if entity_ids[i] == entity)
        assert cal_for_entity == [7, 8, 9]


def test_random_policy_warns_on_multiple_entities() -> None:
    entity_ids, wti = _panel(3, 5)
    with pytest.warns(UserWarning, match="leaks future information"):
        compute_three_way_split(
            entity_ids,
            wti,
            val_fraction=0.2,
            cal_fraction=0.2,
            val_split_strategy="random",
            calibration_set_provided=False,
        )


def test_random_policy_no_warn_single_entity() -> None:
    entity_ids, wti = _panel(1, 20)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train, val, cal = compute_three_way_split(
            entity_ids,
            wti,
            val_fraction=0.2,
            cal_fraction=0.2,
            val_split_strategy="random",
            calibration_set_provided=False,
        )
    assert len(train) + len(val) + len(cal) == 20


def test_provided_and_cal_fraction_positive_raises() -> None:
    entity_ids, wti = _panel(2, 5)
    with pytest.raises(ConfigError, match="cal_fraction"):
        compute_three_way_split(
            entity_ids,
            wti,
            val_fraction=0.2,
            cal_fraction=0.2,
            val_split_strategy="time_ordered",
            calibration_set_provided=True,
        )


def test_calibration_set_provided_cal_zero_returns_empty_cal() -> None:
    entity_ids, wti = _panel(2, 5)
    train, val, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.2,
        cal_fraction=0.0,
        val_split_strategy="time_ordered",
        calibration_set_provided=True,
    )
    assert cal.size == 0
    assert cal.dtype == np.dtype(int)
    assert len(train) + len(val) == 10


def test_calibration_none_threshold_false_collapses_cal_into_train() -> None:
    entity_ids, wti = _panel(2, 10)
    train, val, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.2,
        cal_fraction=0.0,
        val_split_strategy="time_ordered",
        calibration_set_provided=False,
    )
    assert cal.size == 0
    # With two entities of 10 windows and val_fraction 0.2, val takes the
    # two windows just before the (now-absent) calibration tail per
    # entity. Everything else, including the windows that would have been
    # calibration, falls into train.
    for entity in (0, 1):
        train_wti = sorted(int(wti[i]) for i in train if entity_ids[i] == entity)
        val_wti = sorted(int(wti[i]) for i in val if entity_ids[i] == entity)
        assert val_wti == [8, 9]
        assert train_wti == [0, 1, 2, 3, 4, 5, 6, 7]


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="window_time_index"):
        compute_three_way_split(
            np.array([0, 0, 1]),
            np.array([0, 1]),
            val_fraction=0.1,
            cal_fraction=0.1,
            val_split_strategy="time_ordered",
            calibration_set_provided=False,
        )


@pytest.mark.parametrize(
    ("vf", "cf"),
    [(-0.1, 0.1), (1.0, 0.1), (0.1, -0.1), (0.1, 1.0), (0.6, 0.5)],
)
def test_invalid_fractions_raise(vf: float, cf: float) -> None:
    entity_ids, wti = _panel(2, 5)
    with pytest.raises(ValueError, match="fraction"):
        compute_three_way_split(
            entity_ids,
            wti,
            val_fraction=vf,
            cal_fraction=cf,
            val_split_strategy="time_ordered",
            calibration_set_provided=False,
        )


def test_unsorted_window_time_index_handled() -> None:
    entity_ids = np.array([0, 0, 0, 0, 0])
    wti = np.array([4, 0, 3, 1, 2])  # shuffled
    _, _, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.0,
        cal_fraction=0.4,
        val_split_strategy="time_ordered",
        calibration_set_provided=False,
    )
    assert sorted(int(wti[i]) for i in cal) == [3, 4]


def test_single_window_entity_no_negative_slices() -> None:
    entity_ids = np.array([0, 1])
    wti = np.array([0, 0])
    train, val, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.6,
        cal_fraction=0.3,
        val_split_strategy="time_ordered",
        calibration_set_provided=False,
    )
    total = len(train) + len(val) + len(cal)
    assert total == 2


def test_random_with_zero_fractions_all_train() -> None:
    entity_ids, wti = _panel(1, 8)
    train, val, cal = compute_three_way_split(
        entity_ids,
        wti,
        val_fraction=0.0,
        cal_fraction=0.0,
        val_split_strategy="random",
        calibration_set_provided=False,
    )
    assert len(train) == 8
    assert val.size == 0
    assert cal.size == 0
