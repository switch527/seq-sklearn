"""Tests for the internal input validators at requirements F1.1 / F2."""

import numpy as np
import pandas as pd
import pytest

from seq_sklearn._validate import check_columns, check_y
from seq_sklearn.errors import DataContractError


class TestCheckY:
    def test_one_d_array_passes(self) -> None:
        y = np.array([0, 1, 0, 1])
        out = check_y(y)
        assert out.shape == (4,)

    def test_two_d_column_vector_is_flattened(self) -> None:
        y = np.array([[0], [1], [0], [1]])
        out = check_y(y)
        assert out.shape == (4,)

    def test_two_d_multi_output_rejected(self) -> None:
        y = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match=r"v1\.1"):
            check_y(y)

    def test_three_d_rejected(self) -> None:
        y = np.zeros((3, 2, 2))
        with pytest.raises(ValueError, match="single-output"):
            check_y(y)

    def test_list_input_is_coerced(self) -> None:
        out = check_y([0, 1, 2])
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [0, 1, 2]


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": pd.to_datetime(["2026-01-01", "2026-02-01"] * 2),
            "spend": [1.0, 2.0, 3.0, 4.0],
        }
    )


class TestCheckColumns:
    def test_legal_panel_passes(self, panel: pd.DataFrame) -> None:
        check_columns(panel, id_col="id", time_col="time", required=["spend"])

    def test_missing_required_column_raises(self, panel: pd.DataFrame) -> None:
        with pytest.raises(DataContractError, match="required columns missing"):
            check_columns(panel, id_col="id", time_col="time", required=["spend", "missing"])

    def test_missing_id_col_raises(self, panel: pd.DataFrame) -> None:
        with pytest.raises(DataContractError, match="id_col"):
            check_columns(panel, id_col="wrong_id", time_col="time")

    def test_missing_time_col_raises(self, panel: pd.DataFrame) -> None:
        with pytest.raises(DataContractError, match="time_col"):
            check_columns(panel, id_col="id", time_col="wrong_time")

    def test_object_dtype_time_rejected(self, panel: pd.DataFrame) -> None:
        bad = panel.copy()
        bad["time"] = bad["time"].astype(str)
        with pytest.raises(DataContractError, match="dtype"):
            check_columns(bad, id_col="id", time_col="time")

    def test_signed_int_time_accepted(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["a", "a", "b", "b"],
                "time": np.array([0, 1, 0, 1], dtype=np.int64),
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        check_columns(df, id_col="id", time_col="time")

    def test_period_time_accepted(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["a", "a"],
                "time": pd.period_range("2026-01", periods=2, freq="M"),
                "value": [1.0, 2.0],
            }
        )
        check_columns(df, id_col="id", time_col="time")

    def test_duplicate_id_time_pair_raises(self, panel: pd.DataFrame) -> None:
        dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        with pytest.raises(DataContractError, match="duplicate"):
            check_columns(dup, id_col="id", time_col="time")

    def test_unsigned_int_time_rejected(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["a", "a"],
                "time": np.array([0, 1], dtype=np.uint64),
                "value": [1.0, 2.0],
            }
        )
        with pytest.raises(DataContractError, match="dtype"):
            check_columns(df, id_col="id", time_col="time")
