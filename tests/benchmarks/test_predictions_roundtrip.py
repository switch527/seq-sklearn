"""Phase B6.2.1 predictions-persistence tests.

The B5 raw-loss driver writes one prediction shard per cell
alongside the metric shard; the B6 ensemble driver inner-joins
two models' shards on `panel_row_index`. The round-trip contract
this file pins:

- `write_predictions` accepts (panel_row_index, y_true, y_pred,
  optional y_proba) and writes a parquet shard.
- `load_predictions` reads the same shard back; the column set
  matches the input + `y_proba_{k}` columns when `y_proba` was
  supplied.
- `has_predictions` reports the existence of a shard.
- Length-mismatch inputs raise `ValueError`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.predictions import (
    has_predictions,
    load_predictions,
    predictions_dir,
    predictions_path,
    proba_columns,
    proba_matrix,
    write_predictions,
)


def test_write_predictions_classification_round_trip(tmp_path: Path) -> None:
    key = "ds_a__model_x__seed_0__default__fold_0"
    panel_row_index = np.array([10, 11, 12])
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    write_predictions(
        tmp_path,
        key,
        panel_row_index=panel_row_index,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    assert has_predictions(tmp_path, key)
    df = load_predictions(tmp_path, key)
    assert set(df.columns) == {
        "panel_row_index",
        "y_true",
        "y_pred",
        "y_proba_0",
        "y_proba_1",
    }
    np.testing.assert_array_equal(df["panel_row_index"].to_numpy(), panel_row_index)
    np.testing.assert_array_equal(df["y_true"].to_numpy(), y_true.astype(float))
    np.testing.assert_array_equal(df["y_pred"].to_numpy(), y_pred.astype(float))
    np.testing.assert_allclose(df["y_proba_0"].to_numpy(), y_proba[:, 0])
    np.testing.assert_allclose(df["y_proba_1"].to_numpy(), y_proba[:, 1])


def test_write_predictions_regression_omits_proba(tmp_path: Path) -> None:
    key = "ds_b__regressor__seed_0__default__fold_0"
    write_predictions(
        tmp_path,
        key,
        panel_row_index=np.array([0, 1, 2]),
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.5, 2.5, 2.0]),
    )
    df = load_predictions(tmp_path, key)
    assert set(df.columns) == {"panel_row_index", "y_true", "y_pred"}
    assert proba_columns(df) == ()
    assert proba_matrix(df) is None


def test_has_predictions_returns_false_before_write(tmp_path: Path) -> None:
    assert not has_predictions(tmp_path, "ds_a__model_x__seed_0__default__fold_0")


def test_load_predictions_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no shard"):
        load_predictions(tmp_path, "ds_a__model_x__seed_0__default__fold_0")


def test_write_predictions_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        write_predictions(
            tmp_path,
            "ds_a__model_x__seed_0__default__fold_0",
            panel_row_index=np.array([0, 1, 2]),
            y_true=np.array([0, 1]),  # too short
            y_pred=np.array([0, 1, 1]),
        )


def test_write_predictions_rejects_proba_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="y_proba length"):
        write_predictions(
            tmp_path,
            "ds_a__model_x__seed_0__default__fold_0",
            panel_row_index=np.array([0, 1, 2]),
            y_true=np.array([0, 1, 0]),
            y_pred=np.array([0, 1, 1]),
            y_proba=np.array([[0.5, 0.5], [0.5, 0.5]]),  # one short
        )


def test_proba_columns_returns_sorted_subset() -> None:
    df = pd.DataFrame(
        {
            "panel_row_index": [0],
            "y_true": [0],
            "y_pred": [0],
            "y_proba_2": [0.1],
            "y_proba_0": [0.3],
            "y_proba_1": [0.6],
        }
    )
    assert proba_columns(df) == ("y_proba_0", "y_proba_1", "y_proba_2")


def test_proba_matrix_extracts_in_class_order() -> None:
    df = pd.DataFrame(
        {
            "panel_row_index": [0, 1],
            "y_true": [0, 1],
            "y_pred": [0, 1],
            "y_proba_2": [0.1, 0.2],
            "y_proba_0": [0.3, 0.4],
            "y_proba_1": [0.6, 0.4],
        }
    )
    matrix = proba_matrix(df)
    assert matrix is not None
    assert matrix.shape == (2, 3)
    # Column 0 -> y_proba_0; column 2 -> y_proba_2.
    np.testing.assert_array_equal(matrix[:, 0], np.array([0.3, 0.4]))
    np.testing.assert_array_equal(matrix[:, 2], np.array([0.1, 0.2]))


def test_predictions_dir_and_path_are_consistent(tmp_path: Path) -> None:
    key = "ds_a__model_x__seed_0__default__fold_0"
    expected_dir = tmp_path / "predictions"
    expected_path = expected_dir / f"{key}.parquet"
    assert predictions_dir(tmp_path) == expected_dir
    assert predictions_path(tmp_path, key) == expected_path


# --- R1 / code-C1: numeric sort for `y_proba_{k}` columns ---


def test_proba_columns_sort_numerically_not_lexicographically(tmp_path: Path) -> None:
    # 12 classes -> y_proba_0 ... y_proba_11. Lexicographic sort
    # would order y_proba_10 BEFORE y_proba_2, corrupting the
    # (N, n_classes) matrix returned by `proba_matrix` and the
    # per-sample NLL the ensemble driver derives from it.
    n_classes = 12
    panel_row_index = np.array([0, 1])
    y_true = np.array([0, 11])
    y_pred = np.array([0, 11])
    proba = np.eye(n_classes)[y_true.astype(int)]  # one-hot per row
    key = "ds_multi__model_x__seed_0__default__fold_0"
    write_predictions(
        tmp_path,
        key,
        panel_row_index=panel_row_index,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=proba,
    )
    df = load_predictions(tmp_path, key)
    cols = proba_columns(df)
    # Expect numeric order: y_proba_0, y_proba_1, ..., y_proba_11.
    assert cols == tuple(f"y_proba_{k}" for k in range(n_classes))
    # And `proba_matrix` must round-trip the one-hot encoding,
    # which requires column-0 of the matrix to be y_proba_0.
    matrix = proba_matrix(df)
    assert matrix is not None
    np.testing.assert_array_equal(matrix, proba)
