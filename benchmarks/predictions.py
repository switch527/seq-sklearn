"""Per-cell out-of-fold prediction persistence (B6.2.1).

The B5 raw-loss driver writes ONE parquet shard per cell with the
B4 metric record. B6 (ensemble) needs the raw predictions too so
the pairwise diversity statistics and error-correlation analysis
have access to per-sample outputs.

Layout: alongside `results/{cell_key}.parquet` (the metric shard)
the driver writes `predictions/{cell_key}.parquet` (the prediction
shard). The two are sibling artifacts produced atomically per cell;
the sentinel is written AFTER both, so a crash in either write path
re-runs the whole cell (B7.2 idempotence).

Prediction-shard schema:

- `panel_row_index`: int64, the positional index INTO the F2-contract
  panel for the row that the test fold included AND that survived
  the A9.1 lookback-floor strip. The B6.2 pairwise driver inner-
  joins two models' prediction shards on this column so different
  per-model strip decisions still align row-by-row.
- `y_true`: float64, the true target. Stored as float so a single
  shard schema covers classification (cast int -> float) and
  regression (already float).
- `y_pred`: float64, the model's point prediction.
- `y_proba_{k}`: float64, ONE column per probability class
  (classification only; absent on regression cells). The column
  count varies per dataset; `load_predictions` reads them by
  prefix-glob.
- `y_quantile_q{level}`: float64, one column per quantile level
  (regression_quantile only; B6-followup deferral. B6 ships the
  `panel_row_index` + `y_true` + `y_pred` triple only for the
  quantile case; the quantile-level columns land alongside the
  protocol-quantile_levels extension).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_PREDICTIONS_DIRNAME = "predictions"
_SHARD_SUFFIX = ".parquet"
_PROBA_COLUMN_PREFIX = "y_proba_"


def predictions_dir(root: Path) -> Path:
    """`{root}/predictions/`: the per-run prediction-shard directory."""
    return root / _PREDICTIONS_DIRNAME


def predictions_path(root: Path, key: str) -> Path:
    """Path of the prediction shard for cell `key`."""
    return predictions_dir(root) / f"{key}{_SHARD_SUFFIX}"


def has_predictions(root: Path, key: str) -> bool:
    """Return True iff a prediction shard exists for `key`.

    Used by the B6 ensemble driver to skip pairs where either model
    has no persisted predictions (e.g., a cell that was skipped at
    the metric level via `skipped_reason`).
    """
    return predictions_path(root, key).exists()


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write `df` as a parquet shard atomically (mirrors `manifest.py`)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


def write_predictions(
    root: Path,
    key: str,
    *,
    panel_row_index: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> None:
    """Persist one cell's predictions to `predictions/{key}.parquet`.

    Every array must have the SAME length (the post-strip number of
    rows in the test fold). `y_proba` is `None` for regression cells
    and for classification cells whose adapter has `supports_proba=
    False`; classification cells with probabilities write one
    `y_proba_{k}` column per class.

    Raises:
        ValueError: array length mismatch.
    """
    n = len(panel_row_index)
    if len(y_true) != n or len(y_pred) != n:
        raise ValueError(
            f"write_predictions: array length mismatch: "
            f"panel_row_index={n}, y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    if y_proba is not None and len(y_proba) != n:
        raise ValueError(
            f"write_predictions: y_proba length {len(y_proba)} does not "
            f"match panel_row_index length {n}"
        )
    columns: dict[str, np.ndarray] = {
        "panel_row_index": panel_row_index.astype(np.int64),
        "y_true": np.asarray(y_true).astype(np.float64),
        "y_pred": np.asarray(y_pred).astype(np.float64),
    }
    if y_proba is not None:
        proba = np.asarray(y_proba).astype(np.float64)
        for k in range(proba.shape[1]):
            columns[f"{_PROBA_COLUMN_PREFIX}{k}"] = proba[:, k]
    df = pd.DataFrame(columns)
    _atomic_write_parquet(df, predictions_path(root, key))


def load_predictions(root: Path, key: str) -> pd.DataFrame:
    """Read one cell's predictions.

    Returns the parquet's DataFrame verbatim. Columns are
    `panel_row_index`, `y_true`, `y_pred`, and a variable number of
    `y_proba_{k}` columns when the cell carried probabilities.

    Raises:
        FileNotFoundError: no shard exists for `key`. The caller
            should check `has_predictions` first when the absence
            is expected (e.g., a skipped cell).
    """
    path = predictions_path(root, key)
    if not path.exists():
        raise FileNotFoundError(
            f"load_predictions: no shard for cell {key!r} at {path}; "
            f"check `has_predictions(root, key)` before loading"
        )
    return pd.read_parquet(path)


def proba_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Return the `y_proba_*` column names sorted by the integer
    class index (NOT lexicographically).

    Lexicographic sort orders `y_proba_10` before `y_proba_2`,
    which silently corrupts the (N, n_classes) matrix returned by
    `proba_matrix` for any dataset with 10 or more classes (and
    therefore the per-sample NLL the ensemble driver computes from
    it). Numeric sort on the trailing integer keeps the columns
    aligned with the canonical class label range.
    """
    matches = [c for c in df.columns if c.startswith(_PROBA_COLUMN_PREFIX)]
    return tuple(sorted(matches, key=lambda c: int(c[len(_PROBA_COLUMN_PREFIX) :])))


def proba_matrix(df: pd.DataFrame) -> np.ndarray | None:
    """Extract the `(N, n_classes)` probability matrix from `df`.

    Returns `None` when no `y_proba_*` columns are present (regression
    cells, or classification cells whose adapter had supports_proba=
    False).
    """
    cols = proba_columns(df)
    if not cols:
        return None
    return df[list(cols)].to_numpy(dtype=np.float64)
