"""Internal input-validation helpers (per requirements F1.1 / F2).

These helpers are imported by :mod:`seq_sklearn.data.tabular_to_sequence`
and by every estimator's ``fit`` entry. v1.1 flips the y-shape contract in
exactly one place (:func:`check_y`); no other code touches y shape.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from seq_sklearn.errors import DataContractError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["check_columns", "check_y"]


def check_y(y: Any) -> np.ndarray:
    """Validate the target array.

    v1 accepts only single-output targets. Multi-output regression and
    multi-label classification are scheduled for v1.1 (per requirements
    F1.1); v1.1 will flip this validator in one place. Calling code MUST
    NOT inspect y shape elsewhere.

    Returns ``y`` coerced to a 1-D :class:`numpy.ndarray`.

    Raises
    ------
    ValueError
        If ``y`` is 2-D with more than one column.
    """
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(
            "seq-sklearn v1 supports single-output targets only. "
            "Multi-output regression and multi-label classification are "
            f"planned for v1.1. Got y with shape {arr.shape}."
        )
    return arr


def check_columns(
    X: pd.DataFrame,  # noqa: N803  # `X` is sklearn convention for feature matrices
    *,
    id_col: str,
    time_col: str,
    required: Sequence[str] = (),
) -> None:
    """Validate the per-row panel-input contract (F2).

    Checks performed in order:

    1. Every column in ``required`` exists in ``X``.
    2. The ``time_col`` dtype is one of the F2 supported types:
       :class:`numpy.datetime64`, pandas :class:`PeriodDtype`, or a signed
       integer numpy dtype.
    3. Within tz-aware datetime columns, tz-naive and tz-aware rows are
       NOT mixed.
    4. ``(id_col, time_col)`` is unique across all rows.

    Raises
    ------
    DataContractError
        With a message naming the first violation found.
    """
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise DataContractError(f"required columns missing from input: {missing!r}")
    if id_col not in X.columns:
        raise DataContractError(f"id_col {id_col!r} missing from input columns")
    if time_col not in X.columns:
        raise DataContractError(f"time_col {time_col!r} missing from input columns")

    time_series = X[time_col]
    dtype = time_series.dtype
    is_datetime = is_datetime64_any_dtype(dtype)
    is_period = isinstance(dtype, pd.PeriodDtype)
    # `dtype.kind == 'i'` is the pandas 3.0-compatible signed-integer check.
    # `pd_types.is_signed_integer_dtype` is deprecated in pandas 2.x and
    # scheduled for removal in 3.0; the kind check is stable across versions
    # (numpy dtype.kind has been part of numpy's public API since 1.x).
    is_signed_int = isinstance(dtype, np.dtype) and dtype.kind == "i"
    if not (is_datetime or is_period or is_signed_int):
        # Object-dtype columns (including ones containing pandas Timestamps)
        # land here because is_datetime64_any_dtype returns False for object
        # dtype. Callers must convert to datetime64 / Period / signed int
        # before passing.
        raise DataContractError(
            f"time_col {time_col!r} dtype {dtype!r} is not supported; "
            "must be datetime64, datetime64[ns, <tz>], pandas "
            "PeriodDtype, or a signed integer numpy dtype"
        )
    # tz-mixing within a single column: pandas blocks at ingest time for
    # datetime64-typed columns. The object-dtype path (which is where mixed
    # tz can sneak in) is rejected by the dtype check above. An explicit
    # tz-mixing error path in TabularToSequence's fit (Phase 2) catches the
    # remaining case where a caller constructs a mixed-tz column at runtime
    # via custom Series assembly.

    duplicates = X.duplicated(subset=[id_col, time_col], keep=False)
    if bool(duplicates.any()):
        first = X.loc[duplicates].iloc[0]
        raise DataContractError(
            f"duplicate ({id_col}, {time_col}) found at "
            f"({first[id_col]!r}, {first[time_col]!r}); the panel must be "
            "unique on the entity-time key"
        )
