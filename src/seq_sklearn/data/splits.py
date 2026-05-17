"""Three-way (train / val / calibration) split function (A5 / F2).

Pure function so unit tests need no Trainer. The calibration fold is
the tail ``cal_fraction`` windows per entity by time ordinal, the
validation fold the preceding ``val_fraction``, the remainder training.
"""

import logging
import warnings
from typing import Literal

import numpy as np

from seq_sklearn.errors import ConfigError

__all__ = ["below_floor_mask", "compute_three_way_split", "window_time_index"]

logger = logging.getLogger(__name__)


def window_time_index(entity_ids: np.ndarray) -> np.ndarray:
    """Per-entity time ordinals from the contiguous emitted blocks (A5).

    ``TabularToSequence.transform`` emits each entity's windows
    contiguously and time-ascending, so the ordinal is
    ``concatenate([arange(n_i) for n_i in per_entity_counts])``. The
    ``np.unique`` count vector is value-sorted, which lines up with the
    emitted blocks only because ``transform`` assigns ``entity_id``
    codes in batch-emission order, making ``entity_id`` monotone
    non-decreasing. The precondition is a ``raise`` (not an ``assert``,
    which ``python -O`` strips) so a non-TTS caller passing an unordered
    ``entity_id`` fails loudly instead of silently corrupting the fold
    ordinals. Single implementation shared by the Trainer and the
    estimator so the calibration fold cannot drift between them.

    Raises:
        ValueError: ``entity_ids`` is not monotone non-decreasing.
    """
    if entity_ids.size == 0:
        return np.empty(0, dtype=int)
    if not bool(np.all(np.diff(entity_ids) >= 0)):
        raise ValueError(
            "entity_ids must be monotone non-decreasing; "
            "TabularToSequence.transform guarantees this by construction"
        )
    _values, counts = np.unique(entity_ids, return_counts=True)
    return np.concatenate([np.arange(c) for c in counts])


def below_floor_mask(entity_ids: np.ndarray, floor: int) -> np.ndarray:
    """Per-window mask of entities with fewer than ``floor`` windows.

    ``TabularToSequence.transform`` emits exactly one window per entity
    row and injects a sentinel target (``-1`` classification / ``NaN``
    regression) for entities whose row count is below
    ``min_periods_predict``. Both the Trainer (so train / val never fit
    a sentinel) and the estimator's calibration fold / predict path must
    exclude the same windows; this is the single implementation both
    delegate to so the rule cannot drift between them (the same
    single-source pattern as :func:`window_time_index`).
    """
    codes, counts = np.unique(entity_ids, return_counts=True)
    below = set(codes[counts < floor].tolist())
    return np.array([e in below for e in entity_ids], dtype=bool)


def compute_three_way_split(
    entity_ids: np.ndarray,
    window_time_index: np.ndarray,
    *,
    val_fraction: float,
    cal_fraction: float,
    val_split_strategy: Literal["time_ordered", "random"],
    calibration_set_provided: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx, cal_idx)`` integer index arrays.

    ``window_time_index[i]`` is the time ordinal of window ``i`` within
    its entity (0 = oldest valid window). The split uses it to take the
    tail per entity, so the caller need not pre-sort.

    Time-ordered policy: for each entity the tail ``cal_fraction``
    windows by ``window_time_index`` form the calibration fold, the
    preceding ``val_fraction`` form the validation fold, the remainder
    is training. Random policy ignores ``window_time_index`` and emits a
    :class:`UserWarning` when more than one unique ``entity_id`` is
    present (random splits leak future information on panel data).

    Return shapes:

    * ``calibration_set_provided=False`` and ``cal_fraction > 0``:
      ``cal_idx`` holds the tail ``cal_fraction`` windows per entity. It
      is empty only in the degenerate case where every entity is too
      short for ``round(cal_fraction * m)`` to reach 1, in which case a
      :class:`UserWarning` is emitted.
    * ``calibration_set_provided=True`` and ``cal_fraction == 0.0``:
      ``cal_idx`` is ``np.empty(0, dtype=int)``; calibration is supplied
      externally via the ``calibration_set`` keyword to ``fit``.
    * ``calibration_set_provided=False`` and ``cal_fraction == 0.0``:
      the F2 collapse rule. ``cal_idx`` is empty and the windows that
      would have been calibration are folded back into training (the
      caller passes ``cal_fraction=0.0`` for the
      ``calibration_strategy="none"`` and ``threshold_tuning=False``
      case).

    Raises:
        ConfigError: ``calibration_set_provided=True`` and
            ``cal_fraction > 0`` (the F2 conflict rule; the caller must
            set ``cal_fraction=0.0`` to opt into external calibration).
        ValueError: ``entity_ids`` and ``window_time_index`` differ in
            length, or a fraction is outside ``[0, 1)`` /
            ``val_fraction + cal_fraction >= 1``.
    """
    if calibration_set_provided and cal_fraction > 0:
        raise ConfigError(
            "calibration_set was provided and cal_fraction > 0; set "
            "cal_fraction=0.0 to opt into externally-supplied calibration"
        )
    if len(entity_ids) != len(window_time_index):
        raise ValueError(
            f"entity_ids length {len(entity_ids)} != window_time_index "
            f"length {len(window_time_index)}"
        )
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not 0.0 <= cal_fraction < 1.0:
        raise ValueError(f"cal_fraction must be in [0, 1), got {cal_fraction}")
    if val_fraction + cal_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + cal_fraction must be < 1, got {val_fraction + cal_fraction}"
        )

    n = len(entity_ids)
    if val_split_strategy == "random":
        return _random_split(n, entity_ids, val_fraction, cal_fraction)
    return _time_ordered_split(entity_ids, window_time_index, val_fraction, cal_fraction)


def _random_split(
    n: int,
    entity_ids: np.ndarray,
    val_fraction: float,
    cal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(np.unique(entity_ids)) > 1:
        warnings.warn(
            "random val_split_strategy on a panel with more than one entity "
            "leaks future information across the time axis",
            UserWarning,
            stacklevel=3,
        )
    order = np.arange(n)
    n_cal = round(cal_fraction * n)
    n_val = round(val_fraction * n)
    cal_idx = order[n - n_cal :] if n_cal else np.empty(0, dtype=int)
    val_idx = order[n - n_cal - n_val : n - n_cal] if n_val else np.empty(0, dtype=int)
    train_idx = order[: n - n_cal - n_val]
    if cal_fraction > 0 and cal_idx.size == 0:
        warnings.warn(
            f"cal_fraction={cal_fraction} but the panel is too short for "
            "round(cal_fraction * n) to reach 1; cal_idx is empty and the "
            "calibration windows folded back into train",
            UserWarning,
            stacklevel=3,
        )
    return train_idx, val_idx, cal_idx


def _time_ordered_split(
    entity_ids: np.ndarray,
    window_time_index: np.ndarray,
    val_fraction: float,
    cal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    for entity in np.unique(entity_ids):
        rows = np.flatnonzero(entity_ids == entity)
        order = rows[np.argsort(window_time_index[rows], kind="stable")]
        m = len(order)
        n_cal = round(cal_fraction * m)
        n_val = round(val_fraction * m)
        n_cal = min(n_cal, m)
        n_val = min(n_val, m - n_cal)
        cal_parts.append(order[m - n_cal :] if n_cal else np.empty(0, dtype=int))
        val_parts.append(order[m - n_cal - n_val : m - n_cal] if n_val else np.empty(0, dtype=int))
        train_parts.append(order[: m - n_cal - n_val])
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=int)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=int)
    cal_idx = np.concatenate(cal_parts) if cal_parts else np.empty(0, dtype=int)
    if cal_fraction > 0 and cal_idx.size == 0:
        warnings.warn(
            f"cal_fraction={cal_fraction} but every entity is too short for "
            "round(cal_fraction * m) to reach 1; cal_idx is empty and the "
            "calibration windows folded back into train",
            UserWarning,
            stacklevel=3,
        )
    return train_idx, val_idx, cal_idx
