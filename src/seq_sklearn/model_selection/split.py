"""Entity-aware expanding-window time-series splitter (A9.1 / F10).

Stable public class. Each entity's rows are sorted by time and chunked
into ``n_splits + 1`` segments; split ``i`` trains on segments ``0..i``
and tests on segment ``i + 1`` extended LEFT by ``lookback - 1`` rows so
``TabularToSequence.transform`` sees real history at test time instead
of padding every test entity.

A9.1 does not pin how the splitter learns the id / time columns. This
implementation defaults to ``"id"`` / ``"time"`` (matching
:class:`~seq_sklearn.config.tabular.TabularToSequenceConfig` defaults,
the F2-consistent choice) and accepts optional ``id_col`` / ``time_col``
keyword overrides.

Entities are iterated in stringified-id sort order here. This differs
from :class:`~seq_sklearn.data.tabular_to_sequence.TabularToSequence`,
which assigns entity codes in raw-value groupby order; the two are not
required to share an order because split correctness is per entity
(train/test disjointness holds independently of the concatenation
order of the per-entity chunks).
"""

import logging
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["EntityTimeSeriesSplit"]

logger = logging.getLogger(__name__)


class EntityTimeSeriesSplit:
    """Expanding-window CV splitter that respects entity grouping (A9.1).

    ``gap`` is measured in WINDOWS (rows), matching
    :class:`sklearn.model_selection.TimeSeriesSplit`. The test fold of
    split ``i`` is segment ``i + 1`` extended left by ``lookback - 1``
    rows per entity from the preceding train segment; train and test
    overlap by exactly ``lookback - 1`` rows per entity (history-only at
    test time, no test target spans the overlap). Entities with fewer
    than ``n_splits + 1 + gap + lookback - 1`` rows are dropped from a
    split with one aggregated :class:`UserWarning` per :meth:`split`
    call.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: int | None = None,
        lookback: int = 12,
        *,
        id_col: str = "id",
        time_col: str = "time",
    ) -> None:
        """Configure the splitter.

        Raises:
            ValueError: ``n_splits < 2``, ``gap < 0``, ``lookback < 1``,
                or ``max_train_size`` is non-positive.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError(f"max_train_size must be >= 1 when set, got {max_train_size}")
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.lookback = lookback
        self.id_col = id_col
        self.time_col = time_col

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,  # noqa: N803
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the configured number of splits.

        ``X`` / ``y`` / ``groups`` are unused; they exist so the class
        matches the sklearn splitter protocol used by ``cross_val_score``.
        """
        del X, y, groups
        return self.n_splits

    @classmethod
    def from_estimator(cls, estimator: Any, **kwargs: Any) -> "EntityTimeSeriesSplit":
        """Build a splitter reading ``lookback`` off ``estimator``.

        Tries ``estimator.get_params()["tabular_config__lookback"]``,
        then ``estimator.tabular_config.lookback``, then
        ``estimator.lookback``. Works on fitted or unfitted estimators.

        Raises:
            ValueError: None of the lookback access paths resolve.
        """
        lookback: int | None = None
        get_params = getattr(estimator, "get_params", None)
        if callable(get_params):
            params = get_params()
            if isinstance(params, dict) and "tabular_config__lookback" in params:
                lookback = params["tabular_config__lookback"]
        if lookback is None:
            tab = getattr(estimator, "tabular_config", None)
            if tab is not None and hasattr(tab, "lookback"):
                lookback = tab.lookback
        if lookback is None:
            lookback = getattr(estimator, "lookback", None)
        if lookback is None:
            raise ValueError(
                "could not discover lookback on the estimator; expose it via "
                "get_params()['tabular_config__lookback'], a tabular_config "
                "attribute, or a lookback attribute"
            )
        return cls(lookback=int(lookback), **kwargs)

    def split(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs over the panel.

        Indices are positions into ``X`` as passed (not pre-sorted). The
        splitter operates purely on the ``(entity_id, time)`` mapping; it
        does not call ``TabularToSequence.transform``. ``y`` / ``groups``
        are unused; they exist for sklearn splitter-protocol parity.

        Raises:
            ValueError: ``id_col`` or ``time_col`` missing from ``X``.
        """
        del y, groups
        if self.id_col not in X.columns:
            raise ValueError(f"id_col {self.id_col!r} missing from X columns")
        if self.time_col not in X.columns:
            raise ValueError(f"time_col {self.time_col!r} missing from X columns")

        positions = np.arange(len(X))
        id_values = X[self.id_col].to_numpy()
        time_values = X[self.time_col].to_numpy()
        per_entity_sorted: dict[str, np.ndarray] = {}
        for entity in sorted({str(v) for v in id_values}):
            mask = np.array([str(v) == entity for v in id_values], dtype=bool)
            entity_positions = positions[mask]
            order = np.argsort(time_values[mask], kind="stable")
            per_entity_sorted[entity] = entity_positions[order]

        min_rows = self.n_splits + 1 + self.gap + self.lookback - 1
        for split_i in range(self.n_splits):
            train_chunks: list[np.ndarray] = []
            test_chunks: list[np.ndarray] = []
            dropped = 0
            for sorted_pos in per_entity_sorted.values():
                m = len(sorted_pos)
                if m < min_rows:
                    dropped += 1
                    continue
                segments = np.array_split(sorted_pos, self.n_splits + 1)
                train_pos = np.concatenate(segments[: split_i + 1])
                if self.gap > 0:
                    # Clamp to zero: a gap wider than the train segment
                    # must empty it. An unclamped negative slice would
                    # instead keep a prefix, breaking the A9.1 gap
                    # separation.
                    train_pos = train_pos[: max(0, len(train_pos) - self.gap)]
                if self.max_train_size is not None:
                    train_pos = train_pos[-self.max_train_size :]
                test_segment = segments[split_i + 1]
                preceding_train = np.concatenate(segments[: split_i + 1])
                left_extension = (
                    preceding_train[-(self.lookback - 1) :]
                    if self.lookback > 1
                    else np.empty(0, dtype=int)
                )
                test_pos = np.concatenate([left_extension, test_segment])
                train_chunks.append(train_pos)
                test_chunks.append(test_pos)
            if dropped:
                warnings.warn(
                    f"{dropped} entity/entities had fewer than {min_rows} rows "
                    f"and were dropped from split {split_i}",
                    UserWarning,
                    stacklevel=2,
                )
            train_idx = np.concatenate(train_chunks) if train_chunks else np.empty(0, dtype=int)
            test_idx = np.concatenate(test_chunks) if test_chunks else np.empty(0, dtype=int)
            yield train_idx, test_idx
