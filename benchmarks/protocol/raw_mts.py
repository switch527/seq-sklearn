"""Panel → 3D tensor reshape for the classical-TSC family (B4.4).

The TSC adapter family (`benchmarks/adapters/tsc.py`) consumes a
`(n_instances, n_channels, n_timesteps)` tensor; aeon's classifier
signature expects exactly that shape. This module is the protocol
leaf that produces it from the F2-contract panel.

Conventions:

- One INSTANCE per entity (entities appear in the order they first
  appear in the panel; the same convention as the splitter).
- `n_timesteps == L_resolved` from
  `benchmarks.protocol.lookback.resolve_lookback`.
- `n_channels == len(spec.feature_real_cols)`. The v1 reshape DROPS
  `spec.feature_categorical_cols` entirely (see D-B12.6 in the B12
  design delta): ordinal encoding violates ROCKET's metric-space
  assumption and one-hot inflates the channel count unpredictably.
  A panel whose channel set is entirely categorical raises
  `RawMTSError`.
- Entities with more than `L_resolved` rows are TRAILING-WINDOW
  clipped (last `L_resolved` rows kept).
- Entities with fewer than `L_resolved` rows are EXCLUDED from the
  tensor; the adapter emits `np.nan` for their panel rows at
  predict time, mirroring the deep model's A9.1 below-floor strip
  convention so the existing `_strip_below_floor_rows` machinery
  surfaces the drop count uniformly across all three adapter
  families.
- The tensor is cast to `np.float32` to halve host RAM cost (aeon
  classifiers accept float32 input natively).
"""

import numpy as np
import pandas as pd

from benchmarks.config import DatasetSpec
from benchmarks.protocol.lookback import resolve_lookback


class RawMTSError(RuntimeError):
    """Typed reshape failure for the raw-MTS data view.

    Raised by `panel_to_tensor` when the panel cannot be reshaped
    into a valid `(n_instances, n_channels, n_timesteps)` tensor:
    no real-valued channels declared on the spec, no entities
    survive the below-floor strip, the channel set has a non-
    numeric column, or the entity rows are not time-monotone.

    Subclasses `RuntimeError` so the B5/B8 driver's narrow
    adapter-error catch tuple (`(RuntimeError, MemoryError,
    NotFittedError, ProbaUnsupportedError,
    QuantilesUnsupportedError)`) routes the failure as a typed
    `adapter_error` skip rather than crashing the run. The
    failure shape IS a runtime data-shape error semantically,
    so the inheritance reflects intent rather than masking it.
    """


def panel_to_tensor(
    panel: pd.DataFrame,
    spec: DatasetSpec,
    *,
    override_lookback: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape `panel` into a 3D tensor for an aeon classifier.

    Returns `(X_3d, panel_row_to_instance)`:

    - `X_3d`: `np.float32` array of shape
      `(n_kept_instances, n_channels, L_resolved)`.
    - `panel_row_to_instance`: `np.int64` array of length
      `len(panel)` mapping each panel row to its instance index in
      `X_3d`. A value of `-1` indicates the row belongs to a
      below-floor entity (excluded from `X_3d`); the adapter
      broadcasts `np.nan` to those panel rows at predict time.

    Raises:
        RawMTSError: the spec declares zero real-valued channels,
            or no entity survives the below-floor strip.
    """
    channel_cols = list(spec.feature_real_cols)
    if not channel_cols:
        raise RawMTSError(
            f"panel_to_tensor: dataset {spec.name!r} declares zero real-valued "
            "feature columns (`spec.feature_real_cols` is empty); the v1 "
            "TSC reshape drops `feature_categorical_cols` per D-B12.6, so an "
            "all-categorical panel has no usable channels"
        )
    L = resolve_lookback(spec, override=override_lookback)  # noqa: N806  # tensor-time-dim convention

    entity_col = spec.entity_col
    time_col = spec.time_col

    panel_row_to_instance = np.full(len(panel), -1, dtype=np.int64)
    instance_blocks: list[np.ndarray] = []
    instance_idx = 0
    # `sort=False` preserves the first-appearance order of entities,
    # matching the splitter's convention.
    for _entity_id, group in panel.groupby(entity_col, sort=False):
        n_rows = len(group)
        if n_rows < L:
            # Below-floor entity: leave panel_row_to_instance at -1
            # for every row of this entity. The adapter broadcasts
            # NaN to those rows.
            continue
        # Validate per-entity time-monotonicity on the FULL group
        # BEFORE the trailing-window slice. The integer-position
        # `iloc[-L:]` reflects insertion order, not time order, so
        # an out-of-order group like [t=3, t=1, t=2] with L=2 would
        # silently yield `kept=[t=1, t=2]` (monotone) and feed the
        # wrong rows to the model. Match `lag_featurize`'s contract:
        # validate the source group, not the post-slice view.
        if not group[time_col].is_monotonic_increasing:
            raise RawMTSError(
                f"panel_to_tensor: panel rows for entity in column "
                f"{entity_col!r} are not in non-decreasing {time_col!r} "
                "order; sort the panel before reshape"
            )
        # Trailing-window clip if the entity has more than L rows.
        kept = group.iloc[-L:]
        # Build the (n_channels, L) block for this instance. Wrap
        # the float32 cast in a typed error: if the dataset spec
        # accidentally declared a non-numeric column under
        # `feature_real_cols`, numpy raises `ValueError`/`TypeError`
        # which would otherwise escape the B5/B8 driver's narrow
        # adapter-error tuple and crash the entire run.
        try:
            block = kept[channel_cols].to_numpy(dtype=np.float32, copy=False).T
        except (ValueError, TypeError) as exc:
            raise RawMTSError(
                f"panel_to_tensor: could not cast channels {channel_cols!r} "
                f"to np.float32 for dataset {spec.name!r}; verify that every "
                "column in `feature_real_cols` is numeric"
            ) from exc
        instance_blocks.append(block)
        # Map every panel row of this entity to this instance index
        # (the adapter broadcasts the per-instance prediction to ALL
        # of an entity's panel rows). Use the kept group's panel
        # positions, not the trailing-window slice's, so trailing
        # below-floor rows of a kept entity are still broadcast to.
        # (Every kept-entity row sees the same per-instance
        # prediction by F2.)
        try:
            panel_positions = panel.index.get_indexer(group.index)
        except pd.errors.InvalidIndexError as exc:
            # pandas raises InvalidIndexError on non-unique indices
            # before the negative-position guard below has a chance
            # to see them. Reraise as RawMTSError so the typed
            # contract holds.
            raise RawMTSError(
                "panel_to_tensor: the panel's row index must be unique so "
                "groupby's group.index labels resolve to a single panel "
                "position"
            ) from exc
        # Defense-in-depth: the InvalidIndexError wrap above catches
        # the only pandas-2.x path that produces -1 here (non-unique
        # panel index). Retained so a future pandas behavior change
        # (e.g. returning -1 instead of raising) still surfaces as
        # the typed RawMTSError. Removing the wrap above is NOT a
        # safe substitution for this guard.
        if (panel_positions < 0).any():
            raise RawMTSError(
                "panel_to_tensor: groupby produced rows not present in "
                "the panel's index; the panel must have a unique row "
                "index that matches groupby's group.index labels"
            )
        panel_row_to_instance[panel_positions] = instance_idx
        instance_idx += 1

    if not instance_blocks:
        raise RawMTSError(
            f"panel_to_tensor: no entity in dataset {spec.name!r} has at "
            f"least L_resolved={L} rows; every entity is below the lookback "
            "floor"
        )
    X_3d = np.stack(instance_blocks, axis=0)  # noqa: N806  # 3D tensor shape convention
    return X_3d, panel_row_to_instance


def instance_labels(
    y: np.ndarray,
    panel_row_to_instance: np.ndarray,
) -> np.ndarray:
    """Project per-row labels to one label per instance.

    For each instance index, take the LAST-period label among the
    panel rows assigned to that instance. Aeon classifiers expect
    one label per (n_instances,)-shaped vector; the panel carries
    one label per row.

    The "last-period" choice matches the trailing-window convention
    used in `panel_to_tensor`: the tensor's last timestep is the
    entity's most recent observation, so the matching label is the
    entity's most recent label.

    Returns a `(n_instances,)` array with `y.dtype`.

    Raises:
        RawMTSError: `panel_row_to_instance` contains no kept rows.
    """
    if len(y) != len(panel_row_to_instance):
        raise RawMTSError(
            f"instance_labels: y length {len(y)} does not match "
            f"panel_row_to_instance length {len(panel_row_to_instance)}"
        )
    kept_mask = panel_row_to_instance >= 0
    if not kept_mask.any():
        raise RawMTSError("instance_labels: every panel row is below-floor")
    n_instances = int(panel_row_to_instance[kept_mask].max()) + 1
    out = np.empty(n_instances, dtype=y.dtype)
    for instance_idx in range(n_instances):
        rows = np.flatnonzero(panel_row_to_instance == instance_idx)
        if rows.size == 0:
            raise RawMTSError(
                f"instance_labels: instance {instance_idx} has no panel rows; "
                "panel_row_to_instance is malformed"
            )
        # Last-period row of this instance: the highest panel position.
        out[instance_idx] = y[rows[-1]]
    return out


def broadcast_per_instance_to_per_row(
    per_instance_values: np.ndarray,
    panel_row_to_instance: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Broadcast a per-instance prediction back to per-row.

    Each row of `per_instance_values` (shape `(n_instances,)` for
    point predictions or `(n_instances, n_classes)` for proba) is
    expanded to every panel row of that instance. Below-floor rows
    (where `panel_row_to_instance == -1`) receive `fill_value`, so
    the B5 driver's `_strip_below_floor_rows` machinery counts the
    drop uniformly across adapter families.

    Returns shape `(len(panel_row_to_instance),)` or
    `(len(panel_row_to_instance), n_classes)`.
    """
    if per_instance_values.ndim == 1:
        out = np.full(len(panel_row_to_instance), fill_value, dtype=np.float64)
        kept_mask = panel_row_to_instance >= 0
        out[kept_mask] = per_instance_values[panel_row_to_instance[kept_mask]]
        return out
    if per_instance_values.ndim == 2:
        n_classes = per_instance_values.shape[1]
        out = np.full((len(panel_row_to_instance), n_classes), fill_value, dtype=np.float64)
        kept_mask = panel_row_to_instance >= 0
        out[kept_mask] = per_instance_values[panel_row_to_instance[kept_mask]]
        return out
    raise RawMTSError(
        f"broadcast_per_instance_to_per_row: per_instance_values.ndim must be "
        f"1 (point) or 2 (proba); got {per_instance_values.ndim}"
    )
