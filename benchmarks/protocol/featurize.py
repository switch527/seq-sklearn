"""Panel → flat tabular X for GBM/sklearn-passthrough adapters (B4.3).

GBM and sklearn-API tabular models take a 2-D feature matrix, not a
panel. The featurizer takes an F2-contract panel + a resolved
lookback `L` and produces, for each row of the panel, a flat
feature vector consisting of:

- For each `time_varying_real` column: `L` lagged values
  (`{col}_lag0`, `{col}_lag1`, ..., `{col}_lag(L-1)`) where `lag0`
  is the current row's value, `lag1` is one period earlier in the
  same entity, etc.
- For each `time_varying_categorical` column: the same L lagged
  values as raw values (the caller's encoder converts them); GBMs
  handle high-cardinality categoricals natively when configured
  to.

Rows whose entity has fewer than `L` preceding periods (the
"warm-up" rows) are zero-imputed for the missing lags and a typed
`missing_lag_count` column is appended so the GBM can learn the
warm-up signal explicitly. Static columns are NOT lagged (they
are entity-constant); they appear as a single column each. F2
invariant: the output is one row per panel row, ordered identically
to the input.

The transform is pure (no fit state): it consumes the spec, the
panel, and L. Phase B3-followup (with the GBM adapter) will wrap
this in a sklearn-API `FunctionTransformer` for `Pipeline`
composition.
"""

import numpy as np
import pandas as pd

from benchmarks.config import DatasetSpec


def _lag_real_column(
    panel: pd.DataFrame,
    col: str,
    entity_col: str,
    lookback: int,
) -> pd.DataFrame:
    """Produce L lagged columns for one real-valued column.

    Lag k for row i is the value at `i - k` within the same entity,
    or 0.0 if the entity has fewer than `k+1` preceding rows.
    """
    grouped = panel.groupby(entity_col, sort=False)[col]
    lagged_columns: dict[str, np.ndarray] = {}
    for k in range(lookback):
        shifted = grouped.shift(k)
        lagged_columns[f"{col}_lag{k}"] = np.asarray(
            shifted.fillna(0.0).to_numpy(), dtype=np.float64
        )
    return pd.DataFrame(lagged_columns, index=panel.index)


def _lag_categorical_column(
    panel: pd.DataFrame,
    col: str,
    entity_col: str,
    lookback: int,
) -> pd.DataFrame:
    """Produce L lagged categorical columns for one column.

    Lag k for row i is the raw value at `i - k` within the same
    entity, or the empty string `""` if the entity has fewer than
    `k+1` preceding rows. Caller encodes (LightGBM handles strings
    natively; XGBoost / CatBoost require an explicit cast).
    """
    grouped = panel.groupby(entity_col, sort=False)[col]
    lagged_columns: dict[str, object] = {}
    for k in range(lookback):
        shifted = grouped.shift(k)
        lagged_columns[f"{col}_lag{k}"] = shifted.fillna("").to_numpy()
    return pd.DataFrame(lagged_columns, index=panel.index)


def lag_featurize(
    panel: pd.DataFrame, spec: DatasetSpec, lookback: int
) -> pd.DataFrame:
    """Build the flat lag-feature matrix.

    Returns a DataFrame with one row per panel row, columns
    ``{col}_lag{k}`` for each `(time-varying col, k in [0, L))`
    plus one ``missing_lag_count`` integer column counting how many
    of the L lags would have been zero-imputed for that row
    (zero-imputed lags include both real `0.0` substitutes and
    categorical `""` substitutes). The output index matches the
    input ``panel.index``.

    Raises:
        ValueError: ``lookback < 1`` or the entity column is
            missing from ``panel``.
        KeyError: any of the spec's feature columns is missing from
            ``panel``.
    """
    if lookback < 1:
        raise ValueError(f"lag_featurize: lookback must be >= 1; got {lookback}")
    if spec.entity_col not in panel.columns:
        raise ValueError(
            f"lag_featurize: entity_col {spec.entity_col!r} missing from panel"
        )
    missing = [
        c
        for c in (*spec.feature_real_cols, *spec.feature_categorical_cols)
        if c not in panel.columns
    ]
    if missing:
        raise KeyError(
            f"lag_featurize: feature columns missing from panel: {missing}"
        )

    blocks: list[pd.DataFrame] = []
    for col in spec.feature_real_cols:
        blocks.append(_lag_real_column(panel, col, spec.entity_col, lookback))
    for col in spec.feature_categorical_cols:
        blocks.append(
            _lag_categorical_column(panel, col, spec.entity_col, lookback)
        )
    if blocks:
        out: pd.DataFrame = pd.concat(blocks, axis=1)
    else:
        out = pd.DataFrame(index=panel.index)

    # missing_lag_count: for each row, how many of the L lags fell
    # before the entity's first observation.
    position_within_entity = (
        panel.groupby(spec.entity_col, sort=False).cumcount().to_numpy()
    )
    missing_lag_count = np.maximum(
        0, lookback - 1 - position_within_entity
    ).astype(np.int64)
    out["missing_lag_count"] = missing_lag_count
    return out
