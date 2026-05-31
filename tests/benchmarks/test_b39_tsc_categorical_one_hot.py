"""Phase B39 / D-B12.6 closure tests.

Covers the optional one-hot encoding path for
`panel_to_tensor` (`categorical_categories` parameter) and
the `compute_categorical_categories` helper. The pre-B39
drop-categoricals behavior is preserved when the reference
is omitted, pinned by existing `test_raw_mts_*` tests; this
file only covers the new opt-in path.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
from benchmarks.config import DatasetSpec
from benchmarks.protocol.raw_mts import (
    RawMTSError,
    compute_categorical_categories,
    panel_to_tensor,
)


def _spec(
    *,
    name: str = "ds_b39",
    lookback: int = 4,
    real_cols: tuple[str, ...] = ("x1",),
    categorical_cols: tuple[str, ...] = ("c1",),
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/x.csv",
        integrity_sha256="0" * 64,
        archive_basename="x.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=real_cols,
        feature_categorical_cols=categorical_cols,
        lookback=lookback,
        positive_label=1,
        citation="b39 test",
    )


def _panel(
    *,
    n_entities: int = 2,
    n_periods: int = 4,
    real_cols: tuple[str, ...] = ("x1",),
    cat_cols: tuple[str, ...] = ("c1",),
    cat_values: Mapping[str, list[Any]] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for entity in range(n_entities):
        for t in range(n_periods):
            row: dict[str, object] = {
                "entity_id": f"e_{entity}",
                "period": t,
                "y": int(entity % 2),
            }
            for col in real_cols:
                row[col] = float(rng.normal())
            for col in cat_cols:
                row[col] = "default"
            rows.append(row)
    panel = pd.DataFrame(rows)
    if cat_values is not None:
        for col, vals in cat_values.items():
            assert len(vals) == len(panel), (
                f"cat_values[{col!r}] length {len(vals)} != panel length {len(panel)}"
            )
            panel[col] = vals
    return panel


# =============================================================================
# B39.4.1 compute_categorical_categories helper
# =============================================================================


def test_compute_categorical_categories_returns_sorted_unique_per_col() -> None:
    spec = _spec(categorical_cols=("c1",))
    panel = _panel(cat_values={"c1": ["b", "a", "b", "a", "c", "a", "b", "c"]})
    refs = compute_categorical_categories(panel, spec)
    assert refs == {"c1": ("a", "b", "c")}


def test_compute_categorical_categories_drops_nan_values() -> None:
    spec = _spec(categorical_cols=("c1",))
    panel = _panel(cat_values={"c1": ["a", np.nan, "b", "a", np.nan, "a", "b", "a"]})
    refs = compute_categorical_categories(panel, spec)
    assert refs == {"c1": ("a", "b")}


def test_compute_categorical_categories_empty_when_no_cat_cols() -> None:
    spec = _spec(categorical_cols=())
    panel = _panel(cat_cols=())
    assert compute_categorical_categories(panel, spec) == {}


# =============================================================================
# B39.4.2 panel_to_tensor one-hot path
# =============================================================================


def test_raw_mts_one_hot_encodes_categoricals() -> None:
    """1 real col + 1 cat col with 2 categories → channel dim 3.
    Per-row, the cat channels match the one-hot of the value."""
    spec = _spec(real_cols=("x1",), categorical_cols=("c1",))
    cat_values = ["a", "b", "a", "b", "b", "a", "a", "b"]  # 2 entities * 4 periods
    panel = _panel(n_entities=2, n_periods=4, cat_values={"c1": cat_values})
    refs = {"c1": ("a", "b")}
    X_3d, _ = panel_to_tensor(panel, spec, categorical_categories=refs)  # noqa: N806
    assert X_3d.shape == (2, 3, 4)  # 2 entities, 1 real + 2 cat, 4 timesteps
    # Entity 0 cat values (after trailing-L window with L=4, n_periods=4 → all rows kept):
    # ["a", "b", "a", "b"] → channel 1 (ref "a") = [1, 0, 1, 0]; channel 2 (ref "b") = [0, 1, 0, 1]
    np.testing.assert_array_equal(X_3d[0, 1], np.array([1, 0, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(X_3d[0, 2], np.array([0, 1, 0, 1], dtype=np.float32))
    # Entity 1 cat values: ["b", "a", "a", "b"] → ref "a" = [0,1,1,0]; ref "b" = [1,0,0,1]
    np.testing.assert_array_equal(X_3d[1, 1], np.array([0, 1, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(X_3d[1, 2], np.array([1, 0, 0, 1], dtype=np.float32))


def test_raw_mts_unseen_category_maps_to_all_zero() -> None:
    """Reference excludes a value present in the panel → that row's
    cat channels are all zero (handle_unknown='ignore' convention)."""
    spec = _spec(real_cols=("x1",), categorical_cols=("c1",))
    cat_values = ["a", "b", "z", "a", "z", "a", "b", "a"]
    panel = _panel(n_entities=2, n_periods=4, cat_values={"c1": cat_values})
    refs = {"c1": ("a", "b")}  # NB: "z" not in reference
    X_3d, _ = panel_to_tensor(panel, spec, categorical_categories=refs)  # noqa: N806
    assert X_3d.shape == (2, 3, 4)
    # Entity 0 cat values: ["a", "b", "z", "a"] → ref "a" = [1,0,0,1], ref "b" = [0,1,0,0]
    np.testing.assert_array_equal(X_3d[0, 1], np.array([1, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(X_3d[0, 2], np.array([0, 1, 0, 0], dtype=np.float32))


def test_raw_mts_multiple_categorical_columns() -> None:
    """2 cat cols with sizes 2 and 3 → channel dim = real + 5."""
    spec = _spec(real_cols=("x1",), categorical_cols=("c1", "c2"))
    panel = _panel(
        n_entities=2,
        n_periods=4,
        cat_cols=("c1", "c2"),
        cat_values={
            "c1": ["a", "b", "a", "b"] * 2,
            "c2": ["x", "y", "z", "x"] * 2,
        },
    )
    refs = {"c1": ("a", "b"), "c2": ("x", "y", "z")}
    X_3d, _ = panel_to_tensor(panel, spec, categorical_categories=refs)  # noqa: N806
    assert X_3d.shape == (2, 1 + 2 + 3, 4)


def test_raw_mts_categorical_categories_none_preserves_drop_behavior() -> None:
    """Default (None) keeps v1 drop-categoricals behavior; channels
    == real-only count."""
    spec = _spec(real_cols=("x1", "x2"), categorical_cols=("c1",))
    panel = _panel(
        n_entities=2,
        n_periods=4,
        real_cols=("x1", "x2"),
        cat_values={"c1": ["a", "b", "a", "b"] * 2},
    )
    X_3d, _ = panel_to_tensor(panel, spec)  # noqa: N806
    assert X_3d.shape[1] == 2  # only the 2 real cols


def test_raw_mts_missing_categorical_reference_raises() -> None:
    """Missing a declared cat_col in the reference dict raises
    RawMTSError. Silently dropping it would change channel count."""
    spec = _spec(real_cols=("x1",), categorical_cols=("c1", "c2"))
    panel = _panel(
        n_entities=2,
        n_periods=4,
        cat_cols=("c1", "c2"),
        cat_values={
            "c1": ["a"] * 8,
            "c2": ["x"] * 8,
        },
    )
    # Reference omits "c2".
    with pytest.raises(RawMTSError, match=r"does not cover"):
        panel_to_tensor(panel, spec, categorical_categories={"c1": ("a",)})
