"""Phase B12 raw-MTS reshape tests.

The protocol leaf `benchmarks.protocol.raw_mts.panel_to_tensor`
turns an F2-contract panel into a 3D
`(n_instances, n_channels, n_timesteps)` float32 tensor for the
TSC adapter family. Tests pin: the shape contract, the boundary
cases at `L_resolved` (exact / below / above), the determinism
property, the dropped-categoricals convention (D-B12.6), and the
all-categorical-rejection path.
"""

import numpy as np
import pandas as pd
import pytest
from benchmarks.config import DatasetSpec
from benchmarks.protocol.raw_mts import (
    RawMTSError,
    broadcast_per_instance_to_per_row,
    instance_labels,
    panel_to_tensor,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _spec(
    *,
    name: str = "ds_raw_mts",
    lookback: int = 4,
    real_cols: tuple[str, ...] = ("x1", "x2"),
    categorical_cols: tuple[str, ...] = (),
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
        citation="raw-mts test",
    )


def _equal_length_panel(
    n_entities: int = 3,
    n_periods: int = 6,
    *,
    real_cols: tuple[str, ...] = ("x1", "x2"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    for entity in range(n_entities):
        for t in range(n_periods):
            row: dict[str, object] = {
                "entity_id": f"e_{entity}",
                "period": t,
                "y": int(entity % 2),
            }
            for col in real_cols:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


# --- shape contract + L_resolved boundaries (R2-qa-C4) ----------------------


def test_panel_to_tensor_shape_contract() -> None:
    """Returns a 3D tensor of shape `(n_instances, n_channels,
    L_resolved)` plus a 1D mapping array of length `len(panel)`."""
    spec = _spec(lookback=4)
    panel = _equal_length_panel(n_entities=3, n_periods=6)
    X_3d, mapping = panel_to_tensor(panel, spec)  # noqa: N806
    # 3 entities, 2 channels, 4 timesteps.
    assert X_3d.shape == (3, 2, 4)
    assert mapping.shape == (len(panel),)


def test_panel_to_tensor_returns_float32() -> None:
    """Gemini-I2: the cached tensor is `np.float32` regardless of
    source panel dtype. Halves host RAM at scale."""
    spec = _spec()
    panel = _equal_length_panel()
    X_3d, _ = panel_to_tensor(panel, spec)  # noqa: N806
    assert X_3d.dtype == np.float32


def test_panel_to_tensor_boundary_rows_exact() -> None:
    """An entity with EXACTLY `L_resolved` rows is accepted; every
    row maps to a real instance index."""
    spec = _spec(lookback=6)
    panel = _equal_length_panel(n_entities=2, n_periods=6)
    X_3d, mapping = panel_to_tensor(panel, spec)  # noqa: N806
    assert X_3d.shape == (2, 2, 6)
    # No row is below-floor.
    assert (mapping >= 0).all()


def test_panel_to_tensor_boundary_rows_below() -> None:
    """An entity with `L_resolved - 1` rows is excluded; its rows
    map to -1 so the adapter broadcasts NaN. The tensor's instance
    count drops by one."""
    spec = _spec(lookback=6)
    # Build a panel where entity 0 has only 5 rows (below floor)
    # but entity 1 has 6 rows (exactly at floor).
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for t in range(5):  # entity 0
        rows.append(
            {
                "entity_id": "e_0",
                "period": t,
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "y": 0,
            }
        )
    for t in range(6):  # entity 1
        rows.append(
            {
                "entity_id": "e_1",
                "period": t,
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "y": 1,
            }
        )
    panel = pd.DataFrame(rows)
    X_3d, mapping = panel_to_tensor(panel, spec)  # noqa: N806
    # Only one entity survives.
    assert X_3d.shape == (1, 2, 6)
    # Entity 0's 5 rows are below-floor.
    assert (mapping[:5] == -1).all()
    # Entity 1's 6 rows map to instance 0.
    assert (mapping[5:] == 0).all()


def test_panel_to_tensor_boundary_rows_above() -> None:
    """An entity with `L_resolved + 1` rows is accepted; the
    trailing-window clip keeps the last `L_resolved` rows."""
    spec = _spec(lookback=4)
    rows: list[dict[str, object]] = []
    # entity 0 has 6 rows (above floor); the last 4 should be kept.
    for t in range(6):
        rows.append(
            {
                "entity_id": "e_0",
                "period": t,
                "x1": float(t * 10),
                "x2": float(t * 100),
                "y": 0,
            }
        )
    panel = pd.DataFrame(rows)
    X_3d, mapping = panel_to_tensor(panel, spec)  # noqa: N806
    assert X_3d.shape == (1, 2, 4)
    # The kept window is periods 2..5; channel 0 trailing window
    # should be (2*10, 3*10, 4*10, 5*10) = (20, 30, 40, 50).
    np.testing.assert_array_equal(X_3d[0, 0], np.array([20, 30, 40, 50], dtype=np.float32))
    # Every panel row maps to instance 0 (the broadcast layer
    # decides whether the prediction reaches the trailing-removed
    # rows at predict time).
    assert (mapping == 0).all()


def test_panel_to_tensor_all_below_floor_raises() -> None:
    """If every entity is below `L_resolved`, raise `RawMTSError`."""
    spec = _spec(lookback=100)
    panel = _equal_length_panel(n_entities=3, n_periods=6)
    with pytest.raises(RawMTSError, match="below the lookback floor"):
        panel_to_tensor(panel, spec)


# --- categorical handling (Gemini-C3 / D-B12.6) ------------------------------


def test_raw_mts_drops_categorical_channels() -> None:
    """A spec declaring `feature_categorical_cols` MUST NOT have
    them in the returned tensor's channel dim (D-B12.6)."""
    spec = _spec(
        real_cols=("x1",),
        categorical_cols=("cat1",),
    )
    panel = _equal_length_panel(n_entities=3, n_periods=4, real_cols=("x1",))
    panel["cat1"] = ["a", "b"] * (len(panel) // 2)
    X_3d, _ = panel_to_tensor(panel, spec)  # noqa: N806
    # 1 channel (x1), not 2 (would be 2 if cat1 were ordinal-encoded).
    assert X_3d.shape[1] == 1


def test_raw_mts_all_categorical_panel_raises() -> None:
    """A spec with zero real channels (only categoricals) raises
    `RawMTSError` at config validation time."""
    spec = _spec(real_cols=(), categorical_cols=("cat1",))
    panel = _equal_length_panel(n_entities=3, n_periods=4)
    panel["cat1"] = "a"
    with pytest.raises(RawMTSError, match="zero real-valued"):
        panel_to_tensor(panel, spec)


# --- determinism (R2-qa-I1 family) ------------------------------------------


def test_panel_to_tensor_is_deterministic() -> None:
    """Two calls on the same immutable input produce bit-identical
    `(X_3d, mapping)`. Pins B9 reproducibility."""
    spec = _spec()
    panel = _equal_length_panel()
    out1 = panel_to_tensor(panel, spec)
    out2 = panel_to_tensor(panel, spec)
    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])


# --- Hypothesis properties (R1-qa-I3) ---------------------------------------


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_entities=st.integers(min_value=1, max_value=5),
    n_periods=st.integers(min_value=4, max_value=10),
    lookback=st.integers(min_value=2, max_value=4),
)
def test_panel_to_tensor_property_equal_length_happy_path(
    n_entities: int, n_periods: int, lookback: int
) -> None:
    """Every panel satisfying F2 + per-entity time-monotone period
    + at least `L_resolved` rows per entity produces a tensor of
    the declared shape."""
    spec = _spec(lookback=lookback)
    panel = _equal_length_panel(n_entities=n_entities, n_periods=n_periods)
    X_3d, mapping = panel_to_tensor(panel, spec)  # noqa: N806
    assert X_3d.shape == (n_entities, 2, lookback)
    assert mapping.shape == (len(panel),)
    assert (mapping >= 0).all()


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    short_n_rows=st.integers(min_value=1, max_value=3),
    lookback=st.integers(min_value=4, max_value=6),
)
def test_panel_to_tensor_property_short_entity_excluded(short_n_rows: int, lookback: int) -> None:
    """A panel with at least one entity carrying fewer than
    `L_resolved` rows excludes that entity from the tensor; the
    mapping for those rows is -1 so the broadcast layer emits
    NaN."""
    spec = _spec(lookback=lookback)
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    # Short entity (below floor).
    for t in range(short_n_rows):
        rows.append(
            {
                "entity_id": "short",
                "period": t,
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "y": 0,
            }
        )
    # Long entity (above floor, will be clipped).
    for t in range(lookback + 2):
        rows.append(
            {
                "entity_id": "long",
                "period": t,
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "y": 1,
            }
        )
    panel = pd.DataFrame(rows)
    X_3d, mapping = panel_to_tensor(panel, spec)  # noqa: N806
    # Only the long entity survives.
    assert X_3d.shape[0] == 1
    # Short entity rows map to -1.
    assert (mapping[:short_n_rows] == -1).all()


# --- instance_labels (last-period projection) -------------------------------


def test_instance_labels_takes_last_period_label() -> None:
    """The per-instance label is the LAST-period row's label (the
    trailing-window convention)."""
    spec = _spec(lookback=3)
    rows: list[dict[str, object]] = []
    # Entity 0: 4 rows with labels [0, 0, 0, 1] -> last is 1.
    for t, label in enumerate([0, 0, 0, 1]):
        rows.append(
            {
                "entity_id": "e_0",
                "period": t,
                "x1": float(t),
                "x2": float(t),
                "y": label,
            }
        )
    # Entity 1: 3 rows with labels [1, 1, 0] -> last is 0.
    for t, label in enumerate([1, 1, 0]):
        rows.append(
            {
                "entity_id": "e_1",
                "period": t,
                "x1": float(t),
                "x2": float(t),
                "y": label,
            }
        )
    panel = pd.DataFrame(rows)
    y = panel["y"].to_numpy()
    _, mapping = panel_to_tensor(panel, spec)
    labels = instance_labels(y, mapping)
    assert labels.tolist() == [1, 0]


def test_instance_labels_raises_on_length_mismatch() -> None:
    spec = _spec()
    panel = _equal_length_panel()
    _, mapping = panel_to_tensor(panel, spec)
    short_y = np.array([0, 1, 0])
    with pytest.raises(RawMTSError, match="does not match"):
        instance_labels(short_y, mapping)


def test_instance_labels_raises_on_all_below_floor() -> None:
    mapping = np.array([-1, -1, -1], dtype=np.int64)
    y = np.array([0, 1, 0])
    with pytest.raises(RawMTSError, match="every panel row is below-floor"):
        instance_labels(y, mapping)


# --- broadcast_per_instance_to_per_row --------------------------------------


def test_broadcast_per_instance_proba_emits_nan_for_below_floor() -> None:
    """Below-floor rows (mapping == -1) receive NaN per the A9.1
    strip convention so the B5 driver's
    `_strip_below_floor_rows` counts them uniformly across
    families."""
    per_instance = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
    mapping = np.array([0, 0, -1, 1, 1], dtype=np.int64)
    out = broadcast_per_instance_to_per_row(per_instance, mapping)
    assert out.shape == (5, 2)
    # Rows 0+1 carry instance 0's proba.
    np.testing.assert_array_equal(out[0], np.array([0.7, 0.3]))
    np.testing.assert_array_equal(out[1], np.array([0.7, 0.3]))
    # Row 2 is below-floor -> NaN.
    assert np.isnan(out[2]).all()
    # Rows 3+4 carry instance 1's proba.
    np.testing.assert_array_equal(out[3], np.array([0.2, 0.8]))
    np.testing.assert_array_equal(out[4], np.array([0.2, 0.8]))


def test_broadcast_per_instance_point_prediction_emits_nan_for_below_floor() -> None:
    per_instance = np.array([1.5, 2.5], dtype=np.float64)
    mapping = np.array([0, -1, 1], dtype=np.int64)
    out = broadcast_per_instance_to_per_row(per_instance, mapping)
    assert out.shape == (3,)
    assert out[0] == 1.5
    assert np.isnan(out[1])
    assert out[2] == 2.5


def test_broadcast_per_instance_invalid_ndim_raises() -> None:
    bad = np.zeros((2, 2, 2), dtype=np.float64)
    mapping = np.array([0, 1], dtype=np.int64)
    with pytest.raises(RawMTSError, match="ndim must be"):
        broadcast_per_instance_to_per_row(bad, mapping)


# --- broadcast preserves entity-homogeneity (Gemini-correctness) -------------


def test_broadcast_is_entity_homogeneous() -> None:
    """The R-B12-2 oracle: every row belonging to the same instance
    carries an IDENTICAL probability vector. A bug that permuted
    predictions across instances would still satisfy shape +
    sum-to-1 but violate this."""
    per_instance = np.array([[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]], dtype=np.float64)
    mapping = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    out = broadcast_per_instance_to_per_row(per_instance, mapping)
    # Within each instance, all rows must be identical.
    for instance_idx in range(3):
        rows = out[mapping == instance_idx]
        for row in rows[1:]:
            np.testing.assert_array_equal(row, rows[0])
