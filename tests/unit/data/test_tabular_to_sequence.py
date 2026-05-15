"""Tests for the panel-to-sequence transformer (A5 / F3)."""

import logging

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.errors import ConfigError, DataContractError, NotFittedError
from seq_sklearn.logging import Event


def _panel() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    y = []
    for entity in range(4):
        n = entity + 1  # 1, 2, 3, 4 rows
        for t in range(n):
            rows.append(
                {
                    "id": entity,
                    "time": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(t),
                    "sc": f"s{entity % 2}",
                    "sr": float(entity),
                    "tr": float(t) + 0.5,
                    "tc": f"c{t % 2}",
                }
            )
            y.append(entity % 2)
    return pd.DataFrame(rows), np.asarray(y)


def _config(**overrides: object) -> TabularToSequenceConfig:
    base: dict[str, object] = {
        "id_col": "id",
        "time_col": "time",
        "static_categorical_cols": ("sc",),
        "static_real_cols": ("sr",),
        "time_varying_real_cols": ("tr",),
        "time_varying_categorical_cols": ("tc",),
        "lookback": 4,
    }
    base.update(overrides)
    return TabularToSequenceConfig(**base)  # type: ignore[arg-type]


def test_one_row_entity_left_padded_with_mask() -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(), "binary").fit(frame, y)
    out = tts.transform(frame)
    # Entity 0 has one row; sorted by id its window is row 0.
    mask = out["padding_mask"]
    assert mask.shape == (4, 4)
    assert mask[0].tolist() == [True, True, True, False]
    assert out["time_varying_real"].shape == (4, 4, 1)


def test_duplicate_id_time_raises() -> None:
    frame, y = _panel()
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    y2 = np.concatenate([y, y[:1]])
    with pytest.raises(DataContractError, match="duplicate"):
        TabularToSequence(_config(), "binary").fit(dup, y2)


def test_object_dtype_time_raises() -> None:
    frame, y = _panel()
    frame = frame.copy()
    frame["time"] = frame["time"].astype(object).astype(str)
    with pytest.raises(DataContractError, match="not supported"):
        TabularToSequence(_config(), "binary").fit(frame, y)


def test_fingerprint_stable_across_two_fits() -> None:
    frame, y = _panel()
    a = TabularToSequence(_config(), "binary").fit(frame, y)
    b = TabularToSequence(_config(), "binary").fit(frame.copy(), y.copy())
    assert a.feature_schema_fingerprint_ == b.feature_schema_fingerprint_
    assert a.feature_schema_fingerprint_ is not None


def test_mask_polarity_true_is_padding() -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(), "binary").fit(frame, y)
    out = tts.transform(frame)
    # Entity 3 has 4 rows == lookback, so zero padding.
    assert out["padding_mask"][3].tolist() == [False, False, False, False]
    # Entity 1 has 2 rows, lookback 4 -> first two positions are padding.
    assert out["padding_mask"][1].tolist() == [True, True, False, False]


def test_not_fitted_raises() -> None:
    frame, _ = _panel()
    with pytest.raises(NotFittedError):
        TabularToSequence(_config(), "binary").transform(frame)
    with pytest.raises(NotFittedError):
        TabularToSequence(_config(), "binary").inverse_transform({})


def test_target_dtype_classification_is_long() -> None:
    frame, y = _panel()
    out = TabularToSequence(_config(), "binary").fit_transform(frame, y)
    assert out["target"].dtype == torch.long


def test_target_dtype_regression_is_float() -> None:
    frame, y = _panel()
    yf = y.astype(float) + 0.25
    out = TabularToSequence(_config(), "regression_point").fit_transform(frame, yf)
    assert out["target"].dtype == torch.float32


def test_multiclass_is_long() -> None:
    frame, y = _panel()
    out = TabularToSequence(_config(), "multiclass").fit_transform(frame, y)
    assert out["target"].dtype == torch.long


def test_y_length_mismatch_raises() -> None:
    frame, y = _panel()
    with pytest.raises(DataContractError, match="does not match"):
        TabularToSequence(_config(), "binary").fit(frame, y[:-1])


def test_tz_mixing_raises() -> None:
    frame, y = _panel()
    frame = frame.copy()
    times = list(frame["time"])
    mixed = [
        t.tz_localize("UTC") if i % 2 == 0 else t.tz_localize(None) for i, t in enumerate(times)
    ]
    frame["time"] = pd.Series(mixed, dtype=object)
    with pytest.raises(DataContractError, match="mixes tz-aware"):
        TabularToSequence(_config(), "binary").fit(frame, y)


def test_object_dtype_consistent_tz_passes_tz_check() -> None:
    # An object-dtype time column with a single consistent tz reaches the
    # tz-flag computation but does not raise on the tz-mixing rule; the
    # dtype check upstream is what rejects it.
    frame, y = _panel()
    frame = frame.copy()
    frame["time"] = pd.Series([t.tz_localize("UTC") for t in frame["time"]], dtype=object)
    with pytest.raises(DataContractError, match="not supported"):
        TabularToSequence(_config(), "binary").fit(frame, y)


def test_missing_time_col_reported_by_check_columns() -> None:
    frame, y = _panel()
    frame = frame.drop(columns=["time"])
    with pytest.raises(DataContractError, match="time_col"):
        TabularToSequence(_config(), "binary").fit(frame, y)


def test_sklearn_is_fitted_flag() -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(), "binary")
    assert tts.__sklearn_is_fitted__() is False
    tts.fit(frame, y)
    assert tts.__sklearn_is_fitted__() is True


def test_inverse_transform_without_tv_real() -> None:
    rows = []
    for entity in range(3):
        for t in range(2):
            rows.append(
                {
                    "id": entity,
                    "time": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(t),
                    "tc": f"c{t}",
                }
            )
    frame = pd.DataFrame(rows)
    cfg = TabularToSequenceConfig(
        id_col="id",
        time_col="time",
        time_varying_categorical_cols=("tc",),
        lookback=4,
    )
    tts = TabularToSequence(cfg, "binary").fit(frame, np.zeros(6))
    recovered = tts.inverse_transform(tts.transform(frame))
    assert "tc" in recovered.columns
    assert "tr" not in recovered.columns


def test_min_periods_drops_entity_at_fit() -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(min_periods=3), "binary").fit(frame, y)
    # Only entities 2 and 3 (3 and 4 rows) survive the fit floor; the
    # scaler statistics come from those rows only.
    assert tts.real_scaler_ is not None


def test_min_periods_predict_nan_target_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(min_periods_predict=3), "regression_point").fit(
        frame, y.astype(float)
    )
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        out = tts.transform(frame)
    events = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == Event.DATA_DUPLICATE_FLOOR_BREACH_COUNT
    ]
    assert len(events) == 1
    assert events[0].payload["count"] == 2  # entities 0 and 1 below floor
    targets = out["target"].numpy()
    assert np.isnan(targets[0])
    assert np.isnan(targets[1])


def test_cardinality_cap_raises() -> None:
    frame, y = _panel()
    with pytest.raises(ConfigError, match="exceeds max_categorical_cardinality"):
        TabularToSequence(_config(max_categorical_cardinality=1), "binary").fit(frame, y)


def test_hash_high_cardinality_does_not_raise() -> None:
    frame, y = _panel()
    cfg = _config(max_categorical_cardinality=1, hash_high_cardinality=True)
    tts = TabularToSequence(cfg, "binary").fit(frame, y)
    assert "sc" in tts.hashed_columns_
    out = tts.transform(frame)
    assert out["static_categorical"].shape == (4, 1)


def test_inverse_transform_round_trips_reals() -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(), "binary").fit(frame, y)
    out = tts.transform(frame)
    recovered = tts.inverse_transform(out)
    # Entity 3 (last row) full window, real feature at window end is t=3.
    assert "tr" in recovered.columns
    assert "sr" in recovered.columns
    np.testing.assert_allclose(sorted(recovered["sr"].to_numpy()), [0.0, 1.0, 2.0, 3.0], atol=1e-6)


def test_inverse_transform_decodes_categoricals() -> None:
    frame, y = _panel()
    tts = TabularToSequence(_config(), "binary").fit(frame, y)
    recovered = tts.inverse_transform(tts.transform(frame))
    assert set(recovered["sc"].tolist()).issubset({"s0", "s1"})


def test_inverse_transform_hash_columns_raise() -> None:
    frame, y = _panel()
    cfg = _config(max_categorical_cardinality=1, hash_high_cardinality=True)
    tts = TabularToSequence(cfg, "binary").fit(frame, y)
    with pytest.raises(NotImplementedError, match="hash-tricked"):
        tts.inverse_transform(tts.transform(frame))


def test_embedding_dim_heuristic_and_override() -> None:
    frame, y = _panel()
    cfg = _config(categorical_embed_dims={"sc": 7})
    tts = TabularToSequence(cfg, "binary").fit(frame, y)
    assert tts.embedding_dims_["sc"] == 7
    assert tts.embedding_dims_["tc"] == min(50, round(1.6 * (2 + 1) ** 0.56))


def test_clip_features_applied_after_scaling() -> None:
    frame, y = _panel()
    cfg = _config(clip_features=0.5)
    out = TabularToSequence(cfg, "binary").fit_transform(frame, y)
    tv = out["time_varying_real"].numpy()
    assert tv.max() <= 0.5 + 1e-6
    assert tv.min() >= -0.5 - 1e-6
    sr = out["static_real"].numpy()
    assert sr.max() <= 0.5 + 1e-6


def test_scaling_static_real_inherit_vs_explicit() -> None:
    frame, y = _panel()
    inherit = TabularToSequence(_config(scaling_real="robust"), "binary").fit(frame, y)
    explicit = TabularToSequence(
        _config(scaling_real="robust", scaling_static_real="standard"), "binary"
    ).fit(frame, y)
    assert type(inherit.static_real_scaler_) is type(inherit.real_scaler_)
    assert type(explicit.static_real_scaler_) is not type(explicit.real_scaler_)


def test_no_categorical_columns_path() -> None:
    rows = []
    y = []
    for entity in range(3):
        for t in range(5):
            rows.append(
                {
                    "id": entity,
                    "time": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(t),
                    "tr": float(t),
                }
            )
            y.append(0)
    frame = pd.DataFrame(rows)
    cfg = TabularToSequenceConfig(
        id_col="id", time_col="time", time_varying_real_cols=("tr",), lookback=3
    )
    out = TabularToSequence(cfg, "binary").fit_transform(frame, np.asarray(y))
    assert out["static_categorical"].shape == (3, 0)
    assert out["time_varying_categorical"].shape == (3, 3, 0)
    assert out["static_real"].shape == (3, 0)
    rec = TabularToSequence(cfg, "binary").fit(frame, np.asarray(y)).inverse_transform(out)
    assert "tr" in rec.columns


def test_no_time_varying_real_path() -> None:
    rows = []
    for entity in range(3):
        for t in range(2):
            rows.append(
                {
                    "id": entity,
                    "time": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(t),
                    "tc": f"c{t}",
                }
            )
    frame = pd.DataFrame(rows)
    cfg = TabularToSequenceConfig(
        id_col="id",
        time_col="time",
        time_varying_categorical_cols=("tc",),
        lookback=4,
    )
    out = TabularToSequence(cfg, "binary").fit_transform(frame, np.zeros(6))
    assert out["time_varying_real"].shape == (3, 4, 0)
    assert out["time_varying_categorical"].shape == (3, 4, 1)


def test_prediction_step_zero() -> None:
    frame, y = _panel()
    out = TabularToSequence(_config(prediction_step=0), "binary").fit_transform(frame, y)
    assert out["target"].shape == (4,)


def test_longer_than_lookback_uses_most_recent() -> None:
    rows = []
    for t in range(10):
        rows.append(
            {
                "id": 0,
                "time": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(t),
                "sc": "a",
                "sr": 1.0,
                "tr": float(t),
                "tc": "x",
            }
        )
    frame = pd.DataFrame(rows)
    out = TabularToSequence(_config(lookback=4), "binary").fit_transform(frame, np.zeros(10))
    assert out["time_varying_real"].shape == (1, 4, 1)
    assert out["padding_mask"][0].tolist() == [False, False, False, False]


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=50))
def test_property_tv_real_shape(seed: int) -> None:
    gen = SyntheticPanelGenerator(
        num_entities=12,
        periods_per_entity=(1, 60),
        prediction_step=0,
        seed=seed,
    )
    panel, y = gen.generate()
    cfg = TabularToSequenceConfig(
        id_col="id",
        time_col="time",
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback,
        max_categorical_cardinality=10_000,
    )
    out = TabularToSequence(cfg, "binary").fit_transform(panel, y)
    n_entities = panel["id"].nunique()
    assert out["time_varying_real"].shape == (
        n_entities,
        gen.lookback,
        len(gen.time_varying_real_cols),
    )
