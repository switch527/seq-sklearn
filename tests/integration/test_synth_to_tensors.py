"""End-to-end: synthetic panel through TabularToSequence to tensors.

Locks the cross-family input contract: a DGP-generated panel fits and
transforms into the documented batched tensor dict, including the
variable-history path where one entity has a single row and another the
full lookback in the same call.
"""

import numpy as np
import pytest
import torch

from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.data.splits import compute_three_way_split
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.data.tabular_to_sequence import TabularToSequence


def _config(gen: SyntheticPanelGenerator) -> TabularToSequenceConfig:
    return TabularToSequenceConfig(
        id_col=gen.id_col,
        time_col=gen.time_col,
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback,
        max_categorical_cardinality=10_000,
    )


@pytest.mark.integration
@pytest.mark.parametrize("target_kind", ["binary", "multiclass", "regression_point"])
def test_synth_panel_to_tensor_dict_shapes(target_kind: str) -> None:
    gen = SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=80,
        periods_per_entity=(1, 60),
        seed=42,
    )
    panel, y = gen.generate()
    task = "binary" if target_kind == "binary" else target_kind
    tts = TabularToSequence(_config(gen), task)  # type: ignore[arg-type]
    out = tts.fit_transform(panel, y)

    per_entity_counts = (
        panel.sort_values([gen.id_col, gen.time_col])
        .groupby(gen.id_col, sort=True)
        .size()
        .to_numpy()
    )
    total = int(per_entity_counts.sum())
    lookback = gen.lookback
    assert out["static_categorical"].shape == (
        total,
        len(gen.static_categorical_cols),
    )
    assert out["static_real"].shape == (total, len(gen.static_real_cols))
    assert out["time_varying_real"].shape == (
        total,
        lookback,
        len(gen.time_varying_real_cols),
    )
    assert out["time_varying_categorical"].shape == (
        total,
        lookback,
        len(gen.time_varying_categorical_cols),
    )
    assert out["padding_mask"].shape == (total, lookback)
    assert out["padding_mask"].dtype == torch.bool
    assert out["target"].shape == (total,)
    assert out["entity_id"].shape == (total,)

    expected_target_dtype = torch.float32 if target_kind == "regression_point" else torch.long
    assert out["target"].dtype == expected_target_dtype


@pytest.mark.integration
def test_variable_history_one_and_full_lookback_in_one_call() -> None:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=200,
        periods_per_entity=(1, 60),
        lookback=12,
        seed=137,
    )
    panel, y = gen.generate()
    per_entity = panel.groupby(gen.id_col).size()
    assert per_entity.min() == 1
    assert per_entity.max() >= gen.lookback

    tts = TabularToSequence(_config(gen), "binary")
    out = tts.fit_transform(panel, y)
    mask = out["padding_mask"].numpy()

    # Every entity's first window (window-end 0) is left-padded on
    # lookback - 1 positions; any window-end >= lookback - 1 is full, so
    # a multi-row entity contributes a zero-padding window too.
    pad_counts = mask.sum(axis=1)
    assert pad_counts.max() == gen.lookback - 1
    assert pad_counts.min() == 0


@pytest.mark.integration
def test_inverse_transform_recovers_feature_frame() -> None:
    gen = SyntheticPanelGenerator(
        num_entities=30,
        periods_per_entity=20,
        seed=9999,
    )
    panel, y = gen.generate()
    tts = TabularToSequence(_config(gen), "binary").fit(panel, y)
    recovered = tts.inverse_transform(tts.transform(panel))
    for col in gen.time_varying_real_cols:
        assert col in recovered.columns
    for col in gen.static_real_cols:
        assert col in recovered.columns
    # Round-trip the scaled real columns within tolerance. With one
    # sliding window per window-end position, the window-end timestep
    # ranges over every row of every entity, so the recovered multiset
    # equals the full panel's tv-real column (not just the per-entity
    # tail). Compare multisets since the emitted entity_id is a
    # contiguous code, not the raw id.
    for col in gen.time_varying_real_cols:
        np.testing.assert_allclose(
            np.sort(recovered[col].to_numpy()),
            np.sort(panel[col].to_numpy()),
            atol=1e-5,
        )


@pytest.mark.integration
def test_three_way_split_against_real_transform_output() -> None:
    gen = SyntheticPanelGenerator(
        num_entities=60,
        periods_per_entity=(2, 40),
        lookback=12,
        seed=42,
    )
    panel, y = gen.generate()
    tts = TabularToSequence(_config(gen), "binary").fit(panel, y)
    out = tts.transform(panel)

    # Reconstruct entity_ids and window_time_index exactly as the A5
    # split note prescribes: windows for one entity are contiguous and
    # time-ascending in the batch, so per_entity_window_counts is the
    # run-length of each contiguous entity-code block and
    # window_time_index is np.concatenate([np.arange(n_i) ...]).
    entity_ids = out["entity_id"].numpy()
    uniq_codes, first_idx, counts = np.unique(entity_ids, return_index=True, return_counts=True)
    # Order the per-entity counts by first appearance in the batch.
    order = np.argsort(first_idx)
    codes_in_order = uniq_codes[order]
    per_entity_window_counts = counts[order]
    assert int(per_entity_window_counts.sum()) == len(entity_ids)
    window_time_index = np.concatenate([np.arange(n_i) for n_i in per_entity_window_counts])

    train, val, cal = compute_three_way_split(
        entity_ids,
        window_time_index,
        val_fraction=0.2,
        cal_fraction=0.2,
        val_split_strategy="time_ordered",
        calibration_set_provided=False,
    )

    assert cal.size > 0

    def tuples(idx: np.ndarray) -> set[tuple[int, int]]:
        return {(int(entity_ids[i]), int(window_time_index[i])) for i in idx}

    t, v, c = tuples(train), tuples(val), tuples(cal)
    assert t & v == set()
    assert t & c == set()
    assert v & c == set()
    assert len(t | v | c) == len(entity_ids)

    # Calibration holds the last cal_fraction windows per multi-window
    # entity (rounded per entity by window count).
    cal_by_entity: dict[int, list[int]] = {}
    for i in cal:
        cal_by_entity.setdefault(int(entity_ids[i]), []).append(int(window_time_index[i]))
    checked_multi = False
    for code, n_i in zip(codes_in_order, per_entity_window_counts, strict=True):
        n_cal = round(0.2 * n_i)
        if n_i > 1 and n_cal > 0:
            checked_multi = True
            expected_tail = sorted(range(n_i - n_cal, n_i))
            assert sorted(cal_by_entity.get(int(code), [])) == expected_tail
    assert checked_multi
