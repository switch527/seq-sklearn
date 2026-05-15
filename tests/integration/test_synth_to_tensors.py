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
        prediction_step=0,
        seed=42,
    )
    panel, y = gen.generate()
    task = "binary" if target_kind == "binary" else target_kind
    tts = TabularToSequence(_config(gen), task)  # type: ignore[arg-type]
    out = tts.fit_transform(panel, y)

    n_entities = panel[gen.id_col].nunique()
    lookback = gen.lookback
    assert out["static_categorical"].shape == (
        n_entities,
        len(gen.static_categorical_cols),
    )
    assert out["static_real"].shape == (n_entities, len(gen.static_real_cols))
    assert out["time_varying_real"].shape == (
        n_entities,
        lookback,
        len(gen.time_varying_real_cols),
    )
    assert out["time_varying_categorical"].shape == (
        n_entities,
        lookback,
        len(gen.time_varying_categorical_cols),
    )
    assert out["padding_mask"].shape == (n_entities, lookback)
    assert out["padding_mask"].dtype == torch.bool
    assert out["target"].shape == (n_entities,)
    assert out["entity_id"].shape == (n_entities,)

    expected_target_dtype = torch.float32 if target_kind == "regression_point" else torch.long
    assert out["target"].dtype == expected_target_dtype


@pytest.mark.integration
def test_variable_history_one_and_full_lookback_in_one_call() -> None:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=200,
        periods_per_entity=(1, 60),
        prediction_step=0,
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

    # The single-row entity is padded on lookback-1 positions; a
    # full-history entity has no padding.
    pad_counts = mask.sum(axis=1)
    assert pad_counts.max() == gen.lookback - 1
    assert pad_counts.min() == 0


@pytest.mark.integration
def test_inverse_transform_recovers_feature_frame() -> None:
    gen = SyntheticPanelGenerator(
        num_entities=30,
        periods_per_entity=20,
        prediction_step=0,
        seed=9999,
    )
    panel, y = gen.generate()
    tts = TabularToSequence(_config(gen), "binary").fit(panel, y)
    recovered = tts.inverse_transform(tts.transform(panel))
    for col in gen.time_varying_real_cols:
        assert col in recovered.columns
    for col in gen.static_real_cols:
        assert col in recovered.columns
    # Round-trip the scaled real columns within tolerance. Compare the
    # multiset of window-end values rather than a positional join, since
    # the emitted entity_id is a contiguous code, not the raw id.
    ordered = panel.sort_values([gen.id_col, gen.time_col])
    last_rows = ordered.groupby(gen.id_col).tail(1)
    for col in gen.time_varying_real_cols:
        np.testing.assert_allclose(
            np.sort(recovered[col].to_numpy()),
            np.sort(last_rows[col].to_numpy()),
            atol=1e-5,
        )
