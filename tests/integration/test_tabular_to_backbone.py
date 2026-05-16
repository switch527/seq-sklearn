"""End-to-end: synthetic panel through TabularToSequence to TFTBackbone.

Locks the Phase 3 boundary: the Phase 2 transform output dict feeds the
TFTBackbone forward pass and yields a TransformerBackboneOutput with the
documented shapes. No training, no estimator.
"""

import pytest
import torch

from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models._backbone import BackboneOutput
from seq_sklearn.models.transformer._backbone import TransformerBackboneOutput
from seq_sklearn.models.transformer.tft.backbone import TFTBackbone


def _tabular_config(gen: SyntheticPanelGenerator) -> TabularToSequenceConfig:
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
@pytest.mark.parametrize("readout", ["last_valid", "mean_pool"])
def test_synth_panel_through_tft_backbone_shapes(readout: str) -> None:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=40,
        periods_per_entity=(1, 30),
        prediction_step=0,
        lookback=10,
        seed=42,
    )
    panel, y = gen.generate()
    tabular = _tabular_config(gen)
    tts = TabularToSequence(tabular, "binary").fit(panel, y)
    batch = tts.transform(panel)

    cfg = TFTConfig(
        task_type="binary",
        loss=LossConfig(strategy="cross_entropy"),
        tabular_config=tabular,
        hidden_size=16,
        attention_heads=4,
        prediction_readout=readout,  # type: ignore[arg-type]
    )
    backbone = TFTBackbone(
        cfg,
        static_cat_cardinalities=[tts.cardinalities_[c] for c in gen.static_categorical_cols],
        tv_cat_cardinalities=[tts.cardinalities_[c] for c in gen.time_varying_categorical_cols],
        n_static_real=len(gen.static_real_cols),
        n_tv_real=len(gen.time_varying_real_cols),
    )
    backbone.eval()
    with torch.no_grad():
        out = backbone.forward(batch)

    total = batch["entity_id"].shape[0]
    lookback = gen.lookback
    n_tv_vars = len(gen.time_varying_categorical_cols) + len(gen.time_varying_real_cols)
    n_static_vars = len(gen.static_categorical_cols) + len(gen.static_real_cols)

    assert isinstance(out, TransformerBackboneOutput)
    assert isinstance(out, BackboneOutput)
    assert out.representation.shape == (total, cfg.hidden_size)
    assert out.padding_mask.shape == (total, lookback)
    assert out.var_selection_weights.shape == (total, lookback, n_tv_vars)
    assert out.attention_weights.shape == (
        total,
        cfg.attention_heads,
        lookback,
        lookback,
    )
    assert out.static_var_selection_weights.shape == (total, n_static_vars)
    assert not torch.isnan(out.representation).any()

    metrics = backbone.compute_training_metrics(out)
    assert set(metrics) == {"train.var_selection_entropy", "train.attention_entropy"}
    attn_entropy = metrics["train.attention_entropy"]
    assert isinstance(attn_entropy, dict)
    assert len(attn_entropy["entropy_per_head"]) == cfg.attention_heads
