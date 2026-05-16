"""Tests for the TFTBackbone (A6 / A15 / N1)."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.errors import PredictionError
from seq_sklearn.models._backbone import BackboneOutput
from seq_sklearn.models.transformer._backbone import TransformerBackboneOutput
from seq_sklearn.models.transformer.tft.backbone import TFTBackbone


def _backbone(make_tft_config: Callable[..., TFTConfig], **cfg_overrides: object) -> TFTBackbone:
    cfg = make_tft_config(hidden_size=8, attention_heads=2, **cfg_overrides)
    return TFTBackbone(
        cfg,
        static_cat_cardinalities=[3, 4],
        tv_cat_cardinalities=[5],
        n_static_real=2,
        n_tv_real=2,
    )


def _batch(
    batch: int, seq_len: int, *, pad_first: int = 0, seed: int = 0
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    if pad_first:
        mask[:, :pad_first] = True
    return {
        "static_categorical": torch.randint(0, 3, (batch, 2), generator=gen),
        "static_real": torch.randn(batch, 2, generator=gen),
        "time_varying_real": torch.randn(batch, seq_len, 2, generator=gen),
        "time_varying_categorical": torch.randint(0, 5, (batch, seq_len, 1), generator=gen),
        "padding_mask": mask,
    }


def test_forward_emits_documented_shapes(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    model = _backbone(make_tft_config)
    model.eval()
    out = model.forward(_batch(3, 6, seed=1))
    assert isinstance(out, TransformerBackboneOutput)
    assert out.representation.shape == (3, 8)
    assert out.padding_mask.shape == (3, 6)
    assert out.var_selection_weights.shape == (3, 6, 3)  # n_tv_vars = 1 cat + 2 real
    assert out.attention_weights.shape == (3, 2, 6, 6)  # (B, H, L, L)
    assert out.static_var_selection_weights.shape == (3, 4)  # 2 cat + 2 real


def test_lstm_init_tuple_is_c_h_then_c_c(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    model = _backbone(make_tft_config)
    model.eval()
    # B >= 2 with distinct static features so c_h != c_c is detectable.
    captured: dict[str, Any] = {}
    real_forward = nn.LSTM.forward

    def spy(self: nn.LSTM, *args: Any, **kwargs: Any) -> Any:
        hx = args[1] if len(args) > 1 else kwargs.get("hx")
        captured["hx"] = hx
        return real_forward(self, *args, **kwargs)

    batch = _batch(2, 5, seed=2)
    batch["static_real"] = torch.tensor([[3.0, -2.0], [-5.0, 7.0]])
    batch["static_categorical"] = torch.tensor([[0, 1], [2, 3]])
    with torch.no_grad():
        torch.nn.modules.rnn.LSTM.forward = spy  # type: ignore[method-assign]
        try:
            static_vars = model._encode_static(batch)
            static_selected, _ = model.static_vsn(static_vars)
            expected_h = model.context_h(static_selected).unsqueeze(0)
            expected_c = model.context_c(static_selected).unsqueeze(0)
            model.forward(batch)
        finally:
            torch.nn.modules.rnn.LSTM.forward = real_forward  # type: ignore[method-assign]

    h_0, c_0 = captured["hx"]
    assert torch.allclose(h_0, expected_h, atol=1e-6)
    assert torch.allclose(c_0, expected_c, atol=1e-6)
    assert not torch.allclose(h_0, c_0)


def test_single_row_entity_is_non_nan(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    model = _backbone(make_tft_config)
    model.eval()
    out = model.forward(_batch(1, 1, seed=3))
    assert out.representation.shape == (1, 8)
    assert not torch.isnan(out.representation).any()


def test_zero_valid_timesteps_raises(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    model = _backbone(make_tft_config)
    model.eval()
    batch = _batch(2, 4, seed=4)
    batch["padding_mask"] = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(PredictionError, match="zero valid timesteps"):
        model.forward(batch)


def test_mean_pool_readout_mask_correctness(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    torch.use_deterministic_algorithms(True)
    try:
        model = _backbone(make_tft_config, prediction_readout="mean_pool")
        model.eval()
        unpadded = _batch(1, 4, seed=5)
        padded = _batch(1, 7, pad_first=3, seed=99)
        # Place the same valid rows in the padded version's tail.
        for key in (
            "time_varying_real",
            "time_varying_categorical",
        ):
            padded[key][:, 3:] = unpadded[key]
        padded["static_categorical"] = unpadded["static_categorical"]
        padded["static_real"] = unpadded["static_real"]
        with torch.no_grad():
            out_unpadded = model.forward(unpadded)
            out_padded = model.forward(padded)
        assert torch.allclose(out_padded.representation, out_unpadded.representation, atol=1e-5)
    finally:
        torch.use_deterministic_algorithms(False)


def test_last_valid_readout_mask_correctness(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    torch.use_deterministic_algorithms(True)
    try:
        model = _backbone(make_tft_config)  # last_valid is the config default
        model.eval()
        unpadded = _batch(1, 4, seed=6)
        padded = _batch(1, 7, pad_first=3, seed=42)
        for key in ("time_varying_real", "time_varying_categorical"):
            padded[key][:, 3:] = unpadded[key]
        padded["static_categorical"] = unpadded["static_categorical"]
        padded["static_real"] = unpadded["static_real"]
        with torch.no_grad():
            out_unpadded = model.forward(unpadded)
            out_padded = model.forward(padded)
        assert torch.allclose(out_padded.representation, out_unpadded.representation, atol=1e-5)
    finally:
        torch.use_deterministic_algorithms(False)


def test_compute_training_metrics_ignores_padded_positions(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    model = _backbone(make_tft_config)
    # L = 3, one padded position at index 0. Valid slice = positions 1, 2.
    mask = torch.tensor([[True, False, False]])
    n_vars = 4
    n_heads = 2
    # Uniform VSN rows at the padded position (max entropy), a sharp
    # distribution at the valid positions.
    sharp = torch.zeros(n_vars)
    sharp[0] = 1.0
    uniform = torch.full((n_vars,), 1.0 / n_vars)
    var_w = torch.stack([uniform, sharp, sharp]).unsqueeze(0)  # (1, 3, n_vars)

    # Zero attention rows at the padded query, sharp rows elsewhere.
    attn = torch.zeros(1, n_heads, 3, 3)
    attn[:, :, 1, 1] = 1.0
    attn[:, :, 2, 2] = 1.0

    static_w = torch.tensor([[0.25, 0.25, 0.25, 0.25]])  # max entropy over 4 vars

    out = TransformerBackboneOutput(
        representation=torch.zeros(1, 8),
        padding_mask=mask,
        var_selection_weights=var_w,
        attention_weights=attn,
        static_var_selection_weights=static_w,
    )
    metrics = model.compute_training_metrics(out)

    vse = metrics["train.var_selection_entropy"]
    assert isinstance(vse, dict)
    # Sharp distribution -> entropy ~ 0 on both valid positions.
    assert vse["temporal_entropy"] == pytest.approx(0.0, abs=1e-6)
    # Static is unmasked: uniform over 4 -> ln(4).
    assert vse["static_entropy"] == pytest.approx(torch.tensor(4.0).log().item(), abs=1e-6)

    ae = metrics["train.attention_entropy"]
    assert isinstance(ae, dict)
    # Sharp attention at the two valid queries -> per-head entropy 0.
    assert ae["entropy_per_head"] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_empty_feature_sides_use_synthetic_variable(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    cfg = make_tft_config(hidden_size=8, attention_heads=2)
    model = TFTBackbone(
        cfg,
        static_cat_cardinalities=[],
        tv_cat_cardinalities=[],
        n_static_real=0,
        n_tv_real=0,
    )
    model.eval()
    batch = {
        "static_categorical": torch.zeros(2, 0, dtype=torch.long),
        "static_real": torch.zeros(2, 0),
        "time_varying_real": torch.zeros(2, 5, 0),
        "time_varying_categorical": torch.zeros(2, 5, 0, dtype=torch.long),
        "padding_mask": torch.zeros(2, 5, dtype=torch.bool),
    }
    out = model.forward(batch)
    assert out.representation.shape == (2, 8)
    assert out.var_selection_weights.shape == (2, 5, 1)  # one synthetic tv var
    assert out.static_var_selection_weights.shape == (2, 1)  # one synthetic static var
    assert not torch.isnan(out.representation).any()


def test_compute_training_metrics_rejects_base_backbone_output(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    model = _backbone(make_tft_config)
    base_only = BackboneOutput(
        representation=torch.zeros(2, 8),
        padding_mask=torch.zeros(2, 4, dtype=torch.bool),
    )
    with pytest.raises(TypeError, match="TransformerBackboneOutput"):
        model.compute_training_metrics(base_only)
