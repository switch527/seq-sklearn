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

    # Uniform (max-entropy) attention at the padded query, sharp rows
    # at the valid queries. A zero row would have entropy 0 too, making
    # the valid_h mask undetectable; a uniform padded row (entropy
    # ln(3)) is only excluded if valid_h actually masks it.
    attn = torch.zeros(1, n_heads, 3, 3)
    attn[:, :, 0, :] = 1.0 / 3.0
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


def test_readout_last_valid_selects_last_not_first(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    # Direct _readout test: the pipeline-level mask-correctness tests
    # copy data into the padded tail, so first-valid and last-valid
    # carry identical content and a first/last swap is invisible. Here
    # the first and last valid positions hold distinct sentinels, so
    # last_valid must return the LAST one.
    model = _backbone(make_tft_config)
    hidden = model.hidden_size
    seq_len = 4
    hidden_seq = torch.zeros(1, seq_len, hidden)
    hidden_seq[0, 1] = 1.0  # first valid position
    hidden_seq[0, 2] = 5.0
    hidden_seq[0, 3] = 9.0  # last valid position
    mask = torch.tensor([[True, False, False, False]])  # pos 0 padded
    out = model._readout(hidden_seq, mask)
    assert torch.equal(out, torch.full((1, hidden), 9.0))


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


@pytest.mark.parametrize("readout", ["last_valid", "mean_pool"])
def test_readout_mask_correctness_single_valid_step(
    make_tft_config: Callable[..., TFTConfig], readout: str
) -> None:
    # B=1, exactly one valid timestep: the last_valid argmax-on-flip
    # boundary (must resolve to the single valid index, not 0) and the
    # mean_pool divide-by-one edge.
    torch.use_deterministic_algorithms(True)
    try:
        model = _backbone(make_tft_config, prediction_readout=readout)
        model.eval()
        unpadded = _batch(1, 1, seed=71)
        padded = _batch(1, 5, pad_first=4, seed=72)
        for key in ("time_varying_real", "time_varying_categorical"):
            padded[key][:, 4:] = unpadded[key]
        padded["static_categorical"] = unpadded["static_categorical"]
        padded["static_real"] = unpadded["static_real"]
        with torch.no_grad():
            out_unpadded = model.forward(unpadded)
            out_padded = model.forward(padded)
        assert torch.allclose(out_padded.representation, out_unpadded.representation, atol=1e-5)
    finally:
        torch.use_deterministic_algorithms(False)


def test_lstm_mixed_per_row_valid_lengths(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    # The _run_lstm gather/scatter permutation is per row. A batch whose
    # rows have different valid lengths exercises divergent permutation
    # rows that a B=1 or uniform-pad batch cannot. Each row's padded
    # representation must equal its own single-row unpadded forward.
    torch.use_deterministic_algorithms(True)
    try:
        model = _backbone(make_tft_config)
        model.eval()
        seq_len = 6
        full = _batch(3, seq_len, seed=80)
        # Per-row leading pad counts: row 0 -> 1, row 1 -> 3, row 2 -> 0.
        pad_counts = [1, 3, 0]
        mask = torch.zeros(3, seq_len, dtype=torch.bool)
        for r, p in enumerate(pad_counts):
            mask[r, :p] = True
        full["padding_mask"] = mask
        with torch.no_grad():
            out_full = model.forward(full)
            for r, p in enumerate(pad_counts):
                row = {
                    "static_categorical": full["static_categorical"][r : r + 1],
                    "static_real": full["static_real"][r : r + 1],
                    "time_varying_real": full["time_varying_real"][r : r + 1, p:],
                    "time_varying_categorical": full["time_varying_categorical"][r : r + 1, p:],
                    "padding_mask": torch.zeros(1, seq_len - p, dtype=torch.bool),
                }
                out_row = model.forward(row)
                assert torch.allclose(
                    out_full.representation[r : r + 1],
                    out_row.representation,
                    atol=1e-5,
                )
                # var_selection_weights is produced directly by the
                # gather/scatter permutation; a bad inverse permutation
                # would corrupt it even if the readout coincidentally
                # matched. Compare the valid slice per row.
                assert torch.allclose(
                    out_full.var_selection_weights[r : r + 1, p:],
                    out_row.var_selection_weights,
                    atol=1e-5,
                )
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize(
    ("name", "index"),
    [("time_varying_real", (0, 1, 0)), ("static_real", (0, 0))],
)
def test_forward_rejects_nan_and_inf_inputs(
    make_tft_config: Callable[..., TFTConfig],
    name: str,
    index: tuple[int, ...],
) -> None:
    model = _backbone(make_tft_config)
    model.eval()
    for bad in (float("nan"), float("inf")):
        batch = _batch(2, 4, seed=81)
        batch[name][index] = bad
        with pytest.raises(PredictionError, match="NaN or inf"):
            model.forward(batch)


def test_forward_rejects_out_of_range_categorical_index(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    # A direct caller that did not route through TabularToSequence
    # could emit an index past the embedding table; that must be a
    # PredictionError, not an opaque embedding-kernel crash.
    model = _backbone(make_tft_config)
    model.eval()
    for name, oob_index in (
        ("static_categorical", (0, 0)),
        ("time_varying_categorical", (0, 1, 0)),
    ):
        batch = _batch(2, 4, seed=82)
        batch[name][oob_index] = 9999
        with pytest.raises(PredictionError, match="outside"):
            model.forward(batch)


def test_forward_rejects_non_contiguous_padding(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    # Interior padding violates the F3 left-pad contract the packed
    # LSTM gather/scatter relies on.
    model = _backbone(make_tft_config)
    model.eval()
    batch = _batch(1, 4, seed=83)
    batch["padding_mask"] = torch.tensor([[False, True, False, False]])
    with pytest.raises(PredictionError, match="contiguous leading block"):
        model.forward(batch)


def test_empty_side_synthetic_pad_params_receive_gradient(
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
    # eval() (not train()) so dropout does not stochastically zero the
    # already-small synthetic-path gradient. The synthetic vars init to
    # zeros, a degenerate fixed point (GRN(0) skip + size-1-LayerNorm
    # selection) where the gradient is structurally zero; perturb off it
    # so the test asserts the params carry a real training signal.
    model.eval()
    gen = torch.Generator().manual_seed(90)
    with torch.no_grad():
        model.static_pad_vec.copy_(torch.randn(8, generator=gen))
        model.tv_pad_vec.copy_(torch.randn(8, generator=gen))
    batch = {
        "static_categorical": torch.zeros(2, 0, dtype=torch.long),
        "static_real": torch.zeros(2, 0),
        "time_varying_real": torch.zeros(2, 5, 0),
        "time_varying_categorical": torch.zeros(2, 5, 0, dtype=torch.long),
        "padding_mask": torch.zeros(2, 5, dtype=torch.bool),
    }
    out = model.forward(batch)
    # representation ends in a LayerNorm whose weight inits to all-ones,
    # so representation.sum() cancels to sum(bias) with zero gradient
    # everywhere upstream. A squared-norm loss is non-degenerate.
    out.representation.pow(2).sum().backward()
    assert model.static_pad_vec.grad is not None
    assert model.tv_pad_vec.grad is not None
    assert model.static_pad_vec.grad.abs().sum() > 0
    assert model.tv_pad_vec.grad.abs().sum() > 0
    # The empty-side metrics path must also run (single synthetic
    # variable -> entropy 0).
    metrics = model.compute_training_metrics(out)
    vse = metrics["train.var_selection_entropy"]
    assert isinstance(vse, dict)
    assert vse["static_entropy"] == pytest.approx(0.0, abs=1e-6)
    assert vse["temporal_entropy"] == pytest.approx(0.0, abs=1e-6)


def test_compute_training_metrics_entropy_values_all_valid(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    # No padding: uniform weights -> ln(n) on every branch. Pins the
    # entropy formula sign/axis in the unmasked path.
    model = _backbone(make_tft_config)
    n_tv = 4
    n_static = 3
    n_heads = 2
    seq_len = 3
    out = TransformerBackboneOutput(
        representation=torch.zeros(2, 8),
        padding_mask=torch.zeros(2, seq_len, dtype=torch.bool),
        var_selection_weights=torch.full((2, seq_len, n_tv), 1.0 / n_tv),
        attention_weights=torch.full((2, n_heads, seq_len, seq_len), 1.0 / seq_len),
        static_var_selection_weights=torch.full((2, n_static), 1.0 / n_static),
    )
    metrics = model.compute_training_metrics(out)
    vse = metrics["train.var_selection_entropy"]
    assert isinstance(vse, dict)
    assert vse["temporal_entropy"] == pytest.approx(
        torch.tensor(float(n_tv)).log().item(), abs=1e-6
    )
    assert vse["static_entropy"] == pytest.approx(
        torch.tensor(float(n_static)).log().item(), abs=1e-6
    )
    ae = metrics["train.attention_entropy"]
    assert isinstance(ae, dict)
    expected = torch.tensor(float(seq_len)).log().item()
    assert ae["entropy_per_head"] == pytest.approx([expected, expected], abs=1e-6)


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
