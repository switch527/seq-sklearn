"""Tests for the TFT blocks: GLU, AddNorm, GRN, VSN (A6 / N1)."""

from collections.abc import Callable

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.models.transformer.tft.blocks import GLU, GRN, VSN, AddNorm


def _seeded(*shape: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen)


def test_glu_maps_hidden_to_hidden() -> None:
    glu = GLU(8, 8)
    out = glu(_seeded(2, 4, 8))
    assert out.shape == (2, 4, 8)


def test_addnorm_maps_hidden_to_hidden() -> None:
    add_norm = AddNorm(8)
    out = add_norm(_seeded(2, 4, 8, seed=1), _seeded(2, 4, 8, seed=2))
    assert out.shape == (2, 4, 8)


def test_grn_maps_hidden_to_hidden() -> None:
    grn = GRN(8, 8, dropout=0.1)
    out = grn(_seeded(2, 4, 8, seed=3))
    assert out.shape == (2, 4, 8)


def test_grn_resizes_input_via_skip_projection() -> None:
    grn = GRN(16, 8, dropout=0.0)
    out = grn(_seeded(2, 4, 16, seed=4))
    assert out.shape == (2, 4, 8)


def test_grn_with_context_maps_hidden_to_hidden() -> None:
    grn = GRN(8, 8, dropout=0.0, context_size=8)
    out = grn(_seeded(2, 4, 8, seed=5), context=_seeded(2, 4, 8, seed=6))
    assert out.shape == (2, 4, 8)


def test_grn_context_mismatch_raises() -> None:
    grn = GRN(8, 8, dropout=0.0, context_size=8)
    with pytest.raises(ValueError, match="context presence"):
        grn(_seeded(2, 4, 8, seed=7))
    grn_no_ctx = GRN(8, 8, dropout=0.0)
    with pytest.raises(ValueError, match="context presence"):
        grn_no_ctx(_seeded(2, 4, 8, seed=8), context=_seeded(2, 4, 8, seed=9))


def test_vsn_maps_to_hidden_and_weights(make_tft_config: Callable[..., TFTConfig]) -> None:
    cfg = make_tft_config(hidden_size=8, attention_heads=2)
    vsn = VSN(n_vars=3, config=cfg)
    x = _seeded(2, 4, 3, 8, seed=10)
    context = _seeded(2, 8, seed=11)
    selected, weights = vsn(x, context)
    assert selected.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_vsn_masking_neutralizes_padded_inputs(
    make_tft_config: Callable[..., TFTConfig],
) -> None:
    # Asserting only valid-slice equality is vacuous: the VSN is
    # per-timestep independent, so valid outputs never depend on padded
    # inputs whether or not the masking line exists. The discriminating
    # property of `x = x * keep` is that padded-position inputs are
    # neutralized: two batches with identical valid tails but different
    # non-zero garbage in the padded head must produce bitwise-identical
    # outputs everywhere. Removing the masking line breaks this.
    torch.use_deterministic_algorithms(True)
    try:
        cfg = make_tft_config(hidden_size=8, attention_heads=2)
        vsn = VSN(n_vars=3, config=cfg)
        vsn.eval()
        gen = torch.Generator().manual_seed(21)
        valid_len = 4
        total_len = 7
        entity = torch.randn(1, valid_len, 3, 8, generator=gen)
        context = torch.randn(1, 8, generator=gen)

        pad_mask = torch.ones(1, total_len, dtype=torch.bool)
        pad_mask[:, total_len - valid_len :] = False
        unpadded_mask = torch.zeros(1, valid_len, dtype=torch.bool)

        a = torch.randn(1, total_len, 3, 8, generator=gen)
        b = torch.randn(1, total_len, 3, 8, generator=gen)
        a[:, total_len - valid_len :] = entity
        b[:, total_len - valid_len :] = entity

        with torch.no_grad():
            sel_unpadded, w_unpadded = vsn(entity, context, unpadded_mask)
            sel_a, w_a = vsn(a, context, pad_mask)
            sel_b, w_b = vsn(b, context, pad_mask)

        # Masking-sensitive: same shape, same mask, different padded
        # garbage -> identical outputs only if the inputs were zeroed.
        assert torch.equal(sel_a, sel_b)
        assert torch.equal(w_a, w_b)
        # N1 non-leakage: valid slice matches the unpadded reference.
        # Cross-shape (L = 4 vs L = 7) GEMM tiling is not bit-pinned by
        # CPU deterministic mode, so a tight allclose is the strongest
        # achievable check without a false negative from low-order bits.
        valid_sel = sel_a[:, total_len - valid_len :, :]
        valid_w = w_a[:, total_len - valid_len :, :]
        assert torch.allclose(valid_sel, sel_unpadded, atol=1e-6, rtol=0.0)
        assert torch.allclose(valid_w, w_unpadded, atol=1e-6, rtol=0.0)
    finally:
        torch.use_deterministic_algorithms(False)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    batch=st.integers(min_value=1, max_value=3),
    seq_len=st.integers(min_value=1, max_value=6),
    hidden=st.sampled_from([4, 8, 16]),
)
def test_grn_vsn_shape_invariant(batch: int, seq_len: int, hidden: int) -> None:
    gen = torch.Generator().manual_seed(batch * 100 + seq_len * 10 + hidden)
    grn = GRN(hidden, hidden, dropout=0.0)
    grn.eval()
    x = torch.randn(batch, seq_len, hidden, generator=gen)
    with torch.no_grad():
        assert grn(x).shape == x.shape

    cfg = TFTConfig(
        task_type="binary",
        loss=LossConfig(strategy="cross_entropy"),
        tabular_config=TabularToSequenceConfig(id_col="id", time_col="time"),
        hidden_size=hidden,
        attention_heads=2,
    )
    n_vars = 3
    vsn = VSN(n_vars=n_vars, config=cfg)
    vsn.eval()
    vx = torch.randn(batch, seq_len, n_vars, hidden, generator=gen)
    vc = torch.randn(batch, hidden, generator=gen)
    with torch.no_grad():
        sel, w = vsn(vx, vc)
    assert sel.shape == (batch, seq_len, hidden)
    assert w.shape == (batch, seq_len, n_vars)
