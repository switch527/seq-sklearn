"""Tests for shared-V interpretable multi-head attention (A6 / N1)."""

import torch

from seq_sklearn.models.transformer._interpretable_attention import (
    InterpretableMultiHeadAttention,
)


def _seeded_input(batch: int, seq_len: int, hidden: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, hidden, generator=gen)


def test_shared_v_projection_weight_count_is_one_not_n_heads() -> None:
    model = InterpretableMultiHeadAttention(hidden_size=16, n_heads=4)
    v_weights = [name for name, _ in model.named_parameters() if name == "v_proj.weight"]
    assert v_weights == ["v_proj.weight"]
    # The shared V projects to head_dim, not n_heads * head_dim.
    assert model.v_proj.weight.shape == (model.head_dim, 16)


def test_fast_path_equals_interpretable_path() -> None:
    model = InterpretableMultiHeadAttention(hidden_size=16, n_heads=4)
    model.eval()
    x = _seeded_input(3, 5, 16, seed=7)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    with torch.no_grad():
        fast = model(x, mask)
        interp, _ = model.forward_interpretable(x, mask)
    assert torch.allclose(fast, interp, atol=1e-5, rtol=1e-5)


def test_mask_correctness_unpadded_vs_padded() -> None:
    torch.use_deterministic_algorithms(True)
    try:
        model = InterpretableMultiHeadAttention(hidden_size=8, n_heads=2)
        model.eval()
        gen = torch.Generator().manual_seed(11)
        valid_len = 4
        total_len = 7
        entity = torch.randn(1, valid_len, 8, generator=gen)

        padded = torch.zeros(1, total_len, 8)
        padded[:, total_len - valid_len :, :] = entity
        pad_mask = torch.ones(1, total_len, dtype=torch.bool)
        pad_mask[:, total_len - valid_len :] = False

        unpadded_mask = torch.zeros(1, valid_len, dtype=torch.bool)
        with torch.no_grad():
            out_unpadded, _ = model.forward_interpretable(entity, unpadded_mask)
            out_padded, _ = model.forward_interpretable(padded, pad_mask)
        valid_slice = out_padded[:, total_len - valid_len :, :]
        assert torch.equal(valid_slice, out_unpadded)
    finally:
        torch.use_deterministic_algorithms(False)


def test_fast_path_mask_correctness_unpadded_vs_padded() -> None:
    # The fast SDPA path wires the mask through a (B, 1, 1, L) broadcast.
    # A polarity flip or wrong broadcast would silently attend to
    # padding during fit/predict. Mirror the interpretable-path check on
    # forward() so the SDPA mask plumbing is exercised directly.
    torch.use_deterministic_algorithms(True)
    try:
        model = InterpretableMultiHeadAttention(hidden_size=8, n_heads=2)
        model.eval()
        gen = torch.Generator().manual_seed(13)
        valid_len = 4
        total_len = 7
        entity = torch.randn(1, valid_len, 8, generator=gen)

        padded = torch.zeros(1, total_len, 8)
        padded[:, total_len - valid_len :, :] = entity
        pad_mask = torch.ones(1, total_len, dtype=torch.bool)
        pad_mask[:, total_len - valid_len :] = False

        unpadded_mask = torch.zeros(1, valid_len, dtype=torch.bool)
        with torch.no_grad():
            out_unpadded = model(entity, unpadded_mask)
            out_padded = model(padded, pad_mask)
        valid_slice = out_padded[:, total_len - valid_len :, :]
        assert torch.allclose(valid_slice, out_unpadded, atol=1e-6, rtol=0.0)
    finally:
        torch.use_deterministic_algorithms(False)


def test_fully_masked_query_row_is_zeroed_by_nan_to_num() -> None:
    # Every key masked -> softmax over all -inf is NaN. The nan_to_num
    # pass must zero it. Removing that line leaves NaN here, so this
    # test fails if the guard is deleted.
    model = InterpretableMultiHeadAttention(hidden_size=8, n_heads=2)
    model.eval()
    x = _seeded_input(1, 4, 8, seed=5)
    mask = torch.ones(1, 4, dtype=torch.bool)  # entire row is padding
    with torch.no_grad():
        out, attn = model.forward_interpretable(x, mask)
    assert not torch.isnan(out).any()
    assert not torch.isnan(attn).any()
    assert torch.equal(attn, torch.zeros_like(attn))


def test_interpretable_path_has_no_nan_under_mask_invariant() -> None:
    model = InterpretableMultiHeadAttention(hidden_size=8, n_heads=2)
    model.eval()
    x = _seeded_input(2, 6, 8, seed=3)
    # Mask invariant: at least one valid key per batch element.
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, 4:] = True
    with torch.no_grad():
        out, attn = model.forward_interpretable(x, mask)
    assert not torch.isnan(out).any()
    assert not torch.isnan(attn).any()
