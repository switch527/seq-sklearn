"""Tests for the positional encodings (A6)."""

import torch

from seq_sklearn.models.transformer._positional import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)


def test_sinusoidal_adds_signal_and_preserves_shape() -> None:
    enc = SinusoidalPositionalEncoding(d_model=8, max_len=16)
    x = torch.zeros(2, 5, 8)
    out = enc(x)
    assert out.shape == (2, 5, 8)
    # A zero input becomes the positional table, so position 0 != position 1.
    assert not torch.equal(out[0, 0], out[0, 1])


def test_sinusoidal_odd_d_model_shape() -> None:
    enc = SinusoidalPositionalEncoding(d_model=7, max_len=10)
    out = enc(torch.randn(1, 4, 7))
    assert out.shape == (1, 4, 7)


def test_sinusoidal_table_is_non_persistent_buffer() -> None:
    enc = SinusoidalPositionalEncoding(d_model=8, max_len=16)
    assert "pe" not in enc.state_dict()


def test_learned_adds_signal_and_preserves_shape() -> None:
    enc = LearnedPositionalEncoding(d_model=6, max_len=12)
    x = torch.zeros(3, 4, 6)
    out = enc(x)
    assert out.shape == (3, 4, 6)
    out.sum().backward()
    assert enc.embedding.weight.grad is not None
