"""Tests for the mask-polarity flip helper (A6)."""

import torch

from seq_sklearn.models._attention import to_attn_mask


def test_flip_turns_padding_into_participate() -> None:
    # True = padding -> True = participate.
    padding = torch.tensor([[True, False, False], [False, False, True]])
    attn = to_attn_mask(padding)
    expected = torch.tensor([[False, True, True], [True, True, False]])
    assert torch.equal(attn, expected)


def test_flip_is_idempotent_under_double_application() -> None:
    padding = torch.tensor([[True, False], [False, True]])
    assert torch.equal(to_attn_mask(to_attn_mask(padding)), padding)
