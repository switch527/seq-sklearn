"""Positional encodings for transformer-family backbones (per architecture A6).

Two interchangeable encodings: a fixed sinusoidal table (no parameters)
and a learned per-position embedding. Both add a ``(1, L, d_model)``
positional signal to a ``(B, L, d_model)`` input.

``TFTBackbone`` deliberately does not use these: the LSTM encoder
already injects sequence order. They serve the attention-only
transformer-family backbones (PatchTST / TST) added in v2.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from seq_sklearn.models._layers import make_embedding

__all__ = ["LearnedPositionalEncoding", "SinusoidalPositionalEncoding"]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    The table is registered as a non-persistent buffer so it is rebuilt
    from ``max_len`` / ``d_model`` on load rather than stored in the
    checkpoint.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (ceil(d_model / 2),)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add the positional table to ``x`` of shape ``(B, L, d_model)``."""
        seq_len = x.size(1)
        pe = self.get_buffer("pe")
        return x + pe[:, :seq_len, :]  # (B, L, d_model)


class LearnedPositionalEncoding(nn.Module):
    """Learned per-position embedding added to the input."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.embedding = make_embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add the learned positional embedding to ``x`` of shape ``(B, L, d_model)``."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # (L,)
        return x + self.embedding(positions).unsqueeze(0)  # (B, L, d_model)
