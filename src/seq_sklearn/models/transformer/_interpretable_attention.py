"""Shared-V interpretable multi-head self-attention (per architecture A6).

Per-head ``Q`` / ``K`` projections with a SINGLE ``V`` projection shared
across heads (mathematically a per-head ``V`` with all heads tied). Two
forward paths on the same module:

- fast: ``scaled_dot_product_attention``; returns the representation
  only, no per-head score tensor materialized.
- interpretable: manual softmax; returns the representation plus the
  post-softmax / pre-``V`` ``attn_weights`` tensor for introspection.

The mask routes through :func:`seq_sklearn.models._attention.to_attn_mask`
so the ``True = padding`` -> ``True = participate`` flip lives in one
place. ``predict_with_attention`` uses the interpretable path; fit /
predict / predict_proba use the fast path.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from seq_sklearn.models._attention import to_attn_mask
from seq_sklearn.models._layers import make_linear

__all__ = ["InterpretableMultiHeadAttention"]


class InterpretableMultiHeadAttention(nn.Module):
    """Self-attention with per-head Q / K and one shared V projection.

    ``hidden_size`` must be divisible by ``n_heads``; the per-head
    dimension is ``hidden_size // n_heads``.
    """

    def __init__(self, hidden_size: int, n_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = make_linear(hidden_size, hidden_size)
        self.k_proj = make_linear(hidden_size, hidden_size)
        self.v_proj = make_linear(hidden_size, self.head_dim)
        self.out_proj = make_linear(self.head_dim, hidden_size)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        # (B, L, hidden) -> (B, H, L, d)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x)  # (B, L, d); single shared V
        v_broadcast = v.unsqueeze(1).expand(batch, self.n_heads, seq_len, self.head_dim)
        return q, k, v_broadcast

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        """Fast path: SDPA, no per-head score tensor.

        ``x`` is ``(B, L, hidden_size)``; ``padding_mask`` is
        ``(B, L)`` with ``True = padding``. Returns ``(B, L, hidden_size)``.
        """
        q, k, v_broadcast = self._project(x)
        attn_mask_bool = to_attn_mask(padding_mask)  # (B, L); True = participate
        # (B, L) -> (B, 1, 1, L) broadcast over heads and query positions
        attn_mask = attn_mask_bool[:, None, None, :]
        out_per_head = scaled_dot_product_attention(  # (B, H, L, d)
            q, k, v_broadcast, attn_mask=attn_mask
        )
        out = out_per_head.mean(dim=1)  # (B, L, d)
        return self.out_proj(out)  # (B, L, hidden_size)

    def forward_interpretable(self, x: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Interpretable path: manual softmax, returns ``(out, attn_weights)``.

        ``attn_weights`` is ``(B, H, L, L)``, post-softmax and pre-``V``.
        A query row whose keys are all masked would softmax to NaN; the
        A6 ``mask.any(dim=1).all()`` invariant prevents that, and the
        ``nan_to_num`` pass is a defensive guard against an upstream
        mask bug.
        """
        q, k, v_broadcast = self._project(x)
        attn_mask_bool = to_attn_mask(padding_mask)  # (B, L); True = participate
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        scores = scores.masked_fill(~attn_mask_bool[:, None, None, :], float("-inf"))
        attn_weights = scores.softmax(dim=-1)  # (B, H, L, L); post-softmax, pre-V
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        out_per_head = attn_weights @ v_broadcast  # (B, H, L, d)
        out = out_per_head.mean(dim=1)  # (B, L, d)
        return self.out_proj(out), attn_weights
