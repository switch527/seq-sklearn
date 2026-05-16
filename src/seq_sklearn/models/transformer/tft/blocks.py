"""TFT building blocks: GLU, AddNorm, GRN, VSN (per architecture A6).

Block order follows the Google Research TF1 reference and
pytorch-forecasting v1.3 (validated in ``docs/research/tft.md``):

- ``GLU``: gated linear unit, ``a * sigmoid(b)`` over a split projection.
- ``AddNorm``: residual add then ``LayerNorm``.
- ``GRN``: ``linear -> ELU -> linear -> dropout -> GLU -> AddNorm`` with
  an optional context vector added before the ELU and a projected skip
  when the input width differs from ``hidden_size``.
- ``VSN``: per-timestep variable selection. Padded positions are zeroed
  at the input level (so they never contribute) and the selection
  softmax runs over the variable axis, conditioned on a context vector.

Every linear / layer-norm is built through the F4 layer factory so the
v2 precision backend can swap layer types in one place.
"""

import torch
import torch.nn as nn
from torch import Tensor

from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.models._layers import make_layer_norm, make_linear

__all__ = ["GLU", "GRN", "VSN", "AddNorm"]


class GLU(nn.Module):
    """Gated linear unit: project to ``2 * out`` then ``a * sigmoid(b)``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = make_linear(in_features, out_features * 2)
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        """Gate ``x`` from ``(..., in_features)`` to ``(..., out_features)``."""
        projected = self.proj(x)  # (..., 2 * out_features)
        value, gate = projected.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class AddNorm(nn.Module):
    """Residual add followed by layer normalization."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = make_layer_norm(hidden_size)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        """Return ``LayerNorm(x + residual)``."""
        return self.norm(x + residual)


class GRN(nn.Module):
    """Gated residual network with an optional context vector.

    The skip connection is projected to ``hidden_size`` when the input
    width differs, so a GRN can both transform and resize. The context
    vector (when supplied) is added to the first linear's output before
    the ELU; its trailing shape must broadcast against the primary
    input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        context_size: int | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = make_linear(input_size, hidden_size)
        self.context_proj = (
            make_linear(context_size, hidden_size, bias=False) if context_size is not None else None
        )
        self.elu = nn.ELU()
        self.fc2 = make_linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_size, hidden_size)
        self.skip_proj = make_linear(input_size, hidden_size) if input_size != hidden_size else None
        self.add_norm = AddNorm(hidden_size)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        """Transform ``x`` to ``hidden_size`` with a gated residual.

        ``context``, when given, is added before the ELU; for a
        time-distributed primary input pass it broadcastable over the
        time axis.

        Raises:
            ValueError: ``context`` is supplied but the GRN was built
                without a ``context_size``, or vice versa.
        """
        if (context is None) != (self.context_proj is None):
            raise ValueError("GRN context presence must match the configured context_size")
        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        gated = self.glu(hidden)
        skip = x if self.skip_proj is None else self.skip_proj(x)
        return self.add_norm(gated, skip)


class VSN(nn.Module):
    """Per-timestep variable selection network (A6).

    Each of ``n_vars`` variables is a ``hidden_size``-wide vector. A GRN
    over the flattened variables, conditioned on a context vector,
    produces a softmax weight per variable; a per-variable GRN transforms
    each variable; the output is the weighted sum. Padded positions are
    zeroed at the flattened input so the softmax at those positions is
    well-defined but the position never contributes downstream.
    """

    def __init__(self, n_vars: int, config: TFTConfig) -> None:
        super().__init__()
        self.n_vars = n_vars
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.flatten_grn = GRN(
            input_size=n_vars * hidden,
            hidden_size=n_vars,
            dropout=config.variable_selection_dropout,
            context_size=hidden,
        )
        self.var_grns = nn.ModuleList(
            GRN(hidden, hidden, config.variable_selection_dropout) for _ in range(n_vars)
        )

    def forward(
        self, x: Tensor, context: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Select variables for ``x`` of shape ``(B, L, n_vars, hidden)``.

        ``context`` is ``(B, hidden)`` (broadcast over time). When
        ``padding_mask`` (``(B, L)``, ``True = padding``) is given, padded
        positions are zeroed before the selection softmax. Returns the
        selected ``(B, L, hidden)`` tensor and the per-timestep
        ``(B, L, n_vars)`` selection weights.
        """
        batch, seq_len, n_vars, hidden = x.shape
        if padding_mask is not None:
            keep = (~padding_mask).to(x.dtype)  # (B, L); 1 at valid positions
            x = x * keep[:, :, None, None]
        flat = x.reshape(batch, seq_len, n_vars * hidden)  # (B, L, n_vars * hidden)
        ctx = context[:, None, :].expand(batch, seq_len, -1)  # (B, L, hidden)
        weight_logits = self.flatten_grn(flat, ctx)  # (B, L, n_vars)
        weights = torch.softmax(weight_logits, dim=-1)  # (B, L, n_vars)
        transformed = torch.stack(
            [self.var_grns[i](x[:, :, i, :]) for i in range(n_vars)], dim=2
        )  # (B, L, n_vars, hidden)
        selected = (weights[..., None] * transformed).sum(dim=2)  # (B, L, hidden)
        return selected, weights
