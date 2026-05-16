"""Mask-polarity flip helper (per architecture A6).

The library's data contract carries ``padding_mask`` with
``True = padding`` (the position should be ignored). PyTorch attention
primitives expect a boolean ``attn_mask`` with ``True = participate``.
Every attention call site routes the mask through :func:`to_attn_mask`
so the polarity convention lives in exactly one place.
"""

from torch import Tensor

__all__ = ["to_attn_mask"]


def to_attn_mask(padding_mask: Tensor) -> Tensor:
    """Flip ``padding_mask`` (True = padding) to ``attn_mask`` (True = participate).

    The flip is its own inverse, so applying it twice recovers the
    original ``padding_mask``.
    """
    return ~padding_mask
