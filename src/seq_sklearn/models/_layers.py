"""Single construction site for the primitive ``nn`` layers (A6 / F4).

Every ``nn.Linear``, ``nn.LayerNorm``, and ``nn.Embedding`` the backbone
needs is built here. v1 returns standard PyTorch layers; v2's FP8 pass
(per N5) can swap in Transformer Engine equivalents in this one module
without touching any calling code.
"""

import torch.nn as nn

__all__ = ["make_embedding", "make_layer_norm", "make_linear"]


def make_linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
    """Build a fully connected layer.

    Routed through this factory so a future precision backend can
    substitute the layer type without changing call sites.
    """
    return nn.Linear(in_features, out_features, bias=bias)


def make_layer_norm(normalized_shape: int) -> nn.LayerNorm:
    """Build a layer-normalization layer over the last dimension."""
    return nn.LayerNorm(normalized_shape)


def make_embedding(num_embeddings: int, embedding_dim: int) -> nn.Embedding:
    """Build a categorical embedding table."""
    return nn.Embedding(num_embeddings, embedding_dim)
