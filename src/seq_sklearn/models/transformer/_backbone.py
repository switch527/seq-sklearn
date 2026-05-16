"""Transformer-family backbone output (per architecture A15).

Extends :class:`seq_sklearn.models._backbone.BackboneOutput` with the
three introspection tensors every transformer-family backbone exposes
(TFT in v1; PatchTST / TimesNet / TST in v2). Generic training plumbing
still consumes only the two base fields; the extra fields feed
``compute_training_metrics`` and ``predict_with_attention``.
"""

from dataclasses import dataclass

from torch import Tensor

from seq_sklearn.models._backbone import BackboneOutput

__all__ = ["TransformerBackboneOutput"]


@dataclass
class TransformerBackboneOutput(BackboneOutput):
    """Transformer-family backbone output with introspection tensors."""

    var_selection_weights: Tensor  # (B, L, n_vars); softmaxed per timestep
    attention_weights: Tensor  # (B, n_heads, L, L); post-softmax, pre-V
    static_var_selection_weights: Tensor  # (B, n_static_vars); softmaxed across static
