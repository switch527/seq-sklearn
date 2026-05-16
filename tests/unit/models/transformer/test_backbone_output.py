"""Tests for the transformer-family backbone output dataclass (A15)."""

import torch

from seq_sklearn.models._backbone import BackboneOutput
from seq_sklearn.models.transformer._backbone import TransformerBackboneOutput


def test_transformer_output_is_a_backbone_output_with_extra_fields() -> None:
    out = TransformerBackboneOutput(
        representation=torch.zeros(2, 8),
        padding_mask=torch.zeros(2, 4, dtype=torch.bool),
        var_selection_weights=torch.zeros(2, 4, 3),
        attention_weights=torch.zeros(2, 2, 4, 4),
        static_var_selection_weights=torch.zeros(2, 5),
    )
    assert isinstance(out, BackboneOutput)
    assert out.representation.shape == (2, 8)
    assert out.padding_mask.shape == (2, 4)
    assert out.var_selection_weights.shape == (2, 4, 3)
    assert out.attention_weights.shape == (2, 2, 4, 4)
    assert out.static_var_selection_weights.shape == (2, 5)
