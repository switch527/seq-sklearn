"""Tests for the primitive-layer factory (A6 / F4)."""

import torch
import torch.nn as nn

from seq_sklearn.models._layers import make_embedding, make_layer_norm, make_linear


def test_make_linear_returns_linear_with_shape() -> None:
    layer = make_linear(8, 16)
    assert isinstance(layer, nn.Linear)
    assert layer.in_features == 8
    assert layer.out_features == 16
    assert layer.bias is not None


def test_make_linear_bias_false() -> None:
    layer = make_linear(4, 2, bias=False)
    assert isinstance(layer, nn.Linear)
    assert layer.bias is None


def test_make_layer_norm_returns_layer_norm_with_shape() -> None:
    layer = make_layer_norm(32)
    assert isinstance(layer, nn.LayerNorm)
    assert tuple(layer.normalized_shape) == (32,)


def test_make_embedding_returns_embedding_with_shape() -> None:
    layer = make_embedding(10, 4)
    assert isinstance(layer, nn.Embedding)
    assert layer.num_embeddings == 10
    assert layer.embedding_dim == 4
    out = layer(torch.zeros(3, dtype=torch.long))
    assert out.shape == (3, 4)
