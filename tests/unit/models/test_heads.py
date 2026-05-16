"""Tests for the prediction heads (A6)."""

import pytest
import torch

from seq_sklearn.models._heads import ClassificationHead, RegressionHead


@pytest.mark.parametrize(
    ("d_model", "out_dim", "batch"),
    [(16, 1, 4), (16, 5, 3)],
)
def test_classification_head_output_shape(d_model: int, out_dim: int, batch: int) -> None:
    head = ClassificationHead(d_model, out_dim)
    h = torch.randn(batch, d_model)
    logits = head(h)
    assert logits.shape == (batch, out_dim)


def test_classification_head_gradients_flow() -> None:
    head = ClassificationHead(8, 3)
    h = torch.randn(2, 8, requires_grad=True)
    head(h).sum().backward()
    assert h.grad is not None
    assert head.proj.weight.grad is not None


@pytest.mark.parametrize(
    ("d_model", "out_dim", "n_quantiles", "batch"),
    [(16, 1, 1, 4), (16, 1, 3, 5), (16, 2, 1, 3), (16, 2, 4, 2)],
)
def test_regression_head_output_shape(
    d_model: int, out_dim: int, n_quantiles: int, batch: int
) -> None:
    head = RegressionHead(d_model, out_dim, n_quantiles)
    h = torch.randn(batch, d_model)
    out = head(h)
    assert out.shape == (batch, out_dim * n_quantiles)


def test_regression_head_gradients_flow() -> None:
    head = RegressionHead(8, 2, 3)
    h = torch.randn(2, 8, requires_grad=True)
    head(h).sum().backward()
    assert h.grad is not None
    assert head.proj.weight.grad is not None
