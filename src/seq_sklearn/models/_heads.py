"""Prediction heads (per architecture A6).

The classification head emits raw logits; ``BCEWithLogitsLoss`` /
``CrossEntropyLoss`` apply the activation internally during training and
``predict_proba`` applies it post-hoc on cached logits. The regression
head projects to ``out_dim * n_quantiles`` so a single linear layer
serves point and quantile regressors.
"""

import torch.nn as nn
from torch import Tensor

from seq_sklearn.models._layers import make_linear

__all__ = ["ClassificationHead", "RegressionHead"]


class ClassificationHead(nn.Module):
    """Linear projection from the representation to class logits.

    ``out_dim`` is the tensor projection width (``num_classes`` for
    multiclass, ``1`` for binary), distinct from sklearn's
    ``n_outputs_`` attribute (F1.1).
    """

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.proj = make_linear(d_model, out_dim)

    def forward(self, h: Tensor) -> Tensor:
        """Project ``(B, d_model)`` to ``(B, out_dim)`` raw logits."""
        return self.proj(h)


class RegressionHead(nn.Module):
    """Linear projection to ``out_dim * n_quantiles`` regression outputs.

    A point regressor uses ``n_quantiles=1``; a quantile regressor sets
    ``n_quantiles`` to the quantile-vector length.
    """

    def __init__(self, d_model: int, out_dim: int, n_quantiles: int) -> None:
        super().__init__()
        self.proj = make_linear(d_model, out_dim * n_quantiles)

    def forward(self, h: Tensor) -> Tensor:
        """Project ``(B, d_model)`` to ``(B, out_dim * n_quantiles)``."""
        return self.proj(h)
