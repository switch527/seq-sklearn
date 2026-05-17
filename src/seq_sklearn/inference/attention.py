"""``predict_with_attention`` return types (architecture A15.1).

Two frozen, slotted dataclasses. BETA per the requirements stability
table: the field set may grow in a MINOR release, so attribute access
is the contract and tuple-unpacking is deliberately unsupported (a
slotted dataclass defines no ``__iter__``, so ``a, b = out`` raises
``TypeError``; a unit test pins this). The regression variant
intentionally omits ``logits``: a regression head emits raw scalars, so
``predictions`` already carries what ``logits`` exposes for a
classifier (qa r2-I3).
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["AttentionOutput", "RegressionAttentionOutput"]


@dataclass(frozen=True, slots=True)
class AttentionOutput:
    """Returned by ``TFTClassifier.predict_with_attention`` (A15.1)."""

    predictions: np.ndarray  # (N,) class indices
    probabilities: np.ndarray  # (N, num_classes); post sigmoid / softmax
    logits: np.ndarray  # (N, num_classes); pre-activation
    var_selection_weights: np.ndarray  # (N, L, n_vars)
    static_var_selection_weights: np.ndarray  # (N, n_static_vars)
    attention_weights: np.ndarray  # (N, n_heads, L, L)
    padding_mask: np.ndarray  # (N, L); True = padding
    entity_id: np.ndarray  # (N,) for diagnostics


@dataclass(frozen=True, slots=True)
class RegressionAttentionOutput:
    """Returned by ``TFTRegressor.predict_with_attention`` (A15.1)."""

    predictions: np.ndarray  # (N,) point or (N, len(quantiles)) quantile mode
    quantiles_used: tuple[float, ...] | None  # fit-time vector, None = point
    var_selection_weights: np.ndarray  # (N, L, n_vars)
    static_var_selection_weights: np.ndarray  # (N, n_static_vars)
    attention_weights: np.ndarray  # (N, n_heads, L, L)
    padding_mask: np.ndarray  # (N, L); True = padding
    entity_id: np.ndarray  # (N,) for diagnostics
