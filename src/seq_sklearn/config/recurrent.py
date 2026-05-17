"""Recurrent-family config skeleton (architecture A6.1, INTERNAL in v1).

Ships in v1 as a schema only: no concrete LSTM / GRU consumes it until
v3. Its purpose is to lock the v1 -> v3 cross-family config surface so
the recurrent forward port lands without a shape break. INTERNAL-tier:
absent from the public re-export (A3); v3 promotes it to STABLE.

Extends :class:`seq_sklearn.config.base.BaseModelConfig`, so it inherits
the frozen / ``extra="forbid"`` contract, the required ``task_type`` /
``loss`` fields, and the three cross-field validators (validity matrix,
quantile monotonicity, val+cal sum).
"""

from typing import Literal

from pydantic import Field

from seq_sklearn.config.base import BaseModelConfig

__all__ = ["RecurrentSequenceEstimatorConfig"]


class RecurrentSequenceEstimatorConfig(BaseModelConfig):
    """Shared recurrent hyperparameters (A6.1 skeleton, INTERNAL v1).

    ``recurrent_dropout_kind`` selects a library dropout wrapper, NOT
    PyTorch's ``nn.LSTM(dropout=)`` (which is inter-layer only):
    ``weight_drop`` (AWD-LSTM style, keeps cuDNN), ``variational``
    (loses cuDNN), ``bernoulli`` (per-step independent, weaker). The
    dispatch lands with the concrete recurrent models in v3.
    """

    bidirectional: bool = False
    recurrent_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    recurrent_dropout_kind: Literal["weight_drop", "variational", "bernoulli"] = (
        "weight_drop"
    )
    hidden_init_strategy: Literal["zero", "learned", "per_entity"] = "zero"
    readout: Literal["last_valid", "mean_pool", "attention"] = "last_valid"
    bptt_window: int | None = Field(default=None, ge=1)
