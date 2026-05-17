"""Recurrent-family abstract base (architecture A6.1, INTERNAL in v1).

v1 ships the abstract surface only: no concrete LSTM / GRU exists until
v3. Landing it now (alongside the Phase 6a shells) locks the v1 -> v3
cross-family contract so the recurrent forward port does not force a
`BaseSequenceEstimator` shape break (future-proofing item 7).

The three abstract hooks are the recurrent surface a v3 concrete model
fills; the Phase-6a `BaseSequenceEstimator` fit / predict shell does NOT
call them (they feed a recurrent backbone's own forward, landing in
v3). A v1 contract-lock test composes a no-op subclass with the shell to
prove the surface does not break it. INTERNAL-tier: absent from the A3
public re-export; v3 promotes it to STABLE.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from seq_sklearn.models._base import BaseSequenceEstimator

__all__ = ["RecurrentSequenceEstimator"]


class RecurrentSequenceEstimator(BaseSequenceEstimator, ABC):
    """Abstract recurrent estimator (A6.1 skeleton, INTERNAL v1).

    Still abstract: it adds the recurrent hooks but does not supply
    :meth:`BaseSequenceEstimator._build_backbone_head`, so it cannot be
    instantiated until a concrete v3 family (or a test no-op) fills both
    that and the three methods below.
    """

    @abstractmethod
    def _init_hidden(self, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
        """Initial recurrent state (v3 strategies: zero / learned / per-entity)."""

    @abstractmethod
    def _readout(self, hidden_seq: Tensor, mask: Tensor) -> Tensor:
        """Per-window representation from the hidden sequence and pad mask.

        ``hidden_seq`` is ``(B, L, H)``; ``mask`` is ``(B, L)`` with
        ``True`` = padding. v3 strategies: last valid, masked mean pool,
        attention readout.
        """

    @abstractmethod
    def _bptt_window(self) -> int | None:
        """Truncated-BPTT window length, or ``None`` for full BPTT."""
