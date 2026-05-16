"""Test-only backbone / head / loss building blocks (per architecture A14).

These are the minimal pieces ``make_test_module`` composes so the
Phase 4 callback tests (and the Phase 4b LightningModule tests) can
stand up an isolated module without an Estimator or a real TFT. The
full ``_DummySequenceClassifier`` composition lands in Phase 6a; this
module deliberately ships only the three primitives.

``_DummyBackbone`` extends the Phase 3 :class:`BaseBackbone` and keeps
its default empty :meth:`compute_training_metrics` (it returns ``{}``)
so the cross-family abstraction is exercised with no introspection
payload, exactly as a v3 recurrent base would behave before a concrete
model overrides it.
"""

import torch
import torch.nn as nn
from torch import Tensor

from seq_sklearn.models._backbone import BackboneOutput, BaseBackbone

__all__ = ["_DummyBackbone", "_DummyHead", "_LossReturningScalar"]


class _DummyBackbone(BaseBackbone):
    """Smallest concrete :class:`BaseBackbone`.

    Carries one trainable parameter so an optimizer factory has
    something to optimize, and returns a :class:`BackboneOutput` whose
    ``representation`` is a learned affine of the batch's ``features``
    tensor. Keeps the inherited empty ``compute_training_metrics``.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)

    def forward(self, batch: dict[str, Tensor]) -> BackboneOutput:
        """Project ``batch["features"]`` (B, in_dim) to (B, hidden)."""
        features = batch["features"]
        representation = self.proj(features)  # (B, in_dim) -> (B, hidden)
        padding_mask = torch.zeros(features.shape[0], 1, dtype=torch.bool, device=features.device)
        return BackboneOutput(representation=representation, padding_mask=padding_mask)


class _DummyHead(nn.Module):
    """Linear head mapping the backbone representation to a single logit."""

    def __init__(self, hidden: int = 3) -> None:
        super().__init__()
        self.out = nn.Linear(hidden, 1)

    def forward(self, representation: Tensor) -> Tensor:
        """Project ``(B, hidden)`` to ``(B, 1)`` logits."""
        return self.out(representation)


class _LossReturningScalar(nn.Module):
    """Loss stub returning a fixed finite scalar.

    The NaN-loss Variant A test monkey-patches an instance's ``forward``
    to return ``torch.tensor(float("nan"))`` on the offending step; the
    default path returns a constant so a non-NaN step is well-defined.
    """

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return a constant scalar independent of the inputs."""
        return torch.tensor(self._value, requires_grad=True)
