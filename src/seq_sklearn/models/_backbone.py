"""Family-agnostic backbone output and abstract base (per architecture A15).

Generic training plumbing (the Phase 4 LightningModule) sees only the
two :class:`BackboneOutput` fields plus the dict returned by
:meth:`BaseBackbone.compute_training_metrics`. Concrete model families
subclass the dataclass to add introspection fields and override the
metric method to return family-specific event payloads.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

__all__ = ["BackboneOutput", "BaseBackbone"]


@dataclass
class BackboneOutput:
    """Family-agnostic backbone output.

    Plain ``@dataclass`` (not ``Protocol``) so concrete families inherit
    via standard dataclass subclassing and pyright strict mode passes
    without ``@runtime_checkable`` ceremony.
    """

    representation: Tensor  # (B, hidden_size)
    padding_mask: Tensor  # (B, L); True = padding (ignore)


class BaseBackbone(nn.Module, ABC):
    """Abstract sequence backbone.

    Subclasses implement :meth:`forward` returning a
    :class:`BackboneOutput` (or a family subclass) and may override
    :meth:`compute_training_metrics` to emit introspection payloads.
    """

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> BackboneOutput: ...

    def compute_training_metrics(self, output: BackboneOutput) -> dict[str, object]:  # noqa: ARG002
        """Return event-payload dicts keyed by F11 event name.

        Base returns ``{}`` so a backbone with no introspection (a v3
        recurrent base before any concrete model lands) emits nothing.
        Concrete backbones override to return one entry per event.
        """
        return {}
