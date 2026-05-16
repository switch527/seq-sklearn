"""Synthetic concrete estimators that exercise the Phase 6a shell (A14).

``_DummySequenceClassifier`` / ``_DummySequenceRegressor`` are the
smallest concrete subclasses of the family bases: a trivial pooled-linear
backbone over the real ``TabularToSequence`` batch, the real
``ClassificationHead`` / ``RegressionHead``, and the real
Trainer / loss / optimizer / scheduler. They let the estimator shell,
save / load, and the calibrator seam be tested end-to-end before TFT
specifics land in Phase 7. Test-only: never shipped under ``src/``.
"""

import torch
import torch.nn as nn
from torch import Tensor

from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models._backbone import BackboneOutput, BaseBackbone
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models._heads import ClassificationHead, RegressionHead
from seq_sklearn.models._regressor import BaseSequenceRegressor

__all__ = ["_DummySequenceClassifier", "_DummySequenceRegressor"]

_HIDDEN = 8


class _PooledLinearBackbone(BaseBackbone):
    """Mask-aware mean-pool of the real columns -> a learned representation.

    Consumes the real ``TabularToSequence`` batch dict (static_real +
    masked-mean time_varying_real, plus a constant column so ``in_dim``
    is never zero). Carries trainable parameters so the optimizer has
    something to fit; keeps the inherited empty
    ``compute_training_metrics``.
    """

    def __init__(self, n_static_real: int, n_tv_real: int, hidden: int) -> None:
        super().__init__()
        self._in_dim = n_static_real + n_tv_real + 1
        self.proj = nn.Linear(self._in_dim, hidden)

    def forward(self, batch: dict[str, Tensor]) -> BackboneOutput:
        static_real = batch["static_real"]  # (B, n_sr)
        tv_real = batch["time_varying_real"]  # (B, L, n_tr)
        mask = batch["padding_mask"]  # (B, L); True = padding
        valid = (~mask).to(tv_real.dtype).unsqueeze(-1)  # (B, L, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)  # (B, 1)
        tv_mean = (tv_real * valid).sum(dim=1) / denom  # (B, n_tr)
        const = torch.ones(
            static_real.shape[0], 1, dtype=tv_real.dtype, device=tv_real.device
        )
        feats = torch.cat([static_real.to(tv_real.dtype), tv_mean, const], dim=1)
        representation = self.proj(feats)  # (B, in_dim) -> (B, hidden)
        return BackboneOutput(representation=representation, padding_mask=mask)


def _backbone(transformer: TabularToSequence, hidden: int) -> _PooledLinearBackbone:
    cfg = transformer.config
    return _PooledLinearBackbone(
        n_static_real=len(cfg.static_real_cols),
        n_tv_real=len(cfg.time_varying_real_cols),
        hidden=hidden,
    )


class _DummySequenceClassifier(BaseSequenceClassifier):
    """Concrete classifier shell: pooled-linear backbone + real head."""

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        return (
            _backbone(transformer, _HIDDEN),
            ClassificationHead(_HIDDEN, self._head_out_dim()),
        )


class _DummySequenceRegressor(BaseSequenceRegressor):
    """Concrete regressor shell: pooled-linear backbone + real head."""

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        return (
            _backbone(transformer, _HIDDEN),
            RegressionHead(_HIDDEN, self._head_out_dim(), self._n_quantiles()),
        )
