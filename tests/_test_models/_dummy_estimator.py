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
from seq_sklearn.models.transformer._backbone import TransformerBackboneOutput
from seq_sklearn.models.transformer._base import TransformerSequenceEstimator

__all__ = [
    "_DummySequenceClassifier",
    "_DummySequenceRegressor",
    "_DummyTransformerClassifier",
    "_DummyTransformerRegressor",
]

_HIDDEN = 8
_N_HEADS = 2
_N_STATIC_VARS = 2


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
        const = torch.ones(static_real.shape[0], 1, dtype=tv_real.dtype, device=tv_real.device)
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


class _DummyTransformerBackbone(BaseBackbone):
    """Transformer-flavoured dummy: emits a non-degenerate
    :class:`TransformerBackboneOutput`.

    The mean-pool representation is the same trivial stand-in the
    Phase-6a dummy uses; what matters here is that the three
    introspection tensors are NON-degenerate (real softmaxes over
    >1 elements, padded keys masked out of the attention) so the
    Phase 6b attention-extraction path and the F11 entropy code see
    realistic values instead of all-uniform / all-zero tensors.
    """

    def __init__(self, n_static_real: int, n_tv_real: int, hidden: int) -> None:
        super().__init__()
        self._n_vars = max(n_tv_real, 1)
        self._in_dim = n_static_real + n_tv_real + 1
        self.proj = nn.Linear(self._in_dim, hidden)
        self.tproj = nn.Linear(self._n_vars, hidden)  # per-timestep attn embedding
        self.static_logits = nn.Parameter(torch.randn(_N_STATIC_VARS))

    def forward(self, batch: dict[str, Tensor]) -> TransformerBackboneOutput:
        static_real = batch["static_real"]  # (B, n_sr)
        tv_real = batch["time_varying_real"]  # (B, L, n_tr)
        mask = batch["padding_mask"]  # (B, L); True = padding
        b, length, _n_tr = tv_real.shape
        valid = (~mask).to(tv_real.dtype).unsqueeze(-1)  # (B, L, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)  # (B, 1)
        tv_mean = (tv_real * valid).sum(dim=1) / denom  # (B, n_tr)
        const = torch.ones(b, 1, dtype=tv_real.dtype, device=tv_real.device)
        feats = torch.cat([static_real.to(tv_real.dtype), tv_mean, const], dim=1)
        representation = self.proj(feats)  # (B, hidden)

        vars_in = tv_real if tv_real.shape[-1] >= 1 else tv_mean.unsqueeze(1)
        var_selection_weights = torch.softmax(vars_in, dim=-1)  # (B, L, n_vars)

        emb = self.tproj(vars_in)  # (B, L, hidden)
        scores = emb @ emb.transpose(-2, -1) / (emb.shape[-1] ** 0.5)  # (B, L, L)
        key_pad = mask[:, None, :]  # (B, 1, L); True = padding key
        scores = scores.masked_fill(key_pad, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, L, L)
        attn = torch.nan_to_num(attn, nan=0.0)  # all-pad query row -> 0
        attention_weights = attn.unsqueeze(1).expand(b, _N_HEADS, length, length)

        static_var_selection_weights = torch.softmax(self.static_logits, dim=-1).expand(
            b, _N_STATIC_VARS
        )
        return TransformerBackboneOutput(
            representation=representation,
            padding_mask=mask,
            var_selection_weights=var_selection_weights,
            attention_weights=attention_weights,
            static_var_selection_weights=static_var_selection_weights,
        )


def _transformer_backbone(transformer: TabularToSequence, hidden: int) -> _DummyTransformerBackbone:
    cfg = transformer.config
    return _DummyTransformerBackbone(
        n_static_real=len(cfg.static_real_cols),
        n_tv_real=len(cfg.time_varying_real_cols),
        hidden=hidden,
    )


class _DummyTransformerClassifier(TransformerSequenceEstimator.Classifier, BaseSequenceClassifier):
    """Transformer-family classifier: mixin + Phase-6a classifier shell."""

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        return (
            _transformer_backbone(transformer, _HIDDEN),
            ClassificationHead(_HIDDEN, self._head_out_dim()),
        )


class _DummyTransformerRegressor(TransformerSequenceEstimator.Regressor, BaseSequenceRegressor):
    """Transformer-family regressor: mixin + Phase-6a regressor shell."""

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        return (
            _transformer_backbone(transformer, _HIDDEN),
            RegressionHead(_HIDDEN, self._head_out_dim(), self._n_quantiles()),
        )
