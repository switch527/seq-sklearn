"""``TFTClassifier``: v1's first real classification model (A2 / Phase 7).

Composes the shared TFT-family wiring (`_TFTEstimatorMixin`: A4-adapter
``__init__``, ``TFTConfig`` bridge), the transformer-family attention
mixin (`TransformerSequenceEstimator.Classifier`:
``predict_with_attention``), and the Phase-6a classifier shell
(`BaseSequenceClassifier`: ``predict`` / ``predict_proba`` /
``classes_`` / calibrator / threshold tuner). The only TFT-specific
code is ``_build_backbone_head``; the family-wide ``__sklearn_tags__``
is inherited unchanged (no v1 TFT tag flip).
"""

from typing import cast

from torch import nn

from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models._classifier import BaseSequenceClassifier
from seq_sklearn.models._heads import ClassificationHead
from seq_sklearn.models.transformer._base import TransformerSequenceEstimator
from seq_sklearn.models.transformer.tft._estimator import _TFTEstimatorMixin

__all__ = ["TFTClassifier"]


class TFTClassifier(
    _TFTEstimatorMixin,
    TransformerSequenceEstimator.Classifier,
    BaseSequenceClassifier,
):
    """Temporal Fusion Transformer classifier (binary / multiclass)."""

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        hidden = cast("TFTConfig", config).hidden_size
        return (
            self._build_tft_backbone(config, transformer),
            ClassificationHead(hidden, self._head_out_dim()),
        )
