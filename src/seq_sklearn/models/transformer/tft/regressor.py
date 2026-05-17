"""``TFTRegressor``: v1's first real regression model (A2 / Phase 7).

Composes the shared TFT-family wiring (`_TFTEstimatorMixin`), the
transformer-family attention mixin
(`TransformerSequenceEstimator.Regressor`: ``predict_with_attention``),
and the Phase-6a regressor shell (`BaseSequenceRegressor`: ``predict``
/ ``predict_quantiles`` / ``quantiles_`` / conformal / isotonic-quantile
calibrators). Point and quantile modes follow ``task_type`` /
``quantiles`` exactly as the shell defines them; the only TFT-specific
code is ``_build_backbone_head``.
"""

from typing import cast

from torch import nn

from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models._heads import RegressionHead
from seq_sklearn.models._regressor import BaseSequenceRegressor
from seq_sklearn.models.transformer._base import TransformerSequenceEstimator
from seq_sklearn.models.transformer.tft._estimator import _TFTEstimatorMixin

__all__ = ["TFTRegressor"]


class TFTRegressor(
    _TFTEstimatorMixin,
    TransformerSequenceEstimator.Regressor,
    BaseSequenceRegressor,
):
    """Temporal Fusion Transformer regressor (point / quantile)."""

    def _build_backbone_head(
        self,
        config: BaseModelConfig,
        transformer: TabularToSequence,
    ) -> tuple[nn.Module, nn.Module]:
        hidden = cast("TFTConfig", config).hidden_size
        return (
            self._build_tft_backbone(config, transformer),
            RegressionHead(hidden, self._head_out_dim(), self._n_quantiles()),
        )
