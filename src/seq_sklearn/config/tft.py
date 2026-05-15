"""TFT-specific pydantic configuration (per architecture A4).

Extends :class:`seq_sklearn.config.base.BaseModelConfig` with the
hyperparameters specific to the Temporal Fusion Transformer backbone.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig

__all__ = ["TFTConfig"]


class TFTConfig(BaseModelConfig):
    """Configuration for :class:`TFTClassifier` / :class:`TFTRegressor`."""

    hidden_size: int = Field(default=128, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    variable_selection_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    prediction_readout: Literal["last_valid", "mean_pool"] = "last_valid"
    tabular_config: TabularToSequenceConfig

    @model_validator(mode="after")
    def _check_heads_divide_hidden(self) -> Self:
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError(
                f"attention_heads ({self.attention_heads}) must divide "
                f"hidden_size ({self.hidden_size})."
            )
        return self
