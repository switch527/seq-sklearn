"""Tests for the TFTConfig at architecture A4.

The TFT-specific cross-field validator (attention_heads divides
hidden_size) lives here. The base-class ``quantiles`` validator lives on
:class:`BaseModelConfig` and is exercised in ``test_config_base.py``;
this file does NOT re-test the quantile path.
"""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig


def _tab() -> TabularToSequenceConfig:
    return TabularToSequenceConfig(id_col="id", time_col="time")


def _minimal_tft(**overrides: object) -> TFTConfig:
    return TFTConfig(
        task_type="binary",
        loss_strategy="cross_entropy",
        tabular_config=_tab(),
        **overrides,
    )


def test_default_construction_succeeds() -> None:
    cfg = _minimal_tft()
    assert cfg.hidden_size == 128
    assert cfg.attention_heads == 4
    assert cfg.prediction_readout == "last_valid"


def test_attention_heads_must_divide_hidden_size() -> None:
    with pytest.raises(ValidationError, match="divide hidden_size"):
        _minimal_tft(hidden_size=130, attention_heads=4)


def test_attention_heads_one_always_legal() -> None:
    cfg = _minimal_tft(hidden_size=128, attention_heads=1)
    assert cfg.attention_heads == 1


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_tft(not_a_field=1)


def test_dropout_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_tft(dropout=1.5)


def test_frozen_post_construction() -> None:
    cfg = _minimal_tft()
    with pytest.raises(ValidationError):
        cfg.hidden_size = 64  # type: ignore[misc]


def test_prediction_readout_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        _minimal_tft(prediction_readout="argmax")  # type: ignore[arg-type]


def test_inherits_validity_matrix_validator() -> None:
    """TFTConfig inherits BaseModelConfig's validity check."""
    with pytest.raises(ValidationError):
        TFTConfig(
            task_type="regression_point",
            loss_strategy="cross_entropy",  # illegal for regression
            tabular_config=_tab(),
        )
