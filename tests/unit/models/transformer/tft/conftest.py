"""Shared TFT config / panel factories for the tft model tests."""

from collections.abc import Callable

import pytest

from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig


@pytest.fixture
def make_tft_config() -> Callable[..., TFTConfig]:
    """Return a factory building a valid TFTConfig with overrides."""

    def _make(**overrides: object) -> TFTConfig:
        tabular = overrides.pop(
            "tabular_config",
            TabularToSequenceConfig(id_col="id", time_col="time"),
        )
        return TFTConfig(
            task_type="binary",
            loss=LossConfig(strategy="cross_entropy"),
            tabular_config=tabular,
            **overrides,
        )

    return _make
