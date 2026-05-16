"""Tests for the Tier 1 LossConfig sub-config."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.loss import LossConfig


def test_construction_requires_strategy() -> None:
    with pytest.raises(ValidationError):
        LossConfig()  # type: ignore[call-arg]


def test_valid_construction_uses_documented_defaults() -> None:
    """Happy path: strategy is required, every other field defaults."""
    cfg = LossConfig(strategy="cross_entropy")
    assert cfg.strategy == "cross_entropy"
    assert cfg.focal_gamma == 2.0
    assert cfg.focal_alpha is None
    assert cfg.huber_delta == 1.0
    assert cfg.label_smoothing == 0.0
    assert cfg.extra == ()


def test_reserved_keys_collision_raises() -> None:
    with pytest.raises(ValidationError, match=r"focal_gamma"):
        LossConfig(strategy="focal", extra=(("focal_gamma", 2.5),))


@pytest.mark.parametrize("strategy", ["mse", "mae", "pinball"])
def test_empty_reserved_set_strategies_pass(strategy: str) -> None:
    """Exercises the _RESERVED_BY_LOSS keys whose reserved set is empty."""
    cfg = LossConfig(strategy=strategy)  # type: ignore[arg-type]
    assert cfg.strategy == strategy


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        LossConfig(strategy="cross_entropy", undocumented_field=1)  # type: ignore[call-arg]
