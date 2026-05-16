"""Tests for the Tier 1 OptimizerConfig sub-config."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.optimizer import OptimizerConfig


def test_default_construction_uses_documented_defaults() -> None:
    cfg = OptimizerConfig()
    assert cfg.name == "adamw"
    assert cfg.learning_rate == 1e-3
    assert cfg.weight_decay == 1e-4
    assert cfg.betas == (0.9, 0.999)
    assert cfg.eps == 1e-8
    assert cfg.momentum == 0.9
    assert cfg.nesterov is False
    assert cfg.extra == ()


def test_adamw_reserved_keys_collision_raises() -> None:
    with pytest.raises(ValidationError, match=r"lr"):
        OptimizerConfig(name="adamw", extra=(("lr", 0.1),))


def test_sgd_reserved_keys_collision_raises() -> None:
    with pytest.raises(ValidationError, match=r"momentum"):
        OptimizerConfig(name="sgd", extra=(("momentum", 0.5),))


def test_adam_uses_its_own_reserved_set() -> None:
    """Exercises the _RESERVED_BY_OPTIMIZER["adam"] dict key (distinct
    from "adamw"/"sgd" even though the reserved set overlaps)."""
    with pytest.raises(ValidationError, match=r"lr"):
        OptimizerConfig(name="adam", extra=(("lr", 0.1),))


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        OptimizerConfig(undocumented_field=1)  # type: ignore[call-arg]
