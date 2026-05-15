"""Tests for the Tier 1 LossConfig sub-config."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.loss import LossConfig


def test_construction_requires_strategy() -> None:
    with pytest.raises(ValidationError):
        LossConfig()  # type: ignore[call-arg]


def test_valid_construction_uses_documented_defaults() -> None:
    """Happy path: strategy is required, the rest default per the schema.

    Not in the original manifest's 3-test loss set (all failure paths);
    added to hold 100% coverage on the validator's pass branch and to
    satisfy the testing.md per-config happy-path rule, paralleling the
    default-construction tests on the other three families.
    """
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


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        LossConfig(strategy="cross_entropy", undocumented_field=1)  # type: ignore[call-arg]
