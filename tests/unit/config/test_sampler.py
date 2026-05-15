"""Tests for the Tier 1 SamplerConfig sub-config."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.sampler import SamplerConfig


def test_default_strategy_is_none() -> None:
    cfg = SamplerConfig()
    assert cfg.strategy == "none"
    assert cfg.oversample_ratio == 1.0
    assert cfg.replacement is True
    assert cfg.extra == ()


def test_reserved_keys_collision_raises() -> None:
    with pytest.raises(ValidationError, match=r"oversample_ratio"):
        SamplerConfig(strategy="oversample_minority", extra=(("oversample_ratio", 2.0),))


def test_class_weighted_strategy_passes_with_extra() -> None:
    """Exercises the empty _RESERVED_BY_SAMPLER["class_weighted"] key."""
    cfg = SamplerConfig(strategy="class_weighted", extra=(("custom_knob", 1.0),))
    assert cfg.strategy == "class_weighted"
    assert dict(cfg.extra) == {"custom_knob": 1.0}


def test_undersample_majority_clash_raises() -> None:
    """Exercises the _RESERVED_BY_SAMPLER["undersample_majority"] clash path."""
    with pytest.raises(ValidationError, match=r"replacement"):
        SamplerConfig(strategy="undersample_majority", extra=(("replacement", False),))


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SamplerConfig(undocumented_field=1)  # type: ignore[call-arg]
