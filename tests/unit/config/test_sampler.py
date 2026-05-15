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


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SamplerConfig(undocumented_field=1)  # type: ignore[call-arg]
