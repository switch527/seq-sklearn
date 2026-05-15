"""Tests for the Tier 1 SchedulerConfig sub-config."""

import pytest
from pydantic import ValidationError

from seq_sklearn.config.scheduler import SchedulerConfig


def test_default_construction_uses_documented_defaults() -> None:
    cfg = SchedulerConfig()
    assert cfg.name == "cosine_with_warmup"
    assert cfg.warmup_steps == 100
    assert cfg.pct_start == 0.3
    assert cfg.div_factor == 25.0
    assert cfg.final_div_factor == 1e4
    assert cfg.plateau_factor == 0.5
    assert cfg.plateau_patience == 5
    assert cfg.plateau_threshold == 1e-4
    assert cfg.min_lr == 0.0
    assert cfg.extra == ()


def test_reserved_keys_collision_raises() -> None:
    with pytest.raises(ValidationError, match=r"warmup_steps"):
        SchedulerConfig(name="cosine_with_warmup", extra=(("warmup_steps", 50),))


def test_constant_scheduler_construction_succeeds() -> None:
    """Exercises the _RESERVED_BY_SCHEDULER["constant"] (empty) key and
    the no-clash pass branch for a non-default scheduler name."""
    cfg = SchedulerConfig(name="constant")
    assert cfg.name == "constant"
    assert cfg.extra == ()


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig(undocumented_field=1)  # type: ignore[call-arg]
