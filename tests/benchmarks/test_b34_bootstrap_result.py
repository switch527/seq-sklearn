"""Phase B34 BootstrapResult dataclass tests.

Closes D-B21.4: pin the BootstrapResult schema + the
backward-compat tuple-unpack shim that lets existing callers
keep working alongside the new attribute access.
"""

import numpy as np
import pytest
from benchmarks.metrics.bootstrap import (
    BootstrapResult,
    entity_block_bootstrap_ci,
)
from pydantic import ValidationError


def test_bootstrap_result_accepts_required_fields() -> None:
    result = BootstrapResult(mean=0.5, ci_lo=0.4, ci_hi=0.6, fallback_reason=None)
    assert result.mean == 0.5
    assert result.ci_lo == 0.4
    assert result.ci_hi == 0.6
    assert result.fallback_reason is None


def test_bootstrap_result_fallback_reason_defaults_to_none() -> None:
    """fallback_reason has a default of None so callers
    constructing happy-path results can omit it."""
    result = BootstrapResult(mean=0.5, ci_lo=0.4, ci_hi=0.6)
    assert result.fallback_reason is None


def test_bootstrap_result_rejects_extra_field() -> None:
    with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
        BootstrapResult(mean=0.5, ci_lo=0.4, ci_hi=0.6, fallback_reason=None, extra="oops")  # type: ignore[call-arg]


def test_bootstrap_result_is_frozen() -> None:
    """frozen=True config: any attempt to mutate a field after
    construction raises."""
    result = BootstrapResult(mean=0.5, ci_lo=0.4, ci_hi=0.6, fallback_reason=None)
    with pytest.raises(ValidationError):
        result.mean = 999.0  # type: ignore[misc]


def test_bootstrap_result_tuple_unpack_yields_4_fields_in_position_order() -> None:
    """B34 / D-B21.4 backward-compat shim: __iter__ yields the
    fields in original tuple-position order (mean, ci_lo,
    ci_hi, fallback_reason) so existing callers that did
    `m, lo, hi, fr = entity_block_bootstrap_ci(...)` keep
    working without migration."""
    result = BootstrapResult(mean=0.5, ci_lo=0.4, ci_hi=0.6, fallback_reason="a_overshoot")
    mean, ci_lo, ci_hi, fallback = result
    assert mean == 0.5
    assert ci_lo == 0.4
    assert ci_hi == 0.6
    assert fallback == "a_overshoot"
    assert len(result) == 4


def test_entity_block_bootstrap_ci_returns_bootstrap_result_instance() -> None:
    """The primitive's return is a BootstrapResult, not a
    raw tuple."""
    losses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    entities = np.array([0, 0, 1, 1])
    result = entity_block_bootstrap_ci(losses, entities, n_resamples=100, seed=42)
    assert isinstance(result, BootstrapResult)
    # And tuple-unpack still works via the __iter__ shim.
    mean, ci_lo, ci_hi, _fallback = result
    assert isinstance(mean, float)
    assert isinstance(ci_lo, float)
    assert isinstance(ci_hi, float)
