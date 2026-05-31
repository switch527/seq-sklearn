"""Combined hardware-detect + precision-resolution contract (A11 / N1).

Phase 1's ``test_hardware_detect.py`` covers detection in isolation;
this test covers the combined contract per A11 / N5:
parametrized over the six tiers, each row mocks the exact
``torch.cuda`` chain ``detect()`` reads, asserts the returned
``HardwareTier``, then calls ``resolve_precision(tier, "auto")`` and
asserts the concrete precision. The explicit pass-through cases are
covered separately.
"""

from unittest.mock import patch

import pytest

from seq_sklearn.hardware import HardwareTier, detect
from seq_sklearn.training._precision import (
    RequestedPrecision,
    ResolvedPrecision,
    resolve_precision,
)


@pytest.mark.parametrize(
    ("cuda_available", "cc_major", "expected_tier", "expected_auto"),
    [
        (False, None, HardwareTier.CPU, "32-true"),
        (True, 6, HardwareTier.PASCAL, "32-true"),
        (True, 7, HardwareTier.VOLTA_TURING, "32-true"),
        (True, 8, HardwareTier.AMPERE_ADA, "bf16-mixed"),
        (True, 9, HardwareTier.HOPPER, "bf16-mixed"),
        (True, 10, HardwareTier.BLACKWELL, "bf16-mixed"),
    ],
)
def test_hardware_detect_and_resolve_precision_combined(
    cuda_available: bool,
    cc_major: int | None,
    expected_tier: HardwareTier,
    expected_auto: ResolvedPrecision,
) -> None:
    """Detection and `auto`-resolution agree per tier in one run."""
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch(
            "torch.cuda.get_device_capability",
            return_value=(cc_major, 0),
        ),
    ):
        tier = detect()

    assert tier is expected_tier
    assert resolve_precision(tier, "auto") == expected_auto


@pytest.mark.parametrize(
    "tier",
    list(HardwareTier),
)
@pytest.mark.parametrize(
    "requested",
    ["bf16-mixed", "16-mixed", "32-true"],
)
def test_explicit_request_passes_through_every_tier(
    tier: HardwareTier,
    requested: RequestedPrecision,
) -> None:
    """An explicit precision is returned unchanged regardless of tier."""
    assert resolve_precision(tier, requested) == requested
