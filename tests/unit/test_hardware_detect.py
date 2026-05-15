"""Tests for hardware-tier detection at architecture A11 / requirements N1.

One of N1's required tests. Parametrized over every documented tier (CPU
plus the five GPU tiers), each case mocks ``torch.cuda.is_available`` and
``torch.cuda.get_device_capability`` and asserts the returned tier. The
combined ``resolve_precision`` half of N1's parametrized matrix lands in
Phase 4 once the precision module exists.
"""

from unittest.mock import patch

import pytest

from seq_sklearn.hardware import HardwareTier, detect


def test_detect_returns_cpu_when_cuda_unavailable() -> None:
    with patch("seq_sklearn.hardware.torch.cuda.is_available", return_value=False):
        assert detect() is HardwareTier.CPU


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [
        ((6, 0), HardwareTier.PASCAL),
        ((6, 1), HardwareTier.PASCAL),
        ((7, 0), HardwareTier.VOLTA_TURING),
        ((7, 5), HardwareTier.VOLTA_TURING),
        ((8, 0), HardwareTier.AMPERE_ADA),
        ((8, 6), HardwareTier.AMPERE_ADA),
        ((8, 9), HardwareTier.AMPERE_ADA),
        ((9, 0), HardwareTier.HOPPER),
        ((10, 0), HardwareTier.BLACKWELL),
        ((11, 0), HardwareTier.BLACKWELL),
    ],
)
def test_detect_maps_compute_capability_to_tier(
    compute_capability: tuple[int, int],
    expected: HardwareTier,
) -> None:
    with (
        patch("seq_sklearn.hardware.torch.cuda.is_available", return_value=True),
        patch(
            "seq_sklearn.hardware.torch.cuda.get_device_capability",
            return_value=compute_capability,
        ),
    ):
        assert detect() is expected


def test_detect_pre_pascal_falls_back_to_cpu_with_warning() -> None:
    """CC < 6.0 is unsupported per N5; detection returns CPU and warns once.

    A11 specifies "else -> CPU (unsupported, warn once)". The module-level
    `_pre_pascal_warned` latch makes the warning fire only on the first
    pre-Pascal detection within a process; this test resets the latch so
    the assertion is deterministic.
    """
    import seq_sklearn.hardware as hw_mod

    hw_mod._pre_pascal_warned = False
    with (
        patch("seq_sklearn.hardware.torch.cuda.is_available", return_value=True),
        patch(
            "seq_sklearn.hardware.torch.cuda.get_device_capability",
            return_value=(5, 2),
        ),
        pytest.warns(UserWarning, match="Unsupported CUDA compute capability"),
    ):
        assert detect() is HardwareTier.CPU


def test_detect_pre_pascal_warning_fires_only_once() -> None:
    """The module-level latch suppresses repeated warnings for the same process."""
    import warnings as _warnings

    import seq_sklearn.hardware as hw_mod

    hw_mod._pre_pascal_warned = False
    with (
        patch("seq_sklearn.hardware.torch.cuda.is_available", return_value=True),
        patch(
            "seq_sklearn.hardware.torch.cuda.get_device_capability",
            return_value=(5, 2),
        ),
    ):
        detect()  # first call warns
        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            detect()  # second call should not warn
            assert not any(
                "Unsupported CUDA compute capability" in str(w.message) for w in captured
            )


def test_hardware_tier_intenum_ordering() -> None:
    """IntEnum ordering must reflect the CC progression."""
    assert HardwareTier.CPU < HardwareTier.PASCAL
    assert HardwareTier.PASCAL < HardwareTier.VOLTA_TURING
    assert HardwareTier.VOLTA_TURING < HardwareTier.AMPERE_ADA
    assert HardwareTier.AMPERE_ADA < HardwareTier.HOPPER
    assert HardwareTier.HOPPER < HardwareTier.BLACKWELL


def test_hardware_tier_count_matches_documented_six() -> None:
    assert len(HardwareTier) == 6
