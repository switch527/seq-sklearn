"""Hardware-tier detection (per architecture A11 / requirements N5).

:func:`detect` is a pure function that classifies the runtime into one of
six tiers based on ``torch.cuda``'s compute-capability reading. The
precision-resolution step (``auto`` to concrete precision literal) lives
in :mod:`seq_sklearn.training._precision` so the two functions are
independently unit-testable; N1's parametrized test exercises both in
sequence per tier.
"""

import warnings
from enum import IntEnum

import torch

__all__ = ["HardwareTier", "detect"]

_pre_pascal_warned: bool = False


class HardwareTier(IntEnum):
    """NVIDIA compute-capability tiers seq-sklearn dispatches on.

    Ordering matches the historical CC progression; ``IntEnum`` means
    ``HardwareTier.AMPERE_ADA >= HardwareTier.VOLTA_TURING`` compares
    naturally. All six tiers are populated in v1 even though v1 branches
    only on the first four; v2's FP8 path will dispatch on ``HOPPER`` /
    ``BLACKWELL`` without an API change (N5).
    """

    CPU = 0
    PASCAL = 1
    VOLTA_TURING = 2
    AMPERE_ADA = 3
    HOPPER = 4
    BLACKWELL = 5


def detect() -> HardwareTier:
    """Return the current runtime's hardware tier.

    The detection sequence is deliberately narrow (it is the exact chain
    the N1 parametrized hardware-detect test mocks):

    1. If CUDA is not available, return :attr:`HardwareTier.CPU`.
    2. Read ``torch.cuda.get_device_capability(0)`` and dispatch on the
       major version:

       * ``6`` → :attr:`HardwareTier.PASCAL`
       * ``7`` → :attr:`HardwareTier.VOLTA_TURING`
       * ``8`` → :attr:`HardwareTier.AMPERE_ADA`
       * ``9`` → :attr:`HardwareTier.HOPPER`
       * ``>= 10`` → :attr:`HardwareTier.BLACKWELL`
       * anything else → :attr:`HardwareTier.CPU`

    Does NOT call ``torch.cuda.device_count`` or
    ``torch.cuda.current_device``; the test mocks rely on this contract.
    """
    if not torch.cuda.is_available():
        return HardwareTier.CPU
    major, _ = torch.cuda.get_device_capability(0)
    if major == 6:
        return HardwareTier.PASCAL
    if major == 7:
        return HardwareTier.VOLTA_TURING
    if major == 8:
        return HardwareTier.AMPERE_ADA
    if major == 9:
        return HardwareTier.HOPPER
    if major >= 10:
        return HardwareTier.BLACKWELL
    global _pre_pascal_warned
    if not _pre_pascal_warned:
        warnings.warn(
            f"Unsupported CUDA compute capability {major}.x detected; "
            "falling back to CPU. Pascal (6.x) and newer are supported "
            "per N5. Your GPU will be ignored for the lifetime of this "
            "process.",
            UserWarning,
            stacklevel=2,
        )
        _pre_pascal_warned = True
    return HardwareTier.CPU
