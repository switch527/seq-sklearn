"""Precision resolution (per architecture A11 / requirements N5).

The ``(tier, requested)`` to concrete-precision mapping lives here, not
in :func:`seq_sklearn.hardware.detect`, so detection and dispatch are
independently unit-testable. N1's parametrized hardware-detect test
exercises both in sequence per tier.
"""

import logging
from typing import Literal

from seq_sklearn.hardware import HardwareTier

__all__ = ["resolve_precision"]

logger = logging.getLogger(__name__)

RequestedPrecision = Literal["bf16-mixed", "16-mixed", "32-true", "auto"]
ResolvedPrecision = Literal["bf16-mixed", "16-mixed", "32-true"]

# CC>=8.0 (Ampere and newer) gets bf16-mixed under `auto`; everything
# else gets 32-true. bf16 has fp32's exponent range so no GradScaler is
# needed; the TFT quantile-loss + softmax fp16 NaN history rules out
# `16-mixed` ever being picked by `auto` (it stays explicit opt-in).
_BF16_TIERS: frozenset[HardwareTier] = frozenset(
    {HardwareTier.AMPERE_ADA, HardwareTier.HOPPER, HardwareTier.BLACKWELL}
)


def resolve_precision(
    tier: HardwareTier,
    requested: RequestedPrecision,
) -> ResolvedPrecision:
    """Map ``(tier, requested)`` to a concrete Lightning precision literal.

    An explicit request (``bf16-mixed`` / ``16-mixed`` / ``32-true``)
    passes through unchanged; the caller owns the divergence risk of an
    explicit ``16-mixed``. Under ``auto`` the N5 table applies:
    ``bf16-mixed`` on CC>=8.0 (Ampere / Ada / Hopper / Blackwell),
    ``32-true`` on Volta / Turing / Pascal / CPU. ``16-mixed`` is never
    selected by ``auto``.
    """
    if requested != "auto":
        return requested
    if tier in _BF16_TIERS:
        return "bf16-mixed"
    return "32-true"
