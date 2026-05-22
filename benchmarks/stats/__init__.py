"""Benchmark-level statistical tests (Phase B8).

The `friedman` module ships the matrix-level Friedman + Holm
correction the HPO-uplift report consumes (qa-I1, design B7.5).
"""

from benchmarks.stats.friedman import (
    FriedmanHolmResult,
    HolmAdjustedPValue,
    friedman_holm,
    holm_correction,
)

__all__ = [
    "FriedmanHolmResult",
    "HolmAdjustedPValue",
    "friedman_holm",
    "holm_correction",
]
