"""Shared bootstrap-CI aggregate constants + helpers (Phase B14 extraction).

Houses the profile-default `n_resamples` dispatch, the OOM row-
count ceiling, the fixed seed + confidence, and the numpy-version
sniffer. Hoisted from `benchmarks/report/bootstrap_rollup.py` by
B14 so the three rollup aggregators (B5, B6 pairwise, B7 training-
time) share one source of truth.

Package-internal (`_` prefix): consumed only by the three
`bootstrap_*.py` aggregators under `benchmarks/report/`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

BOOTSTRAP_N_RESAMPLES_BY_PROFILE: dict[str, int] = {
    "smoke": 5_000,
    "standard": 10_000,
    "full": 10_000,
}
BOOTSTRAP_DEFAULT_SEED: int = 0xB13_5EED_B007
BOOTSTRAP_CONFIDENCE: float = 0.95

# Defensive row-count ceiling (R-B13-3 / Gemini-C2).
# N * n_resamples > 5e10 is the rough OOM threshold for a naive
# per-resample row-concat path; D-B13.7 names the sufficient-
# statistics optimization for the followup.
BOOTSTRAP_ROW_COUNT_CEILING: int = 50_000_000_000


def numpy_version() -> str:
    """Resolve numpy's installed PEP 440 version string.

    Falls back to `"unknown"` if `importlib.metadata` can't find
    the distribution (defensive; never raises).
    """
    try:
        return _pkg_version("numpy")
    except PackageNotFoundError:
        return "unknown"


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_DEFAULT_SEED",
    "BOOTSTRAP_N_RESAMPLES_BY_PROFILE",
    "BOOTSTRAP_ROW_COUNT_CEILING",
    "numpy_version",
]
