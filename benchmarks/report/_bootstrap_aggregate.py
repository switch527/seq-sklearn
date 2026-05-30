"""Shared bootstrap-CI aggregate constants + helpers (Phase B14 extraction).

Houses the profile-default `n_resamples` dispatch, the OOM row-
count ceiling, the fixed seed + confidence, and the numpy-version
sniffer. Hoisted from `benchmarks/report/bootstrap_rollup.py` by
B14 so the three rollup aggregators (B5, B6 pairwise, B7 training-
time) share one source of truth.

Package-internal (`_` prefix): consumed only by the three
`bootstrap_*.py` aggregators under `benchmarks/report/`.
"""

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Literal

from benchmarks.config import ExperimentSpec

BOOTSTRAP_N_RESAMPLES_BY_PROFILE: dict[str, int] = {
    "smoke": 5_000,
    "standard": 10_000,
    "full": 10_000,
}
BOOTSTRAP_DEFAULT_SEED: int = 0xB13_5EED_B007
# B20 / D-B16.1: XOR mask used to derive an independent PCG64 stream
# for the oracle delta bootstrap in `bootstrap_ensemble_lift`. The
# main delta_loss bootstrap uses BOOTSTRAP_DEFAULT_SEED; the oracle
# bootstrap uses BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET
# so the two streams (which can operate on overlapping cell indices)
# produce uncorrelated resample sequences. `benchmarks/metrics/
# bootstrap.py` constructs `np.random.Generator(np.random.PCG64(seed))`
# where the stream depends ONLY on `seed`; sharing a seed would
# yield identical resample indices on shared `n_entities`.
BOOTSTRAP_ORACLE_SEED_OFFSET: int = 0xB20_07A_C7E
# B21 / D-B16.2: default CI method for all 5 aggregators.
# Aggregators read this via late-binding lookup so tests can
# monkeypatch the canonical source module to verify the seam
# end-to-end (B21 test #12). v1 default is BCa; the percentile
# fallback is opt-in via the primitive's `ci_method` kwarg.
BOOTSTRAP_DEFAULT_CI_METHOD: Literal["percentile", "bca"] = "bca"
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


def resolve_n_resamples(
    experiments: Iterable[ExperimentSpec],
    profile: str,
    *,
    kind: str,
) -> int:
    """Priority: per-spec override > profile default.

    Reads any `ExperimentSpec` whose `kind` matches and returns
    that spec's `bootstrap_n_resamples` if non-None. Otherwise
    falls back to the profile default (defensive `"standard"`
    fallback if the profile is unknown).

    Stage-3 code-I1 closure: a single helper for the three
    rollup aggregators (B5, B6, B7) so the per-kind override
    logic lives in one place.
    """
    for spec in experiments:
        if spec.kind == kind and spec.bootstrap_n_resamples is not None:
            return spec.bootstrap_n_resamples
    return BOOTSTRAP_N_RESAMPLES_BY_PROFILE.get(profile, 10_000)


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_DEFAULT_CI_METHOD",
    "BOOTSTRAP_DEFAULT_SEED",
    "BOOTSTRAP_N_RESAMPLES_BY_PROFILE",
    "BOOTSTRAP_ORACLE_SEED_OFFSET",
    "BOOTSTRAP_ROW_COUNT_CEILING",
    "numpy_version",
    "resolve_n_resamples",
]
