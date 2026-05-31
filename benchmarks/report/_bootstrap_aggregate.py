"""Shared bootstrap-CI aggregate constants + helpers (Phase B14 extraction).

Houses the profile-default `n_resamples` dispatch, the OOM row-
count ceiling, the fixed seed + confidence, and the numpy-version
sniffer. Hoisted from `benchmarks/report/bootstrap_rollup.py` by
B14 so the three rollup aggregators (B5, B6 pairwise, B7 training-
time) share one source of truth.

Package-internal (`_` prefix): consumed only by the three
`bootstrap_*.py` aggregators under `benchmarks/report/`.
"""

from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Literal

import numpy as np

from benchmarks.metrics.bootstrap import entity_block_bootstrap_ci

if TYPE_CHECKING:
    from benchmarks.bootstrap_manifest import FoldCI

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
# B22 / D-B16.3: XOR seed offset for per-fold bootstrap
# invocations. Combined with `fold_index` to produce
# `base_seed ^ BOOTSTRAP_PER_FOLD_SEED_OFFSET ^ fold_index`
# so each fold's PCG64 stream is independent from both the
# pooled stream AND every other fold's stream. The XOR
# distinctness invariant holds for any two non-equal
# non-negative fold_indices (`i XOR j != 0` for `i != j`);
# the pooled-vs-per-fold distinctness requires this constant
# to be non-zero. Resetting this to 0 would silently collapse
# `seed(fold_index=0)` onto `BOOTSTRAP_DEFAULT_SEED` and
# re-correlate the pooled + fold-0 streams.
BOOTSTRAP_PER_FOLD_SEED_OFFSET: int = 0xB22_F01D_C1
# B35 / D-B22.2: XOR seed offset for the per-fold ORACLE
# bootstrap on EnsembleLiftRollupRow. Independent from
# BOOTSTRAP_ORACLE_SEED_OFFSET (pooled oracle) and
# BOOTSTRAP_PER_FOLD_SEED_OFFSET (main per-fold) so the
# three streams stay uncorrelated when they happen to
# share entity indices.
BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET: int = 0xB35_07A_FCD
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


def spec_has_per_fold_cis_enabled(experiments: Iterable[ExperimentSpec], *, kind: str) -> bool:
    """True iff any `ExperimentSpec(kind=...)` has
    `bootstrap_per_fold_cis_enabled=True` (B22 / D-B16.3).

    Each aggregator passes its own `kind` so the per-fold opt-in
    is consulted only for the matching experiment kind.
    """
    return any(spec.kind == kind and spec.bootstrap_per_fold_cis_enabled for spec in experiments)


def compute_per_fold_cis(
    *,
    losses: np.ndarray,
    entities: np.ndarray,
    fold_indices: np.ndarray,
    seeds: np.ndarray,
    n_resamples: int,
    confidence: float,
    base_seed: int,
    fold_seed_offset: int,
    ci_method: Literal["percentile", "bca"],
    metric_fn: Callable[[np.ndarray], float],
    dataset_name: str,
    kind: str,
) -> "list[FoldCI]":
    """Bootstrap CI per fold (B22 / D-B16.3).

    Groups the input row arrays by `fold_indices`, then bootstraps
    each fold independently using
    `seed = base_seed ^ fold_seed_offset ^ fold_index` so each
    fold's PCG64 stream is independent from the pooled stream AND
    from every other fold's stream. Returns one `FoldCI` per fold
    with rows in the input, sorted ascending by `fold_index`.
    Folds with zero rows are omitted from the result.

    Args:
        losses: 1-D per-row loss array (parallel to entities +
            fold_indices + seeds).
        entities: 1-D per-row entity ids (used by the primitive's
            entity-block resample).
        fold_indices: 1-D per-row `fold_index` values; the
            grouping key.
        seeds: 1-D per-row `seed` values; used to compute
            `n_seeds` per FoldCI.
        n_resamples, confidence, base_seed, fold_seed_offset,
        ci_method, metric_fn: forwarded to
        `entity_block_bootstrap_ci` per fold.
        dataset_name, kind: surfaced in the error message when
            the per-fold OOM gate fires.

    Raises:
        RawRollupError: a fold's n_rows * n_resamples exceeds
            `BOOTSTRAP_ROW_COUNT_CEILING`. R-B22-6 contract.
    """
    # Imported lazily to avoid a module-load cycle: bootstrap_rollup
    # imports this module AND defines RawRollupError; this helper
    # imports the error type at call time.
    from benchmarks.bootstrap_manifest import FoldCI
    from benchmarks.report.bootstrap_rollup import RawRollupError

    fold_cis: list[FoldCI] = []
    unique_folds = sorted({int(f) for f in fold_indices})
    for fold_index in unique_folds:
        mask = fold_indices == fold_index
        fold_losses = losses[mask]
        fold_entities = entities[mask]
        fold_seeds = seeds[mask]
        if fold_losses.size == 0:
            continue
        if fold_losses.shape[0] * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING:
            raise RawRollupError(
                f"aggregate_bootstrap_{kind}_rollup: dataset="
                f"{dataset_name!r} fold_index={fold_index} with "
                f"n_rows={fold_losses.shape[0]} * n_resamples="
                f"{n_resamples} exceeds the bootstrap-row-count "
                f"ceiling ({BOOTSTRAP_ROW_COUNT_CEILING}) on the "
                "per-fold CI"
            )
        n_seeds_in_fold = int(np.unique(fold_seeds).shape[0])
        n_unique_entities = int(np.unique(fold_entities).shape[0])
        seed = base_seed ^ fold_seed_offset ^ fold_index
        mean, ci_lo, ci_hi, fallback_reason = entity_block_bootstrap_ci(
            fold_losses,
            fold_entities,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
            metric_fn=metric_fn,
            ci_method=ci_method,
        )
        fold_cis.append(
            FoldCI(
                fold_index=fold_index,
                n_seeds=n_seeds_in_fold,
                n_entities=n_unique_entities,
                metric_mean=mean,
                metric_ci_lo=ci_lo,
                metric_ci_hi=ci_hi,
                ci_method=ci_method,
                ci_fallback_reason=fallback_reason,
            )
        )
    return fold_cis


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_DEFAULT_CI_METHOD",
    "BOOTSTRAP_DEFAULT_SEED",
    "BOOTSTRAP_N_RESAMPLES_BY_PROFILE",
    "BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET",
    "BOOTSTRAP_ORACLE_SEED_OFFSET",
    "BOOTSTRAP_PER_FOLD_SEED_OFFSET",
    "BOOTSTRAP_ROW_COUNT_CEILING",
    "compute_per_fold_cis",
    "numpy_version",
    "resolve_n_resamples",
    "spec_has_per_fold_cis_enabled",
]
