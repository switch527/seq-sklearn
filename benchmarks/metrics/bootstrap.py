"""Entity-block bootstrap CI primitive (Phase B13 / B5.4).

Pure function: takes pre-computed per-row losses + per-row entity
ids + `(n_resamples, seed, metric_fn)`, returns
`(mean, ci_lo, ci_hi)`.

Design contracts (`docs/benchmark_suite_phase_log.md` phase
B13; full delta at the B13 merge commit per the log):

- The bootstrap resamples UNIQUE entity ids WITH replacement
  (entity-block bootstrap, NOT row bootstrap). Panel data carries
  intra-entity correlation; row resampling would report CIs
  sqrt(K) tighter than the truth where K is rows per entity. This
  is the gap the B11/B12 Gemini final-passes both flagged.

- For each resample, the resampled-entity rows are concatenated
  and `metric_fn` is applied to the concatenated per-row loss
  vector to produce ONE scalar. Default `metric_fn` is
  `np.nanmean` (classification's log_loss path); regression
  passes `lambda x: float(np.sqrt(np.nanmean(x)))` so sqrt
  applies PER RESAMPLE (closes the Jensen-inequality gap
  Gemini-C1 flagged).

- The RNG is `np.random.Generator(np.random.PCG64(seed))`
  EXPLICITLY (NOT `np.random.default_rng` which may swap
  algorithms in a future NumPy major). The percentile call
  pins `method="linear"`.

- Input arrays are set read-only (`flags.writeable = False`) at
  entry so a `metric_fn` that attempts in-place mutation raises
  immediately rather than silently corrupting subsequent
  resamples.
"""

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm  # B21 / D-B16.2: BCa transform

__all__ = [
    "BootstrapResult",
    "entity_block_bootstrap_ci",
    "entity_block_bootstrap_ci_mean_fast",
]


class BootstrapResult(BaseModel):
    """B34 / D-B21.4: frozen result type for
    `entity_block_bootstrap_ci`. Replaces the prior 4-tuple
    return. Fields preserve the original tuple-position
    semantics: mean, ci_lo, ci_hi, fallback_reason.

    `fallback_reason` is `None` (no fallback fired or
    `ci_method="percentile"`), `"p0_at_edge"`, or
    `"a_overshoot"`.

    Custom `__iter__` yields the 4 fields in tuple-position
    order so existing tuple-unpack callers
    (`mean, ci_lo, ci_hi, fallback = result`) keep working
    alongside the new attribute access (`result.mean`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    ci_lo: float
    ci_hi: float
    fallback_reason: str | None = None

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        # Yield in original tuple-position order so existing
        # callers keep tuple-unpack semantics. Typed `Any` so
        # the per-position float / str | None narrowing
        # survives at the unpack site (each variable retains
        # its expected type at the call site rather than the
        # union of all field types).
        return iter((self.mean, self.ci_lo, self.ci_hi, self.fallback_reason))

    def __len__(self) -> int:
        # Backward-compat with tuple-style `len(result) == 4`
        # assertions in B21 tests.
        return 4


# Default bootstrap parameters per B5.4 + R3 / R4 of the B13 delta.
_DEFAULT_N_RESAMPLES: int = 10_000
_DEFAULT_SEED: int = 0xB13_5EED_B007
_DEFAULT_CONFIDENCE: float = 0.95
# B21 / D-B16.2: epsilon guard for the BCa acceleration transform's
# denominator. Catches `denom <= 0` overshoot AND the near-zero
# finite-precision saturation case. Module-private; consumed by
# `_bca_percentile_points`.
_BCA_DENOM_EPS: float = 1e-12
# RNG-algorithm pin (Gemini-I2 + R-B13-2): explicit so the
# aggregator can record the actual algorithm name on every
# RollupRow without re-deriving it from `type(rng).__name__`.
BOOTSTRAP_RNG_ALGORITHM: str = "PCG64"


def _default_metric_fn(x: np.ndarray) -> float:
    """Classification default: nanmean (log_loss per-row vector)."""
    return float(np.nanmean(x))


def _compute_acceleration_from_jackknife(jackknife: np.ndarray) -> float:
    """Acceleration `a` from a leave-one-entity-out jackknife.

    Returns 0.0 when the denominator is 0 (all jackknife values
    equal); BCa reduces to BC (bias-corrected percentile).
    """
    m_dot = float(np.mean(jackknife))
    deviations = m_dot - jackknife
    num = float(np.sum(deviations**3))
    denom = 6.0 * float(np.sum(deviations**2)) ** 1.5
    return 0.0 if denom == 0.0 else num / denom


def _bca_percentile_points(
    p0: float, a: float, confidence: float
) -> tuple[float, float, str | None]:
    """BCa percentile points given `p0`, `a`, `confidence`.

    Returns `(alpha_1, alpha_2, fallback_reason)`. The caller
    feeds the returned percentile points to `np.percentile`.
    `fallback_reason` is one of `None`, `"p0_at_edge"`,
    `"a_overshoot"`.
    """
    alpha = (1.0 - confidence) / 2.0
    if p0 <= 0.0 or p0 >= 1.0:
        return alpha, 1.0 - alpha, "p0_at_edge"
    z0 = float(norm.ppf(p0))
    z_lo = float(norm.ppf(alpha))
    z_hi = float(norm.ppf(1.0 - alpha))
    denom_lo = 1.0 - a * (z0 + z_lo)
    denom_hi = 1.0 - a * (z0 + z_hi)
    if denom_lo <= _BCA_DENOM_EPS or denom_hi <= _BCA_DENOM_EPS:
        return alpha, 1.0 - alpha, "a_overshoot"
    alpha_1 = float(norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha_2 = float(norm.cdf(z0 + (z0 + z_hi) / denom_hi))
    return alpha_1, alpha_2, None


def _bca_percentiles(
    resampled: np.ndarray,
    ground_truth: float,
    rows_by_entity: list[np.ndarray],
    losses_view: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    confidence: float,
) -> tuple[float, float, str | None]:
    """BCa percentile points orchestrator.

    Computes `p_0` (bias correction proportion) from the resampled
    distribution, the leave-one-entity-out jackknife `a`
    (acceleration), and delegates to `_bca_percentile_points` for
    the BCa transform.
    """
    p0 = float(np.mean(resampled <= ground_truth))
    n_entities = len(rows_by_entity)
    jackknife = np.empty(n_entities, dtype=np.float64)
    for i in range(n_entities):
        leave_out_rows = np.concatenate([rows_by_entity[j] for j in range(n_entities) if j != i])
        jackknife[i] = float(metric_fn(losses_view[leave_out_rows]))
    a = _compute_acceleration_from_jackknife(jackknife)
    return _bca_percentile_points(p0, a, confidence)


def entity_block_bootstrap_ci(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    seed: int = _DEFAULT_SEED,
    metric_fn: Callable[[np.ndarray], float] = _default_metric_fn,
    ci_method: Literal["percentile", "bca"] = "bca",
) -> BootstrapResult:
    """Entity-block bootstrap CI (BCa default, percentile opt-in).

    Args:
        losses: 1-D array of per-row losses.
        entity_ids: 1-D array of per-row entity identifiers (same
            length as `losses`). Type is unconstrained; the
            primitive only requires equality testing.
        n_resamples: Number of bootstrap resamples. B5.4 default
            is 10_000; the smoke profile may halve it via the
            aggregator's profile dispatch.
        confidence: Two-sided percentile interval width. Default
            0.95 → percentiles at 2.5 / 97.5.
        seed: PCG64 seed; pinned algorithm per Gemini-I2.
        metric_fn: Aggregates the concatenated per-row loss
            vector inside each resample to ONE scalar. Default is
            `np.nanmean` (classification). For RMSE, pass
            `lambda x: float(np.sqrt(np.nanmean(x)))` so sqrt is
            applied PER RESAMPLE (per arch-C1 / qa-C2 Jensen fix).
        ci_method: `"bca"` (default) for the bias-corrected and
            accelerated method (Efron & Tibshirani 1993, §14.3);
            `"percentile"` for the simple percentile method. BCa
            falls back to percentile on degenerate cases (p_0 at
            edge or acceleration overshoot); the `fallback_reason`
            element of the return surfaces which path fired.

    Returns:
        `BootstrapResult(mean, ci_lo, ci_hi, fallback_reason)`
        where `mean` is `metric_fn` applied to the
        unresampled loss vector, the CI is the percentile
        or BCa interval over the `n_resamples` resampled
        `metric_fn` values, and `fallback_reason` is `None`
        (no fallback fired or `ci_method="percentile"`),
        `"p0_at_edge"`, or `"a_overshoot"`.

    Raises:
        ValueError: shapes mismatch, confidence outside (0, 1),
            or n_resamples < 1.
    """
    if losses.ndim != 1 or entity_ids.ndim != 1:
        raise ValueError(
            f"entity_block_bootstrap_ci: losses + entity_ids must be 1-D; "
            f"got losses.ndim={losses.ndim}, entity_ids.ndim={entity_ids.ndim}"
        )
    if losses.shape[0] != entity_ids.shape[0]:
        raise ValueError(
            f"entity_block_bootstrap_ci: losses len {losses.shape[0]} != "
            f"entity_ids len {entity_ids.shape[0]}"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"entity_block_bootstrap_ci: confidence must be in (0, 1); got {confidence}"
        )
    if n_resamples < 1:
        raise ValueError(f"entity_block_bootstrap_ci: n_resamples must be >= 1; got {n_resamples}")

    # Defensive read-only flag so a misbehaving `metric_fn` raises
    # immediately on in-place mutation rather than silently
    # corrupting subsequent resamples.
    losses_view = np.asarray(losses)
    entity_ids_view = np.asarray(entity_ids)
    losses_view.flags.writeable = False
    entity_ids_view.flags.writeable = False

    # Ground-truth mean (unresampled).
    ground_truth_mean = float(metric_fn(losses_view))

    # Index entity → rows. Bootstrap resamples ENTITIES with
    # replacement; for each resample, concatenate the rows of the
    # resampled entities and apply metric_fn.
    unique_entities, inverse = np.unique(entity_ids_view, return_inverse=True)
    n_entities = unique_entities.shape[0]
    # `rows_by_entity[i]` is the row positions belonging to entity i.
    rows_by_entity: list[np.ndarray] = [np.where(inverse == i)[0] for i in range(n_entities)]

    if n_entities <= 1:
        # Degenerate: only one entity → every resample yields the
        # same scalar; CI collapses to the ground-truth mean.
        return BootstrapResult(
            mean=ground_truth_mean,
            ci_lo=ground_truth_mean,
            ci_hi=ground_truth_mean,
            fallback_reason=None,
        )

    rng = np.random.Generator(np.random.PCG64(seed))
    resampled = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        # Sample n_entities entity ids with replacement, gather rows.
        picked = rng.integers(0, n_entities, size=n_entities)
        # `np.concatenate` over a list of int arrays is fine for the
        # naive path; the D-B13.7 sufficient-statistics optimization
        # is reserved for the full-tier dataset followup.
        row_indices = np.concatenate([rows_by_entity[p] for p in picked])
        resampled[r] = float(metric_fn(losses_view[row_indices]))

    alpha = (1.0 - confidence) / 2.0
    if ci_method == "percentile":
        ci_lo = float(np.percentile(resampled, 100.0 * alpha, method="linear"))
        ci_hi = float(np.percentile(resampled, 100.0 * (1.0 - alpha), method="linear"))
        return BootstrapResult(
            mean=ground_truth_mean,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            fallback_reason=None,
        )

    # ci_method == "bca"
    alpha_1, alpha_2, fallback_reason = _bca_percentiles(
        resampled=resampled,
        ground_truth=ground_truth_mean,
        rows_by_entity=rows_by_entity,
        losses_view=losses_view,
        metric_fn=metric_fn,
        confidence=confidence,
    )
    ci_lo = float(np.percentile(resampled, 100.0 * alpha_1, method="linear"))
    ci_hi = float(np.percentile(resampled, 100.0 * alpha_2, method="linear"))
    return BootstrapResult(
        mean=ground_truth_mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        fallback_reason=fallback_reason,
    )


def entity_block_bootstrap_ci_mean_fast(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    seed: int = _DEFAULT_SEED,
    ci_method: Literal["percentile", "bca"] = "bca",
    sqrt_mean: bool = False,
) -> BootstrapResult:
    """Sufficient-statistics fast path for the entity-block bootstrap.

    Hardwired to `nanmean` semantics (or `sqrt(nanmean(x))` when
    `sqrt_mean=True`). Numerically equivalent to
    `entity_block_bootstrap_ci(..., metric_fn=np.nanmean)` (or the
    sqrt variant) modulo float-order drift.

    The naive path takes O(N) memory traffic per resample
    (row-vector concatenation); this fast path uses pre-computed
    per-entity (sum_loss, count_non_nan) statistics, reducing
    memory traffic to O(E) per resample. For the full-tier
    Amex dataset (~500k entities vs 6M rows) this is a ~12x
    memory + ~12x throughput improvement (D-B13.7).

    Use cases:
        - v1 classification aggregators with metric_fn=np.nanmean.
        - v1 regression aggregators with sqrt_mean=True.

    A custom metric_fn that is NOT expressible from sum/count
    sufficient statistics (e.g., median, ROC-AUC) must continue
    to use `entity_block_bootstrap_ci`.

    Args:
        losses: 1-D array of per-row losses. NaN rows are treated
            as missing (matches `np.nanmean` semantics).
        entity_ids: 1-D array of per-row entity identifiers (same
            length as `losses`).
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided percentile interval width.
        seed: PCG64 seed; pinned algorithm per Gemini-I2.
        ci_method: `"bca"` (default) or `"percentile"`. BCa uses
            a leave-one-entity-out sufficient-stats jackknife
            (O(1) per leave-one-out) rather than re-applying
            metric_fn to concatenated leave-out rows.
        sqrt_mean: When True, applies `sqrt` to each per-resample
            mean (matches the regression metric_fn
            `lambda x: float(np.sqrt(np.nanmean(x)))`). Sqrt is
            applied PER RESAMPLE per the Jensen-inequality fix.

    Returns:
        `BootstrapResult(mean, ci_lo, ci_hi, fallback_reason)`.

    Raises:
        ValueError: shapes mismatch, confidence outside (0, 1),
            n_resamples < 1, or all losses are NaN.
    """
    if losses.ndim != 1 or entity_ids.ndim != 1:
        raise ValueError(
            f"entity_block_bootstrap_ci_mean_fast: losses + entity_ids must be 1-D; "
            f"got losses.ndim={losses.ndim}, entity_ids.ndim={entity_ids.ndim}"
        )
    if losses.shape[0] != entity_ids.shape[0]:
        raise ValueError(
            f"entity_block_bootstrap_ci_mean_fast: losses len {losses.shape[0]} != "
            f"entity_ids len {entity_ids.shape[0]}"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"entity_block_bootstrap_ci_mean_fast: confidence must be in (0, 1); got {confidence}"
        )
    if n_resamples < 1:
        raise ValueError(
            f"entity_block_bootstrap_ci_mean_fast: n_resamples must be >= 1; got {n_resamples}"
        )

    losses_arr = np.asarray(losses, dtype=np.float64)
    entity_ids_arr = np.asarray(entity_ids)

    # Per-row NaN mask + safe-fill so the sum reduction stays in
    # float64 without NaN poisoning.
    nan_mask = np.isnan(losses_arr)
    safe_losses = np.where(nan_mask, 0.0, losses_arr)
    non_nan_counts = (~nan_mask).astype(np.int64)

    # Group by entity → per-entity (sum, count_non_nan).
    _unique_entities, inverse = np.unique(entity_ids_arr, return_inverse=True)
    n_entities = int(_unique_entities.shape[0])
    entity_sums = np.zeros(n_entities, dtype=np.float64)
    entity_counts = np.zeros(n_entities, dtype=np.int64)
    # `np.add.at` is unbuffered, so duplicate indices accumulate
    # correctly (a single += would only register the last write).
    np.add.at(entity_sums, inverse, safe_losses)
    np.add.at(entity_counts, inverse, non_nan_counts)

    total_sum = float(entity_sums.sum())
    total_count = int(entity_counts.sum())
    if total_count == 0:
        raise ValueError(
            "entity_block_bootstrap_ci_mean_fast: all losses are NaN; "
            "ground-truth mean is undefined"
        )

    ground_truth_mean = total_sum / total_count
    if sqrt_mean:
        ground_truth_mean = float(np.sqrt(ground_truth_mean))

    if n_entities <= 1:
        return BootstrapResult(
            mean=ground_truth_mean,
            ci_lo=ground_truth_mean,
            ci_hi=ground_truth_mean,
            fallback_reason=None,
        )

    # Same RNG sequence as the naive path: PCG64(seed) →
    # `rng.integers(0, n_entities, size=n_entities)` per resample.
    # Identical `picked` arrays guarantee the fast path's
    # resampled distribution matches the naive path's modulo
    # float-order drift in the sum reduction.
    rng = np.random.Generator(np.random.PCG64(seed))
    resampled = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        picked = rng.integers(0, n_entities, size=n_entities)
        sum_picked = float(entity_sums[picked].sum())
        count_picked = int(entity_counts[picked].sum())
        if count_picked == 0:
            resampled[r] = np.nan
            continue
        val = sum_picked / count_picked
        resampled[r] = float(np.sqrt(val)) if sqrt_mean else val

    alpha = (1.0 - confidence) / 2.0
    if ci_method == "percentile":
        ci_lo = float(np.percentile(resampled, 100.0 * alpha, method="linear"))
        ci_hi = float(np.percentile(resampled, 100.0 * (1.0 - alpha), method="linear"))
        return BootstrapResult(
            mean=ground_truth_mean,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            fallback_reason=None,
        )

    # ci_method == "bca": leave-one-entity-out via sufficient stats.
    p0 = float(np.mean(resampled <= ground_truth_mean))
    jackknife = np.empty(n_entities, dtype=np.float64)
    for i in range(n_entities):
        denom_count = total_count - int(entity_counts[i])
        if denom_count == 0:
            jackknife[i] = np.nan
            continue
        val = (total_sum - float(entity_sums[i])) / denom_count
        jackknife[i] = float(np.sqrt(val)) if sqrt_mean else val
    a = _compute_acceleration_from_jackknife(jackknife)
    alpha_1, alpha_2, fallback_reason = _bca_percentile_points(p0, a, confidence)
    ci_lo = float(np.percentile(resampled, 100.0 * alpha_1, method="linear"))
    ci_hi = float(np.percentile(resampled, 100.0 * alpha_2, method="linear"))
    return BootstrapResult(
        mean=ground_truth_mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        fallback_reason=fallback_reason,
    )
