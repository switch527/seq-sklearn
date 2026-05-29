"""Entity-block bootstrap CI primitive (Phase B13 / B5.4).

Pure function: takes pre-computed per-row losses + per-row entity
ids + `(n_resamples, seed, metric_fn)`, returns
`(mean, ci_lo, ci_hi)`.

Design contracts (see `docs/benchmark_suite_design_b13_delta.md`
R1-R5):

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

from collections.abc import Callable

import numpy as np

__all__ = [
    "entity_block_bootstrap_ci",
]


# Default bootstrap parameters per B5.4 + R3 / R4 of the B13 delta.
_DEFAULT_N_RESAMPLES: int = 10_000
_DEFAULT_SEED: int = 0xB13_5EED_B007
_DEFAULT_CONFIDENCE: float = 0.95
# RNG-algorithm pin (Gemini-I2 + R-B13-2): explicit so the
# aggregator can record the actual algorithm name on every
# RollupRow without re-deriving it from `type(rng).__name__`.
BOOTSTRAP_RNG_ALGORITHM: str = "PCG64"


def _default_metric_fn(x: np.ndarray) -> float:
    """Classification default: nanmean (log_loss per-row vector)."""
    return float(np.nanmean(x))


def entity_block_bootstrap_ci(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    confidence: float = _DEFAULT_CONFIDENCE,
    seed: int = _DEFAULT_SEED,
    metric_fn: Callable[[np.ndarray], float] = _default_metric_fn,
) -> tuple[float, float, float]:
    """Entity-block percentile bootstrap CI.

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

    Returns:
        `(mean, ci_lo, ci_hi)` where `mean` is `metric_fn`
        applied to the unresampled loss vector and the CI is the
        percentile interval over the `n_resamples` resampled
        `metric_fn` values.

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
        return ground_truth_mean, ground_truth_mean, ground_truth_mean

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
    ci_lo = float(np.percentile(resampled, 100.0 * alpha, method="linear"))
    ci_hi = float(np.percentile(resampled, 100.0 * (1.0 - alpha), method="linear"))
    return ground_truth_mean, ci_lo, ci_hi
