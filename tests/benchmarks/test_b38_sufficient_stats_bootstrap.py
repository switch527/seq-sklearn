"""Phase B38 sufficient-stats bootstrap fast-path tests.

Closes D-B13.7: covers `entity_block_bootstrap_ci_mean_fast`,
the O(E) per-resample primitive that bypasses row
concatenation when the metric is `nanmean` (or
`sqrt(nanmean(x))` for RMSE).
"""

import numpy as np
import pytest
from benchmarks.metrics.bootstrap import (
    BootstrapResult,
    entity_block_bootstrap_ci,
    entity_block_bootstrap_ci_mean_fast,
)


def _sqrt_mean_metric(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x)))


_ABS_TOL = 1e-9
# BCa CI bounds tolerance: documented in B38.4 R-B38-Risk-1. The
# `p0 = mean(resampled <= ground_truth)` step has a discrete
# `<=` boundary; ULP-level drift between fast/naive resample
# values can flip one sample across the boundary, shifting p0
# by 1/n_resamples and propagating nonlinearly through the BCa
# transform. The mean and percentile-method bounds use the
# strict ULP-level _ABS_TOL.
_BCA_BOUNDS_TOL = 1e-2


def _fixture_losses_and_entities(
    *, n_entities: int = 6, rows_per_entity: int = 50, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    n_rows = n_entities * rows_per_entity
    losses = rng.uniform(0.1, 2.5, size=n_rows).astype(np.float64)
    entity_ids = np.repeat(np.arange(n_entities, dtype=np.int64), rows_per_entity)
    return losses, entity_ids


# =============================================================================
# B38.3.1 Equivalence: fast vs naive
# =============================================================================


def test_fast_path_matches_naive_for_percentile_mean() -> None:
    losses, entity_ids = _fixture_losses_and_entities()
    naive = entity_block_bootstrap_ci(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="percentile",
    )
    fast = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="percentile",
    )
    assert fast.mean == pytest.approx(naive.mean, abs=_ABS_TOL)
    assert fast.ci_lo == pytest.approx(naive.ci_lo, abs=_ABS_TOL)
    assert fast.ci_hi == pytest.approx(naive.ci_hi, abs=_ABS_TOL)
    assert fast.fallback_reason is None
    assert naive.fallback_reason is None


def test_fast_path_matches_naive_for_bca_mean() -> None:
    losses, entity_ids = _fixture_losses_and_entities()
    naive = entity_block_bootstrap_ci(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="bca",
    )
    fast = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="bca",
    )
    assert fast.mean == pytest.approx(naive.mean, abs=_ABS_TOL)
    assert fast.ci_lo == pytest.approx(naive.ci_lo, abs=_BCA_BOUNDS_TOL)
    assert fast.ci_hi == pytest.approx(naive.ci_hi, abs=_BCA_BOUNDS_TOL)
    assert fast.fallback_reason == naive.fallback_reason


def test_fast_path_matches_naive_for_percentile_sqrt_mean() -> None:
    losses, entity_ids = _fixture_losses_and_entities()
    naive = entity_block_bootstrap_ci(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="percentile",
        metric_fn=_sqrt_mean_metric,
    )
    fast = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="percentile",
        sqrt_mean=True,
    )
    assert fast.mean == pytest.approx(naive.mean, abs=_ABS_TOL)
    assert fast.ci_lo == pytest.approx(naive.ci_lo, abs=_ABS_TOL)
    assert fast.ci_hi == pytest.approx(naive.ci_hi, abs=_ABS_TOL)
    assert fast.fallback_reason is None


def test_fast_path_matches_naive_for_bca_sqrt_mean() -> None:
    losses, entity_ids = _fixture_losses_and_entities()
    naive = entity_block_bootstrap_ci(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="bca",
        metric_fn=_sqrt_mean_metric,
    )
    fast = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=500,
        seed=123,
        ci_method="bca",
        sqrt_mean=True,
    )
    assert fast.mean == pytest.approx(naive.mean, abs=_ABS_TOL)
    assert fast.ci_lo == pytest.approx(naive.ci_lo, abs=_BCA_BOUNDS_TOL)
    assert fast.ci_hi == pytest.approx(naive.ci_hi, abs=_BCA_BOUNDS_TOL)
    assert fast.fallback_reason == naive.fallback_reason


# =============================================================================
# B38.3.2 NaN handling
# =============================================================================


def test_fast_path_handles_partial_nan_rows() -> None:
    """NaN rows must be ignored (nanmean semantics). The fast path's
    ground-truth mean and per-resample mean must equal naive
    `np.nanmean` applied to the same data."""
    losses, entity_ids = _fixture_losses_and_entities()
    # Mark 20% of rows NaN deterministically.
    losses = losses.copy()
    rng = np.random.Generator(np.random.PCG64(99))
    nan_idx = rng.choice(losses.size, size=losses.size // 5, replace=False)
    losses[nan_idx] = np.nan

    naive = entity_block_bootstrap_ci(
        losses,
        entity_ids,
        n_resamples=300,
        seed=42,
        ci_method="percentile",
    )
    fast = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=300,
        seed=42,
        ci_method="percentile",
    )
    assert fast.mean == pytest.approx(naive.mean, abs=_ABS_TOL)
    assert fast.ci_lo == pytest.approx(naive.ci_lo, abs=_ABS_TOL)
    assert fast.ci_hi == pytest.approx(naive.ci_hi, abs=_ABS_TOL)


def test_fast_path_raises_on_all_nan_input() -> None:
    losses = np.full(20, np.nan, dtype=np.float64)
    entity_ids = np.repeat(np.arange(4, dtype=np.int64), 5)
    with pytest.raises(ValueError, match="all losses are NaN"):
        entity_block_bootstrap_ci_mean_fast(
            losses,
            entity_ids,
            n_resamples=10,
            seed=0,
        )


# =============================================================================
# B38.3.3 Degenerate single-entity
# =============================================================================


def test_fast_path_single_entity_returns_degenerate_ci() -> None:
    rng = np.random.Generator(np.random.PCG64(0))
    losses = rng.uniform(0.5, 1.5, size=30)
    entity_ids = np.zeros(30, dtype=np.int64)
    result = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=100,
        seed=0,
    )
    assert isinstance(result, BootstrapResult)
    assert result.ci_lo == result.mean == result.ci_hi
    assert result.fallback_reason is None


# =============================================================================
# B38.3.4 Determinism + seed
# =============================================================================


def test_fast_path_deterministic_at_fixed_seed() -> None:
    losses, entity_ids = _fixture_losses_and_entities()
    a = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=200,
        seed=2026,
        ci_method="bca",
    )
    b = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=200,
        seed=2026,
        ci_method="bca",
    )
    assert a == b


def test_fast_path_different_seeds_diverge() -> None:
    losses, entity_ids = _fixture_losses_and_entities()
    a = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=200,
        seed=1,
        ci_method="percentile",
    )
    b = entity_block_bootstrap_ci_mean_fast(
        losses,
        entity_ids,
        n_resamples=200,
        seed=2,
        ci_method="percentile",
    )
    # Same mean (deterministic from data), different bounds.
    assert a.mean == pytest.approx(b.mean, abs=_ABS_TOL)
    assert a.ci_lo != b.ci_lo or a.ci_hi != b.ci_hi
