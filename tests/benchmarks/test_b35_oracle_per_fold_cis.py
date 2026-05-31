"""Phase B35 oracle per-fold CIs tests.

Closes D-B22.2: per-fold CIs for the B16 ensemble_lift
ORACLE delta path. Parquet-audit-only at v1.
"""

from pathlib import Path

import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    FoldCI,
    ensemble_lift_rollup_path,
    load_ensemble_lift_rollup,
    write_ensemble_lift_rollup,
)
from benchmarks.experiments.ensemble_lift import PerCellLiftDelta

# Reuse the b22 test harness for the aggregator end-to-end runs.
from tests.benchmarks.test_b22_per_fold_cis import (
    _happy_cells,
    _run_ensemble_lift,
)

# =============================================================================
# 1. Schema field default is None
# =============================================================================


def test_ensemble_lift_rollup_row_per_fold_oracle_cis_default_is_none() -> None:
    """Constructing without per_fold_oracle_cis yields None."""
    row = EnsembleLiftRollupRow(
        dataset_name="ds",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=4,
        n_pair_grid=4,
        n_oracle_cells_paired=4,
        n_skipped_cells=0,
        primary_metric_mean=0.2,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=0.1,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="f" * 64,
    )
    assert row.per_fold_oracle_cis is None


# =============================================================================
# 2. Schema field accepts a populated list
# =============================================================================


def test_ensemble_lift_rollup_row_accepts_per_fold_oracle_cis_list() -> None:
    fold_cis = [
        FoldCI(
            fold_index=0,
            n_seeds=2,
            n_entities=4,
            metric_mean=0.10,
            metric_ci_lo=0.08,
            metric_ci_hi=0.12,
            ci_method="bca",
            ci_fallback_reason=None,
        ),
        FoldCI(
            fold_index=1,
            n_seeds=2,
            n_entities=4,
            metric_mean=0.11,
            metric_ci_lo=0.09,
            metric_ci_hi=0.13,
            ci_method="bca",
            ci_fallback_reason=None,
        ),
    ]
    row = EnsembleLiftRollupRow(
        dataset_name="ds",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=4,
        n_pair_grid=4,
        n_oracle_cells_paired=4,
        n_skipped_cells=0,
        primary_metric_mean=0.2,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=0.1,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_skipped_reason=None,
        per_fold_oracle_cis=fold_cis,
        manifest_fingerprint="f" * 64,
    )
    assert row.per_fold_oracle_cis == fold_cis


# =============================================================================
# 3. Parquet round-trip preserves per_fold_oracle_cis
# =============================================================================


def test_ensemble_lift_rollup_row_per_fold_oracle_cis_round_trips_through_parquet(
    tmp_path: Path,
) -> None:
    fold_cis = [
        FoldCI(
            fold_index=0,
            n_seeds=2,
            n_entities=4,
            metric_mean=0.10,
            metric_ci_lo=0.08,
            metric_ci_hi=0.12,
            ci_method="bca",
            ci_fallback_reason=None,
        ),
    ]
    row = EnsembleLiftRollupRow(
        dataset_name="ds",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=4,
        n_pair_grid=4,
        n_oracle_cells_paired=4,
        n_skipped_cells=0,
        primary_metric_mean=0.2,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=0.1,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_skipped_reason=None,
        per_fold_oracle_cis=fold_cis,
        manifest_fingerprint="f" * 64,
    )
    write_ensemble_lift_rollup(tmp_path, [row])
    loaded = load_ensemble_lift_rollup(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].per_fold_oracle_cis == fold_cis
    # Confirm the file exists at the documented path.
    assert ensemble_lift_rollup_path(tmp_path).exists()


# =============================================================================
# 4. Aggregator populates per_fold_oracle_cis when enabled + oracle cells present
# =============================================================================


def test_aggregator_populates_per_fold_oracle_cis_when_enabled_and_oracle_cells_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollup = _run_ensemble_lift(
        tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells()
    )
    row = rollup[0]
    assert row.per_fold_oracle_cis is not None
    assert len(row.per_fold_oracle_cis) >= 1
    for fc in row.per_fold_oracle_cis:
        assert isinstance(fc, FoldCI)
        assert fc.ci_method == "bca"
        assert fc.ci_fallback_reason in (None, "p0_at_edge", "a_overshoot")


# =============================================================================
# 5. per_fold_oracle_cis is None when flag is disabled
# =============================================================================


def test_aggregator_per_fold_oracle_cis_is_none_when_flag_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollup = _run_ensemble_lift(
        tmp_path, monkeypatch, per_fold_enabled=False, cells=_happy_cells()
    )
    row = rollup[0]
    assert row.per_fold_oracle_cis is None


# =============================================================================
# 6. per_fold_oracle_cis is None when n_oracle_cells_paired == 0
# =============================================================================


def test_aggregator_per_fold_oracle_cis_is_none_when_n_oracle_cells_paired_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cells with oracle_loss=None across every cell yield
    n_oracle_cells_paired=0; per_fold_oracle_cis must be None
    regardless of the flag."""
    cells_no_oracle = tuple(
        PerCellLiftDelta(
            seed=s,
            fold_index=f,
            loss_gbm=0.60,
            loss_gbm_plus_seq=0.40,
            delta_loss=0.20,
            oracle_loss=None,  # No oracle data for any cell.
        )
        for s in range(2)
        for f in range(2)
    )
    rollup = _run_ensemble_lift(
        tmp_path, monkeypatch, per_fold_enabled=True, cells=cells_no_oracle
    )
    row = rollup[0]
    assert row.n_oracle_cells_paired == 0
    assert row.per_fold_oracle_cis is None


# =============================================================================
# 7. Oracle per-fold uses independent seed offset
# =============================================================================


def test_aggregator_per_fold_oracle_cis_uses_independent_seed_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture all entity_block_bootstrap_ci seed kwargs across
    a per-fold-enabled run. With n_folds=2, the expected call
    order on EnsembleLift is:
    1. Main pooled (BOOTSTRAP_DEFAULT_SEED)
    2. Main per-fold x2 (BOOTSTRAP_DEFAULT_SEED ^
       BOOTSTRAP_PER_FOLD_SEED_OFFSET ^ fold_index)
    3. Oracle pooled (BOOTSTRAP_DEFAULT_SEED ^
       BOOTSTRAP_ORACLE_SEED_OFFSET)
    4. Oracle per-fold x2 (BOOTSTRAP_DEFAULT_SEED ^
       BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET ^ fold_index)

    The 2 oracle per-fold seeds must:
    - differ from each other,
    - differ from the 2 main per-fold seeds,
    - differ from BOOTSTRAP_DEFAULT_SEED and from the pooled
      oracle seed.
    """
    seeds_captured: list[int] = []

    import benchmarks.report._bootstrap_aggregate as _agg
    import benchmarks.report.bootstrap_ensemble_lift as _module
    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as _real,
    )

    def _capture(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seeds_captured.append(int(kwargs["seed"]))  # type: ignore[arg-type]
        return _real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capture)
    monkeypatch.setattr(_agg, "entity_block_bootstrap_ci", _capture)

    _run_ensemble_lift(
        tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells()
    )

    from benchmarks.report._bootstrap_aggregate import (
        BOOTSTRAP_DEFAULT_SEED,
        BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET,
        BOOTSTRAP_ORACLE_SEED_OFFSET,
    )

    # 1 main pool + 2 main per-fold + 1 oracle pool + 2 oracle per-fold = 6
    assert len(seeds_captured) == 6
    assert seeds_captured[0] == BOOTSTRAP_DEFAULT_SEED
    main_per_fold_seeds = seeds_captured[1:3]
    assert seeds_captured[3] == BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET
    oracle_per_fold_seeds = seeds_captured[4:6]
    expected_oracle_per_fold = [
        BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET ^ 0,
        BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET ^ 1,
    ]
    assert oracle_per_fold_seeds == expected_oracle_per_fold
    # Cross-check the 3 streams stay distinct.
    for s in oracle_per_fold_seeds:
        assert s not in main_per_fold_seeds
        assert s != BOOTSTRAP_DEFAULT_SEED
        assert s != BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET
    # And the two oracle per-fold seeds differ from each other.
    assert oracle_per_fold_seeds[0] != oracle_per_fold_seeds[1]
