"""Phase B29 cell-count bounds tests.

Closes D-B28.1 (n_pair_grid <= n_seeds * n_folds on
EnsembleLiftRollupRow) and D-B28.2 partial (n_entities <=
n_rows on RollupRow).

Test layout:
- B29.4.1: EnsembleLift n_pair_grid bound (#1-#3)
- B29.4.2: RollupRow n_entities bound (#4-#6)
- B29.4.3: existing-fixture backstop (#7)
"""

import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    RollupRow,
)
from pydantic import ValidationError

# B29.4.1 EnsembleLift base: valid CI-sentinel + oracle state
# so the B23 + B26 + B28 clauses pass silently and the new
# B29 row-count clause is the one that fires/passes.

_ENSEMBLE_LIFT_BASE: dict[str, object] = dict(
    dataset_name="ds",
    task_type="binary",
    primary_metric="delta_loss",
    primary_loss_column="log_loss",
    n_skipped_cells=0,
    primary_metric_mean=0.20,
    primary_metric_ci_lo=0.15,
    primary_metric_ci_hi=0.25,
    oracle_metric_mean=0.10,
    oracle_metric_ci_lo=0.08,
    oracle_metric_ci_hi=0.12,
    bootstrap_seed=42,
    bootstrap_n_resamples=10_000,
    bootstrap_numpy_version="2.3.0",
    bootstrap_skipped_reason=None,
    manifest_fingerprint="f" * 64,
)

# B29.4.2 RollupRow base: non-sentinel so the B26 CI-sentinel
# clause passes silently.

_ROLLUP_BASE: dict[str, object] = dict(
    dataset_name="fake_binary",
    model_name="m1",
    task_type="binary",
    primary_metric="log_loss",
    n_seeds=2,
    n_cells_evaluated=4,
    n_skipped_cells=0,
    primary_metric_mean=0.215,
    primary_metric_ci_lo=0.200,
    primary_metric_ci_hi=0.230,
    bootstrap_seed=42,
    bootstrap_n_resamples=10_000,
    bootstrap_numpy_version="2.3.0",
    bootstrap_skipped_reason=None,
    manifest_fingerprint="f" * 64,
)


# =============================================================================
# B29.4.1 EnsembleLift n_pair_grid bound (R-B29-1)
# =============================================================================


def test_ensemble_lift_rollup_row_accepts_n_pair_grid_below_bound() -> None:
    EnsembleLiftRollupRow(
        **_ENSEMBLE_LIFT_BASE,
        n_seeds=2,
        n_folds=2,
        n_pair_grid=3,
        n_cells_paired=3,
        n_oracle_cells_paired=3,
    )


def test_ensemble_lift_rollup_row_accepts_n_pair_grid_equals_bound() -> None:
    EnsembleLiftRollupRow(
        **_ENSEMBLE_LIFT_BASE,
        n_seeds=2,
        n_folds=2,
        n_pair_grid=4,
        n_cells_paired=4,
        n_oracle_cells_paired=4,
    )


def test_ensemble_lift_rollup_row_rejects_n_pair_grid_exceeds_bound() -> None:
    with pytest.raises(
        ValidationError,
        match=r"n_pair_grid.*exceeds n_seeds \* n_folds",
    ):
        EnsembleLiftRollupRow(
            **_ENSEMBLE_LIFT_BASE,
            n_seeds=2,
            n_folds=2,
            n_pair_grid=5,
            n_cells_paired=4,
            n_oracle_cells_paired=4,
        )


# =============================================================================
# B29.4.2 RollupRow n_entities bound (R-B29-2)
# =============================================================================


def test_rollup_row_accepts_n_entities_below_n_rows() -> None:
    RollupRow(
        **_ROLLUP_BASE,
        n_rows=100,
        n_entities=10,
    )


def test_rollup_row_accepts_n_entities_equals_n_rows() -> None:
    RollupRow(
        **_ROLLUP_BASE,
        n_rows=4,
        n_entities=4,
    )


def test_rollup_row_rejects_n_entities_exceeds_n_rows() -> None:
    with pytest.raises(
        ValidationError,
        match=r"n_entities.*exceeds n_rows",
    ):
        RollupRow(
            **_ROLLUP_BASE,
            n_rows=4,
            n_entities=5,
        )


# =============================================================================
# B29.4.3 Existing-fixture backstop
# =============================================================================


def test_existing_fixtures_satisfy_b29_invariants() -> None:
    """arch-R1-I2 closure: B17 EnsembleLift fixture (B17 byte-
    pin) and B14 RollupRow fixture (test_bootstrap_render_regression)
    must both continue to construct under the new B29 validators.
    Construction IS the gate post-B29."""
    from tests.benchmarks.test_b17_byte_identity_pins import (
        _make_ensemble_lift_rollup,
    )
    from tests.benchmarks.test_bootstrap_render_regression import (
        _rollup_row,
    )

    ensemble_lift = _make_ensemble_lift_rollup()
    rollup = _rollup_row()

    # Sanity: both invariants satisfied.
    assert ensemble_lift[0].n_pair_grid <= ensemble_lift[0].n_seeds * ensemble_lift[0].n_folds
    assert rollup.n_entities <= rollup.n_rows
