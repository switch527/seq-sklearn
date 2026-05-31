"""Phase B28 schema-validator cleanups tests.

Closes D-B27.2 (oracle CI-sentinel invariant on
EnsembleLiftRollupRow composed with the B27 primary clause)
and D-B26.1 (cell-count bounds on HPOUpliftRollupRow).

Test layout:
- B28.4.1: oracle clause on EnsembleLiftRollupRow (#1-#5)
- B28.4.2: cell-count bounds on HPOUpliftRollupRow (#6-#10)
- B28.4.3: B17 fixture backstop (#11)
"""

import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    HPOUpliftRollupRow,
)
from pydantic import ValidationError

# Shared bases. The primary axis state is always valid
# (non-sentinel = all-set primaries + skipped=None) so the
# primary CI-sentinel clause passes silently and tests target
# only the new clauses (oracle for EnsembleLift, cell-count
# for HPOUplift).

_ENSEMBLE_LIFT_BASE: dict[str, object] = dict(
    dataset_name="ds",
    task_type="binary",
    primary_metric="delta_loss",
    primary_loss_column="log_loss",
    n_seeds=2,
    n_folds=2,
    n_cells_paired=4,
    n_pair_grid=4,
    n_skipped_cells=0,
    primary_metric_mean=0.20,
    primary_metric_ci_lo=0.15,
    primary_metric_ci_hi=0.25,
    bootstrap_seed=42,
    bootstrap_n_resamples=10_000,
    bootstrap_numpy_version="2.3.0",
    bootstrap_skipped_reason=None,
    manifest_fingerprint="f" * 64,
)

_HPO_UPLIFT_NON_SENTINEL_BASE: dict[str, object] = dict(
    dataset_name="ds",
    model_name="m",
    task_type="binary",
    primary_metric="delta",
    primary_loss_column="log_loss",
    n_seeds=2,
    n_folds=2,
    primary_metric_mean=0.1,
    primary_metric_ci_lo=0.05,
    primary_metric_ci_hi=0.15,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    bootstrap_skipped_reason=None,
    manifest_fingerprint="f" * 64,
)

_HPO_UPLIFT_SENTINEL_BASE: dict[str, object] = dict(
    dataset_name="ds",
    model_name="m",
    task_type="binary",
    primary_metric="delta",
    primary_loss_column="log_loss",
    n_seeds=2,
    n_folds=2,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    bootstrap_skipped_reason="no_data",
    manifest_fingerprint="f" * 64,
)


# =============================================================================
# B28.4.1 Oracle CI-sentinel invariant (R-B28-1)
# =============================================================================


def test_ensemble_lift_rollup_row_accepts_oracle_all_none_with_zero_oracle_cells() -> None:
    EnsembleLiftRollupRow(
        **{**_ENSEMBLE_LIFT_BASE, "n_oracle_cells_paired": 0},
        oracle_metric_mean=None,
        oracle_metric_ci_lo=None,
        oracle_metric_ci_hi=None,
    )


def test_ensemble_lift_rollup_row_accepts_oracle_all_set_with_positive_oracle_cells() -> None:
    EnsembleLiftRollupRow(
        **{**_ENSEMBLE_LIFT_BASE, "n_oracle_cells_paired": 4},
        oracle_metric_mean=0.10,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
    )


_ORACLE_MIXED_VARIANTS: list[tuple[str, dict[str, float | None]]] = [
    ("oracle_mean_only", {"oracle_metric_mean": 0.10}),
    ("oracle_ci_lo_only", {"oracle_metric_ci_lo": 0.08}),
    ("oracle_ci_hi_only", {"oracle_metric_ci_hi": 0.12}),
    ("oracle_mean_and_ci_lo", {"oracle_metric_mean": 0.10, "oracle_metric_ci_lo": 0.08}),
    ("oracle_mean_and_ci_hi", {"oracle_metric_mean": 0.10, "oracle_metric_ci_hi": 0.12}),
    ("oracle_ci_lo_and_ci_hi", {"oracle_metric_ci_lo": 0.08, "oracle_metric_ci_hi": 0.12}),
]


@pytest.mark.parametrize(("variant_name", "oracle_kwargs"), _ORACLE_MIXED_VARIANTS)
def test_ensemble_lift_rollup_row_rejects_oracle_partially_set(
    variant_name: str, oracle_kwargs: dict[str, float | None]
) -> None:
    """B28 / D-B27.2 closure: 6 partial-oracle variants
    rejected. Uses n_oracle_cells_paired=4 so the partial
    branch fires (not the gating branches)."""
    with pytest.raises(
        ValidationError,
        match=r"oracle_metric_mean.*must be all-None or all-non-None",
    ):
        EnsembleLiftRollupRow(
            **{**_ENSEMBLE_LIFT_BASE, "n_oracle_cells_paired": 4},
            **oracle_kwargs,
        )


def test_ensemble_lift_rollup_row_rejects_oracle_none_with_positive_oracle_cells() -> None:
    with pytest.raises(
        ValidationError,
        match=r"all None but n_oracle_cells_paired > 0",
    ):
        EnsembleLiftRollupRow(
            **{**_ENSEMBLE_LIFT_BASE, "n_oracle_cells_paired": 4},
            oracle_metric_mean=None,
            oracle_metric_ci_lo=None,
            oracle_metric_ci_hi=None,
        )


def test_ensemble_lift_rollup_row_rejects_oracle_set_with_zero_oracle_cells() -> None:
    with pytest.raises(
        ValidationError,
        match=r"all populated but n_oracle_cells_paired == 0",
    ):
        EnsembleLiftRollupRow(
            **{**_ENSEMBLE_LIFT_BASE, "n_oracle_cells_paired": 0},
            oracle_metric_mean=0.10,
            oracle_metric_ci_lo=0.08,
            oracle_metric_ci_hi=0.12,
        )


# =============================================================================
# B28.4.2 HPOUplift cell-count invariants (R-B28-2)
# =============================================================================


def test_hpo_uplift_rollup_row_accepts_n_cells_paired_equals_bound() -> None:
    HPOUpliftRollupRow(
        **_HPO_UPLIFT_NON_SENTINEL_BASE,
        n_cells_paired=4,
        n_skipped_cells=0,
    )


def test_hpo_uplift_rollup_row_accepts_n_skipped_equals_n_paired() -> None:
    """Matches the `paired_but_no_valid_loss` aggregator emit
    shape at bootstrap_hpo_uplift.py:252-264."""
    HPOUpliftRollupRow(
        **_HPO_UPLIFT_NON_SENTINEL_BASE,
        n_cells_paired=4,
        n_skipped_cells=4,
    )


def test_hpo_uplift_rollup_row_accepts_sentinel_with_zero_counts() -> None:
    HPOUpliftRollupRow(
        **_HPO_UPLIFT_SENTINEL_BASE,
        n_cells_paired=0,
        n_skipped_cells=0,
    )


def test_hpo_uplift_rollup_row_rejects_n_cells_paired_exceeds_seeds_times_folds() -> None:
    with pytest.raises(
        ValidationError,
        match=r"n_cells_paired.*exceeds n_seeds \* n_folds",
    ):
        HPOUpliftRollupRow(
            **_HPO_UPLIFT_NON_SENTINEL_BASE,
            n_cells_paired=5,
            n_skipped_cells=0,
        )


def test_hpo_uplift_rollup_row_rejects_n_skipped_cells_exceeds_n_cells_paired() -> None:
    with pytest.raises(
        ValidationError,
        match=r"n_skipped_cells.*exceeds n_cells_paired",
    ):
        HPOUpliftRollupRow(
            **_HPO_UPLIFT_NON_SENTINEL_BASE,
            n_cells_paired=3,
            n_skipped_cells=4,
        )


# =============================================================================
# B28.4.3 B17 fixture backstop
# =============================================================================


def test_existing_b17_byte_pin_fixtures_satisfy_b28_invariants() -> None:
    """Existing B17 helpers must continue to construct under
    both the new EnsembleLift oracle clause AND the new
    HPOUplift cell-count clause."""
    from tests.benchmarks.test_b17_byte_identity_pins import (
        _make_ensemble_lift_rollup,
        _make_hpo_uplift_rollup,
    )

    ensemble_lift = _make_ensemble_lift_rollup()
    hpo_uplift = _make_hpo_uplift_rollup()

    # EnsembleLift: oracle metrics populated, n_oracle_cells_paired > 0.
    assert ensemble_lift[0].n_oracle_cells_paired > 0
    assert ensemble_lift[0].oracle_metric_mean is not None

    # HPOUplift: cell-count bounds satisfied.
    row = hpo_uplift[0]
    assert row.n_cells_paired <= row.n_seeds * row.n_folds
    assert row.n_skipped_cells <= row.n_cells_paired
