"""Phase B26 cleanup bundle tests (26 named; 26 collected).

Closes D-B23.2 (CI-sentinel @model_validator on the 4 non-
EnsembleLift RollupRow schemas) + D-B23.3 + D-B24.3 (helper
guard symmetry on _render_partial_coverage_footnote).

Test layout:
- B26.4.1: per-schema happy + 2 mixed-reject variants
  (#1-#16) + reject pairs (#17-#24)
- B26.4.2: helper guard (#25)
- B26.4.3: B17 fixture backstop (#26)
"""

import pytest
from benchmarks.bootstrap_manifest import (
    HPOUpliftRollupRow,
    PairwiseRollupRow,
    RollupRow,
    TrainingTimeRollupRow,
)
from benchmarks.report.ensemble_lift import _render_partial_coverage_footnote
from pydantic import ValidationError

# Shared base kwargs for each schema (omit metric_* and
# bootstrap_skipped_reason so per-test cases vary them).

_ROLLUP_BASE: dict[str, object] = dict(
    dataset_name="ds",
    model_name="m",
    task_type="binary",
    primary_metric="log_loss",
    n_seeds=2,
    n_cells_evaluated=4,
    n_skipped_cells=0,
    n_rows=100,
    n_entities=10,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    manifest_fingerprint="f" * 64,
)

_PAIRWISE_BASE: dict[str, object] = dict(
    dataset_name="ds",
    model_a="a",
    model_b="b",
    task_type="binary",
    primary_metric="complementarity_score",
    n_seeds=2,
    n_cells_evaluated=4,
    n_skipped_cells=0,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    manifest_fingerprint="f" * 64,
)

_TRAINING_TIME_BASE: dict[str, object] = dict(
    dataset_name="ds",
    model_name="m",
    hardware_tier="cpu",
    task_type="binary",
    primary_metric="wall_seconds",
    n_seeds=2,
    n_cells_evaluated=4,
    n_skipped_cells=0,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    manifest_fingerprint="f" * 64,
)

_HPO_UPLIFT_BASE: dict[str, object] = dict(
    dataset_name="ds",
    model_name="m",
    task_type="binary",
    primary_metric="delta",
    primary_loss_column="log_loss",
    n_seeds=2,
    n_folds=2,
    n_cells_paired=4,
    n_skipped_cells=0,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    manifest_fingerprint="f" * 64,
)


# =============================================================================
# B26.4.1 CI-sentinel validator per schema (#1-#16 happy + mixed)
# =============================================================================

# --- RollupRow ---


def test_rollup_row_accepts_non_sentinel() -> None:
    RollupRow(**_ROLLUP_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason=None)


def test_rollup_row_accepts_sentinel() -> None:
    RollupRow(**_ROLLUP_BASE, bootstrap_skipped_reason="no_data")


def test_rollup_row_rejects_mixed_a_mean_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        RollupRow(**_ROLLUP_BASE, primary_metric_mean=0.5)


def test_rollup_row_rejects_mixed_b_ci_lo_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        RollupRow(**_ROLLUP_BASE, primary_metric_ci_lo=0.4)


# --- PairwiseRollupRow ---


def test_pairwise_rollup_row_accepts_non_sentinel() -> None:
    PairwiseRollupRow(**_PAIRWISE_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason=None)


def test_pairwise_rollup_row_accepts_sentinel() -> None:
    PairwiseRollupRow(**_PAIRWISE_BASE, bootstrap_skipped_reason="no_data")


def test_pairwise_rollup_row_rejects_mixed_a_mean_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        PairwiseRollupRow(**_PAIRWISE_BASE, primary_metric_mean=0.5)


def test_pairwise_rollup_row_rejects_mixed_b_ci_lo_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        PairwiseRollupRow(**_PAIRWISE_BASE, primary_metric_ci_lo=0.4)


# --- TrainingTimeRollupRow ---


def test_training_time_rollup_row_accepts_non_sentinel() -> None:
    TrainingTimeRollupRow(**_TRAINING_TIME_BASE, primary_metric_mean=10.0, primary_metric_ci_lo=8.0, primary_metric_ci_hi=12.0, bootstrap_skipped_reason=None)


def test_training_time_rollup_row_accepts_sentinel() -> None:
    TrainingTimeRollupRow(**_TRAINING_TIME_BASE, bootstrap_skipped_reason="no_data")


def test_training_time_rollup_row_rejects_mixed_a_mean_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        TrainingTimeRollupRow(**_TRAINING_TIME_BASE, primary_metric_mean=10.0)


def test_training_time_rollup_row_rejects_mixed_b_ci_lo_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        TrainingTimeRollupRow(**_TRAINING_TIME_BASE, primary_metric_ci_lo=8.0)


# --- HPOUpliftRollupRow ---


def test_hpo_uplift_rollup_row_accepts_non_sentinel() -> None:
    HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, primary_metric_mean=0.1, primary_metric_ci_lo=0.05, primary_metric_ci_hi=0.15, bootstrap_skipped_reason=None)


def test_hpo_uplift_rollup_row_accepts_sentinel() -> None:
    HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, bootstrap_skipped_reason="no_data")


def test_hpo_uplift_rollup_row_rejects_mixed_a_mean_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, primary_metric_mean=0.1)


def test_hpo_uplift_rollup_row_rejects_mixed_b_ci_lo_only_set() -> None:
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, primary_metric_ci_lo=0.05)


# =============================================================================
# B26.4.1 Reject pairs per schema (#17-#24)
# =============================================================================


def test_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(
        ValidationError,
        match=r"non-sentinel rows must have bootstrap_skipped_reason=None",
    ):
        RollupRow(**_ROLLUP_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason="oops")


def test_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(
        ValidationError,
        match=r"sentinel rows must populate bootstrap_skipped_reason",
    ):
        RollupRow(**_ROLLUP_BASE, bootstrap_skipped_reason=None)


def test_pairwise_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(
        ValidationError,
        match=r"non-sentinel rows must have bootstrap_skipped_reason=None",
    ):
        PairwiseRollupRow(**_PAIRWISE_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason="oops")


def test_pairwise_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(
        ValidationError,
        match=r"sentinel rows must populate bootstrap_skipped_reason",
    ):
        PairwiseRollupRow(**_PAIRWISE_BASE, bootstrap_skipped_reason=None)


def test_training_time_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(
        ValidationError,
        match=r"non-sentinel rows must have bootstrap_skipped_reason=None",
    ):
        TrainingTimeRollupRow(**_TRAINING_TIME_BASE, primary_metric_mean=10.0, primary_metric_ci_lo=8.0, primary_metric_ci_hi=12.0, bootstrap_skipped_reason="oops")


def test_training_time_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(
        ValidationError,
        match=r"sentinel rows must populate bootstrap_skipped_reason",
    ):
        TrainingTimeRollupRow(**_TRAINING_TIME_BASE, bootstrap_skipped_reason=None)


def test_hpo_uplift_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(
        ValidationError,
        match=r"non-sentinel rows must have bootstrap_skipped_reason=None",
    ):
        HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, primary_metric_mean=0.1, primary_metric_ci_lo=0.05, primary_metric_ci_hi=0.15, bootstrap_skipped_reason="oops")


def test_hpo_uplift_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(
        ValidationError,
        match=r"sentinel rows must populate bootstrap_skipped_reason",
    ):
        HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, bootstrap_skipped_reason=None)


# =============================================================================
# B26.4.2 Helper guard (#25)
# =============================================================================


def test_render_partial_coverage_footnote_empty_input_returns_empty_string() -> None:
    """R-B26-2: empty input returns "" (matches the sibling
    helper symmetry)."""
    assert _render_partial_coverage_footnote([]) == ""


# =============================================================================
# B26.4.3 B17 byte-pin fixture backstop (#26)
# =============================================================================


def test_existing_b17_byte_pin_fixtures_satisfy_ci_sentinel_invariant() -> None:
    """qa-R1-C2 closure: the B17 byte-pin helpers produce
    non-sentinel rows (populated metrics + None skipped_reason).
    They must continue to construct under the new validator."""
    from tests.benchmarks.test_b17_byte_identity_pins import (
        _make_ensemble_lift_rollup,
        _make_hpo_uplift_rollup,
        _make_pairwise_rollup,
        _make_training_time_rollup,
    )

    # All 4 helpers construct without raising.
    pairwise = _make_pairwise_rollup()
    training_time = _make_training_time_rollup()
    hpo_uplift = _make_hpo_uplift_rollup()
    ensemble_lift = _make_ensemble_lift_rollup()

    # Sanity: each fixture is a non-sentinel row.
    assert pairwise[0].bootstrap_skipped_reason is None
    assert training_time[0].bootstrap_skipped_reason is None
    assert hpo_uplift[0].bootstrap_skipped_reason is None
    assert ensemble_lift[0].bootstrap_skipped_reason is None

    # Sanity: each fixture has populated metric fields.
    assert pairwise[0].primary_metric_mean is not None
    assert training_time[0].primary_metric_mean is not None
    assert hpo_uplift[0].primary_metric_mean is not None
    assert ensemble_lift[0].primary_metric_mean is not None
