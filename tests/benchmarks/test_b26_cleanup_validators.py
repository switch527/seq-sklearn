"""Phase B26 + B27 cleanup bundle tests.

Closes D-B23.2 (CI-sentinel @model_validator on 4 non-
EnsembleLift RollupRow schemas; B26), D-B23.3 + D-B24.3
(helper guard symmetry on _render_partial_coverage_footnote;
B26), and D-B26.2 (compose CI-sentinel into
EnsembleLiftRollupRow; B27) + D-B26.3 (full 6-variant
mixed-reject coverage; B27).

Test layout:
- per-schema happy + sentinel + 6-variant mixed-reject
  (parametrized over 5 schemas)
- per-schema reject-pair (skipped-with-metrics +
  no-skipped-with-None-metrics)
- helper guard
- B17 backstop
"""

from typing import Any

import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
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

_ENSEMBLE_LIFT_BASE: dict[str, object] = dict(
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
    # Oracle metrics are independent of the CI-sentinel invariant
    # (D-B27.2); supply non-None values so the row stays valid
    # regardless of the primary_metric_* state.
    oracle_metric_mean=0.10,
    oracle_metric_ci_lo=0.08,
    oracle_metric_ci_hi=0.12,
    bootstrap_seed=42,
    bootstrap_n_resamples=100,
    bootstrap_numpy_version="2.0.0",
    manifest_fingerprint="f" * 64,
)


# =============================================================================
# Per-schema happy non-sentinel + happy sentinel (10 tests)
# =============================================================================


def test_rollup_row_accepts_non_sentinel() -> None:
    RollupRow(**_ROLLUP_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason=None)


def test_rollup_row_accepts_sentinel() -> None:
    RollupRow(**_ROLLUP_BASE, bootstrap_skipped_reason="no_data")


def test_pairwise_rollup_row_accepts_non_sentinel() -> None:
    PairwiseRollupRow(**_PAIRWISE_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason=None)


def test_pairwise_rollup_row_accepts_sentinel() -> None:
    PairwiseRollupRow(**_PAIRWISE_BASE, bootstrap_skipped_reason="no_data")


def test_training_time_rollup_row_accepts_non_sentinel() -> None:
    TrainingTimeRollupRow(**_TRAINING_TIME_BASE, primary_metric_mean=10.0, primary_metric_ci_lo=8.0, primary_metric_ci_hi=12.0, bootstrap_skipped_reason=None)


def test_training_time_rollup_row_accepts_sentinel() -> None:
    TrainingTimeRollupRow(**_TRAINING_TIME_BASE, bootstrap_skipped_reason="no_data")


def test_hpo_uplift_rollup_row_accepts_non_sentinel() -> None:
    HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, primary_metric_mean=0.1, primary_metric_ci_lo=0.05, primary_metric_ci_hi=0.15, bootstrap_skipped_reason=None)


def test_hpo_uplift_rollup_row_accepts_sentinel() -> None:
    HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, bootstrap_skipped_reason="no_data")


def test_ensemble_lift_rollup_row_accepts_non_sentinel() -> None:
    EnsembleLiftRollupRow(**_ENSEMBLE_LIFT_BASE, primary_metric_mean=0.2, primary_metric_ci_lo=0.15, primary_metric_ci_hi=0.25, bootstrap_skipped_reason=None)


def test_ensemble_lift_rollup_row_accepts_sentinel() -> None:
    EnsembleLiftRollupRow(**_ENSEMBLE_LIFT_BASE, bootstrap_skipped_reason="no_data")


# =============================================================================
# Per-schema mixed-reject parametrized (1 test, 30 cases)
# =============================================================================


# B27 / D-B26.3 closure: 6-variant mixed coverage (the 2^3 - 2
# = 6 partial-set combinations) across all 5 schemas.
_MIXED_VARIANTS: list[tuple[str, dict[str, float | None]]] = [
    ("mean_only", {"primary_metric_mean": 0.5}),
    ("ci_lo_only", {"primary_metric_ci_lo": 0.4}),
    ("ci_hi_only", {"primary_metric_ci_hi": 0.6}),
    ("mean_and_ci_lo", {"primary_metric_mean": 0.5, "primary_metric_ci_lo": 0.4}),
    ("mean_and_ci_hi", {"primary_metric_mean": 0.5, "primary_metric_ci_hi": 0.6}),
    ("ci_lo_and_ci_hi", {"primary_metric_ci_lo": 0.4, "primary_metric_ci_hi": 0.6}),
]


def _make_row(schema_name: str, **kwargs: Any) -> Any:
    if schema_name == "rollup_row":
        return RollupRow(**_ROLLUP_BASE, **kwargs)
    if schema_name == "pairwise":
        return PairwiseRollupRow(**_PAIRWISE_BASE, **kwargs)
    if schema_name == "training_time":
        return TrainingTimeRollupRow(**_TRAINING_TIME_BASE, **kwargs)
    if schema_name == "hpo_uplift":
        return HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, **kwargs)
    if schema_name == "ensemble_lift":
        return EnsembleLiftRollupRow(**_ENSEMBLE_LIFT_BASE, **kwargs)
    raise ValueError(f"unknown schema_name: {schema_name}")


@pytest.mark.parametrize(
    "schema_name",
    ["rollup_row", "pairwise", "training_time", "hpo_uplift", "ensemble_lift"],
)
@pytest.mark.parametrize(
    ("variant_name", "metric_kwargs"),
    _MIXED_VARIANTS,
)
def test_rollup_row_rejects_mixed_metric_state(
    schema_name: str, variant_name: str, metric_kwargs: dict[str, float | None]
) -> None:
    """B27 / D-B26.3 closure: all 6 partially-set metric
    combinations rejected by the CI-sentinel validator across
    all 5 schemas. 5 schemas x 6 variants = 30 parametrized
    cases."""
    with pytest.raises(ValidationError, match=r"must be all-None or all-non-None"):
        _make_row(schema_name, **metric_kwargs)


# =============================================================================
# Per-schema reject pairs (10 tests)
# =============================================================================


def test_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(ValidationError, match=r"non-sentinel rows must have bootstrap_skipped_reason=None"):
        RollupRow(**_ROLLUP_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason="oops")


def test_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(ValidationError, match=r"sentinel rows must populate bootstrap_skipped_reason"):
        RollupRow(**_ROLLUP_BASE, bootstrap_skipped_reason=None)


def test_pairwise_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(ValidationError, match=r"non-sentinel rows must have bootstrap_skipped_reason=None"):
        PairwiseRollupRow(**_PAIRWISE_BASE, primary_metric_mean=0.5, primary_metric_ci_lo=0.4, primary_metric_ci_hi=0.6, bootstrap_skipped_reason="oops")


def test_pairwise_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(ValidationError, match=r"sentinel rows must populate bootstrap_skipped_reason"):
        PairwiseRollupRow(**_PAIRWISE_BASE, bootstrap_skipped_reason=None)


def test_training_time_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(ValidationError, match=r"non-sentinel rows must have bootstrap_skipped_reason=None"):
        TrainingTimeRollupRow(**_TRAINING_TIME_BASE, primary_metric_mean=10.0, primary_metric_ci_lo=8.0, primary_metric_ci_hi=12.0, bootstrap_skipped_reason="oops")


def test_training_time_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(ValidationError, match=r"sentinel rows must populate bootstrap_skipped_reason"):
        TrainingTimeRollupRow(**_TRAINING_TIME_BASE, bootstrap_skipped_reason=None)


def test_hpo_uplift_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    with pytest.raises(ValidationError, match=r"non-sentinel rows must have bootstrap_skipped_reason=None"):
        HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, primary_metric_mean=0.1, primary_metric_ci_lo=0.05, primary_metric_ci_hi=0.15, bootstrap_skipped_reason="oops")


def test_hpo_uplift_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(ValidationError, match=r"sentinel rows must populate bootstrap_skipped_reason"):
        HPOUpliftRollupRow(**_HPO_UPLIFT_BASE, bootstrap_skipped_reason=None)


def test_ensemble_lift_rollup_row_rejects_metrics_set_with_skipped_reason() -> None:
    """B27 / D-B26.2 closure: same reject contract for the 5th
    schema."""
    with pytest.raises(ValidationError, match=r"non-sentinel rows must have bootstrap_skipped_reason=None"):
        EnsembleLiftRollupRow(**_ENSEMBLE_LIFT_BASE, primary_metric_mean=0.2, primary_metric_ci_lo=0.15, primary_metric_ci_hi=0.25, bootstrap_skipped_reason="oops")


def test_ensemble_lift_rollup_row_rejects_metrics_none_with_skipped_none() -> None:
    with pytest.raises(ValidationError, match=r"sentinel rows must populate bootstrap_skipped_reason"):
        EnsembleLiftRollupRow(**_ENSEMBLE_LIFT_BASE, bootstrap_skipped_reason=None)


# =============================================================================
# Helper guard + B17 backstop
# =============================================================================


def test_render_partial_coverage_footnote_empty_input_returns_empty_string() -> None:
    """R-B26-2: empty input returns "" (matches the sibling
    helper symmetry)."""
    assert _render_partial_coverage_footnote([]) == ""


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

    pairwise = _make_pairwise_rollup()
    training_time = _make_training_time_rollup()
    hpo_uplift = _make_hpo_uplift_rollup()
    ensemble_lift = _make_ensemble_lift_rollup()

    assert pairwise[0].bootstrap_skipped_reason is None
    assert training_time[0].bootstrap_skipped_reason is None
    assert hpo_uplift[0].bootstrap_skipped_reason is None
    assert ensemble_lift[0].bootstrap_skipped_reason is None

    assert pairwise[0].primary_metric_mean is not None
    assert training_time[0].primary_metric_mean is not None
    assert hpo_uplift[0].primary_metric_mean is not None
    assert ensemble_lift[0].primary_metric_mean is not None
