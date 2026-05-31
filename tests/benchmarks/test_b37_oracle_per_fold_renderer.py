"""Phase B37 oracle per-fold CIs renderer tests.

Closes D-B35.1: extend render_per_fold_cis_footnote with a
`scope` column ("main" or "oracle") so per_fold_oracle_cis
(B35) surfaces alongside per_fold_cis (B25).
"""

from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    FoldCI,
    RollupRow,
)
from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    PerDatasetLift,
    WilcoxonResult,
)
from benchmarks.report._bootstrap_render import render_per_fold_cis_footnote
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown_with_ci

_SECTION = "### Per-fold CIs"


def _fold_ci(
    *,
    fold_index: int = 0,
    n_seeds: int = 2,
    n_entities: int = 4,
    metric_mean: float | None = 0.215,
    metric_ci_lo: float | None = 0.200,
    metric_ci_hi: float | None = 0.230,
    ci_method: str = "bca",
    ci_fallback_reason: str | None = None,
) -> FoldCI:
    return FoldCI(
        fold_index=fold_index,
        n_seeds=n_seeds,
        n_entities=n_entities,
        metric_mean=metric_mean,
        metric_ci_lo=metric_ci_lo,
        metric_ci_hi=metric_ci_hi,
        ci_method=ci_method,
        ci_fallback_reason=ci_fallback_reason,
    )


def _ensemble_lift_row(
    *,
    dataset_name: str = "ds_one",
    per_fold_cis: list[FoldCI] | None = None,
    per_fold_oracle_cis: list[FoldCI] | None = None,
) -> EnsembleLiftRollupRow:
    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=4,
        n_pair_grid=4,
        n_oracle_cells_paired=4,
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
        per_fold_cis=per_fold_cis,
        per_fold_oracle_cis=per_fold_oracle_cis,
        manifest_fingerprint="e" * 64,
    )


def _rollup_row(*, per_fold_cis: list[FoldCI] | None = None) -> RollupRow:
    return RollupRow(
        dataset_name="fake_binary",
        model_name="m1",
        task_type="binary",
        primary_metric="log_loss",
        n_seeds=2,
        n_cells_evaluated=4,
        n_skipped_cells=0,
        n_rows=40,
        n_entities=4,
        primary_metric_mean=0.215,
        primary_metric_ci_lo=0.200,
        primary_metric_ci_hi=0.230,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        per_fold_cis=per_fold_cis,
        manifest_fingerprint="d" * 64,
    )


# =============================================================================
# 1. Header includes the scope column
# =============================================================================


def test_per_fold_cis_footnote_emits_scope_column_in_header() -> None:
    row = _ensemble_lift_row(
        per_fold_cis=[_fold_ci()], per_fold_oracle_cis=[_fold_ci()]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert "| scope |" in md


# =============================================================================
# 2. Main rows carry scope="main"
# =============================================================================


def test_per_fold_cis_footnote_emits_main_rows_with_scope_main() -> None:
    row = _ensemble_lift_row(
        per_fold_cis=[_fold_ci()], per_fold_oracle_cis=None
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    section = md[md.find(_SECTION):]
    assert "| main |" in section
    # No oracle rows when per_fold_oracle_cis is None.
    fold_lines = [
        ln for ln in section.split("\n")
        if ln.startswith("| ds_one |") and "| oracle |" in ln
    ]
    assert fold_lines == []


# =============================================================================
# 3. Oracle rows carry scope="oracle"
# =============================================================================


def test_per_fold_cis_footnote_emits_oracle_rows_with_scope_oracle() -> None:
    row = _ensemble_lift_row(
        per_fold_cis=None, per_fold_oracle_cis=[_fold_ci()]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    section = md[md.find(_SECTION):]
    assert "| oracle |" in section
    main_lines = [
        ln for ln in section.split("\n")
        if ln.startswith("| ds_one |") and "| main |" in ln
    ]
    assert main_lines == []


# =============================================================================
# 4. Main rows emit before oracle rows when both present
# =============================================================================


def test_per_fold_cis_footnote_emits_main_before_oracle_when_both_present() -> None:
    row = _ensemble_lift_row(
        per_fold_cis=[_fold_ci(fold_index=0), _fold_ci(fold_index=1)],
        per_fold_oracle_cis=[_fold_ci(fold_index=0), _fold_ci(fold_index=1)],
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    section = md[md.find(_SECTION):]
    main_idx = section.find("| main |")
    oracle_idx = section.find("| oracle |")
    assert main_idx >= 0
    assert oracle_idx >= 0
    assert main_idx < oracle_idx


# =============================================================================
# 5. ensemble_lift renderer wires the oracle field through
# =============================================================================


def test_ensemble_lift_renderer_emits_oracle_per_fold_rows_when_present() -> None:
    result = EnsembleLiftExperimentResult(
        run_id="b37-test",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(
            PerDatasetLift(
                dataset_name="ds_one",
                task_type="binary",
                primary_loss_column="log_loss",
                n_cells_paired=4,
                loss_gbm_only_mean=0.50,
                loss_gbm_plus_seq_mean=0.30,
                delta_loss_mean=0.20,
                delta_loss_std=0.01,
                oracle_loss_mean=0.25,
                oracle_delta_loss_mean=0.25,
            ),
        ),
        wilcoxon=WilcoxonResult(
            statistic=1.0,
            p_value=0.10,
            holm_adjusted_p_value=0.10,
            n_datasets=1,
            family_size=1,
        ),
    )
    rollup = [
        _ensemble_lift_row(
            per_fold_cis=[_fold_ci()],
            per_fold_oracle_cis=[_fold_ci()],
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    section = md[md.find(_SECTION):]
    assert "| main |" in section
    assert "| oracle |" in section


# =============================================================================
# 6. Non-EnsembleLift schema renders only main rows
# =============================================================================


def test_per_fold_cis_footnote_for_non_ensemble_lift_schema_omits_oracle() -> None:
    """RollupRow has no per_fold_oracle_cis field; the helper
    falls back to [] via getattr. Scope column still appears
    in the header but only main rows render."""
    row = _rollup_row(per_fold_cis=[_fold_ci()])
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert "| scope |" in md  # header still has scope column
    section = md[md.find(_SECTION):]
    assert "| main |" in section
    # No oracle rows for RollupRow.
    oracle_lines = [
        ln for ln in section.split("\n")
        if ln.startswith("| fake_binary |") and "| oracle |" in ln
    ]
    assert oracle_lines == []
