"""Phase B25 per-fold CIs renderer surface tests (26 named; 34 collected).

Closes D-B22.1: per-fold CIs footnote across all 5
`*_markdown_with_ci` renderers. The footnote reads EXACTLY 6
FoldCI fields (fold_index, metric_mean, metric_ci_lo,
metric_ci_hi, ci_method, ci_fallback_reason). The audit
fields n_seeds + n_entities are NOT rendered (D-B25.3).

Test layout:
- B25.3.1: shared helper (#1-#10b)
- B25.3.2: per-renderer present emission (#11-#15)
- B25.3.3: per-renderer silent-when-absent (#16-#20)
- B25.3.4: mixed parametrized (#21)
- B25.3.5: empty list edge (#22)
- B25.3.6: section ordering (#22a)
"""

from collections.abc import Callable

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    FoldCI,
    HPOUpliftRollupRow,
    PairwiseRollupRow,
    RollupRow,
    TrainingTimeRollupRow,
)
from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    PerDatasetLift,
    WilcoxonResult,
)
from benchmarks.report._bootstrap_render import render_per_fold_cis_footnote
from benchmarks.report.ensemble import render_pairwise_markdown_with_ci
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown_with_ci
from benchmarks.report.hpo_uplift import render_hpo_uplift_markdown_with_ci
from benchmarks.report.raw_loss import render_leaderboard_markdown_with_ci
from benchmarks.report.training_time import render_training_time_markdown_with_ci

_SECTION = "### Per-fold CIs"
_BCA_SECTION = "### Bootstrap CI method"


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


# =============================================================================
# Fixture helpers (one per RollupRow type)
# =============================================================================


def _raw_loss_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_name": ["fake_binary"] * 4,
            "model_name": ["m1"] * 4,
            "task_type": ["binary"] * 4,
            "seed": [0, 0, 1, 1],
            "fold_index": [0, 1, 0, 1],
            "log_loss": [0.20, 0.22, 0.21, 0.23],
            "wall_seconds": [1.0, 1.1, 1.2, 1.3],
            "skipped_reason": [None, None, None, None],
        }
    )


def _raw_loss_row(
    *,
    model_name: str = "m1",
    bootstrap_ci_fallback_reason: str | None = None,
    per_fold_cis: list[FoldCI] | None = None,
) -> RollupRow:
    return RollupRow(
        dataset_name="fake_binary",
        model_name=model_name,
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
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        per_fold_cis=per_fold_cis,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="d" * 64,
    )


def _pairwise_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for s in (0, 1):
        rows.append(
            {
                "library_git_sha": "0" * 40,
                "run_id": "b25-test",
                "started_at_utc": "2026-05-30T00:00:00+00:00",
                "dataset_name": "ds_one",
                "model_a": "model_a",
                "model_b": "model_b",
                "seed": s,
                "fold_index": 0,
                "task_type": "binary",
                "skipped_reason": None,
                "n_samples": 100,
                "n11": 40,
                "n10": 10,
                "n01": 15,
                "n00": 35,
                "yule_q": 0.7,
                "phi": 0.5,
                "disagreement_rate": 0.25,
                "double_fault_rate": 0.1,
                "pearson_pred_corr": 0.6,
                "spearman_pred_corr": 0.55,
                "pearson_error_corr": 0.30,
                "complementarity_score": 0.40,
                "model_a_log_loss": 0.45,
                "model_b_log_loss": 0.50,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_row(
    *,
    model_a: str = "model_a",
    model_b: str = "model_b",
    bootstrap_ci_fallback_reason: str | None = None,
    per_fold_cis: list[FoldCI] | None = None,
) -> PairwiseRollupRow:
    return PairwiseRollupRow(
        dataset_name="ds_one",
        model_a=model_a,
        model_b=model_b,
        task_type="binary",
        primary_metric="complementarity_score",
        n_seeds=2,
        n_cells_evaluated=2,
        n_skipped_cells=0,
        primary_metric_mean=0.40,
        primary_metric_ci_lo=0.35,
        primary_metric_ci_hi=0.45,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        per_fold_cis=per_fold_cis,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="a" * 64,
    )


def _training_time_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "ds_one",
                "model_name": "model_a",
                "task_type": "binary",
                "seed": s,
                "fold_index": 0,
                "skipped_reason": None,
                "wall_seconds": 12.5,
                "peak_rss_bytes": 1_000_000.0,
                "peak_cuda_bytes": None,
                "hardware_tier": "cpu",
                "log_loss": 0.30,
            }
            for s in (0, 1)
        ]
    )


def _training_time_row(
    *,
    model_name: str = "model_a",
    bootstrap_ci_fallback_reason: str | None = None,
    per_fold_cis: list[FoldCI] | None = None,
) -> TrainingTimeRollupRow:
    return TrainingTimeRollupRow(
        dataset_name="ds_one",
        model_name=model_name,
        hardware_tier="cpu",
        task_type="binary",
        primary_metric="wall_seconds",
        n_seeds=2,
        n_cells_evaluated=2,
        n_skipped_cells=0,
        primary_metric_mean=12.5,
        primary_metric_ci_lo=10.0,
        primary_metric_ci_hi=15.0,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        per_fold_cis=per_fold_cis,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="b" * 64,
    )


def _hpo_uplift_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "ds_one",
                "model_name": "model_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "variant": "default",
                "skipped_reason": None,
                "log_loss": 0.50,
                "hpo_search_space_size": 5,
                "hpo_tier": "smoke",
            },
            {
                "dataset_name": "ds_one",
                "model_name": "model_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "variant": "tuned",
                "skipped_reason": None,
                "log_loss": 0.30,
                "hpo_search_space_size": 5,
                "hpo_tier": "smoke",
                "hpo_n_trials_completed": 10,
                "hpo_time_to_best_seconds": 5.0,
            },
        ]
    )


def _hpo_uplift_row(
    *,
    model_name: str = "model_a",
    bootstrap_ci_fallback_reason: str | None = None,
    per_fold_cis: list[FoldCI] | None = None,
) -> HPOUpliftRollupRow:
    return HPOUpliftRollupRow(
        dataset_name="ds_one",
        model_name=model_name,
        task_type="binary",
        primary_metric="delta",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=1,
        n_cells_paired=2,
        n_skipped_cells=0,
        primary_metric_mean=0.20,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        per_fold_cis=per_fold_cis,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="c" * 64,
    )


def _ensemble_lift_result(dataset_names: list[str] | None = None) -> EnsembleLiftExperimentResult:
    names = dataset_names or ["ds_one"]
    return EnsembleLiftExperimentResult(
        run_id="b25-test",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=tuple(
            PerDatasetLift(
                dataset_name=name,
                task_type="binary",
                primary_loss_column="log_loss",
                n_cells_paired=2,
                loss_gbm_only_mean=0.50,
                loss_gbm_plus_seq_mean=0.30,
                delta_loss_mean=0.20,
                delta_loss_std=0.01,
                oracle_loss_mean=0.25,
                oracle_delta_loss_mean=0.25,
            )
            for name in names
        ),
        wilcoxon=WilcoxonResult(
            statistic=1.0,
            p_value=0.10,
            holm_adjusted_p_value=0.10,
            n_datasets=len(names),
            family_size=len(names),
        ),
    )


def _ensemble_lift_row(
    *,
    dataset_name: str = "ds_one",
    bootstrap_ci_fallback_reason: str | None = None,
    per_fold_cis: list[FoldCI] | None = None,
) -> EnsembleLiftRollupRow:
    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=1,
        n_cells_paired=2,
        n_pair_grid=2,
        n_oracle_cells_paired=2,
        n_skipped_cells=0,
        primary_metric_mean=0.20,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=0.10,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        per_fold_cis=per_fold_cis,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="e" * 64,
    )


# =============================================================================
# B25.3.1 Shared helper (R-B25-1 core)
# =============================================================================


def test_per_fold_cis_footnote_section_heading() -> None:
    row = _raw_loss_row(per_fold_cis=[_fold_ci()])
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert _SECTION in md


def test_per_fold_cis_footnote_unequal_lengths_raises() -> None:
    with pytest.raises(ValueError, match=r"equal length"):
        render_per_fold_cis_footnote(
            [],
            group_columns=("dataset_name", "model_name"),
            header_labels=("Dataset",),
        )


def test_per_fold_cis_footnote_long_reason_exact_boundary() -> None:
    long_reason = "x" * 121
    row = _raw_loss_row(per_fold_cis=[_fold_ci(ci_fallback_reason=long_reason)])
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert "x" * 117 + "..." in md
    assert long_reason not in md


def test_per_fold_cis_footnote_empty_input_returns_empty_string() -> None:
    result = render_per_fold_cis_footnote(
        [], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert result == ""


def test_per_fold_cis_footnote_120_char_reason_not_truncated() -> None:
    boundary_reason = "y" * 120
    row = _raw_loss_row(per_fold_cis=[_fold_ci(ci_fallback_reason=boundary_reason)])
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert boundary_reason in md


def test_per_fold_cis_footnote_null_metric_renders_dash() -> None:
    """qa-R1-build-I1 closure: pin the EXACT column position of
    the three null metric cells. Splitting the fold row on
    ' | ' and asserting cell-by-cell catches a mutation that
    nulls a different column (e.g. ci_method instead of
    metric_ci_lo) while still rendering three consecutive
    dashes at a different position."""
    row = _raw_loss_row(
        per_fold_cis=[
            _fold_ci(metric_mean=None, metric_ci_lo=None, metric_ci_hi=None)
        ]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    # Find the fold row (the only non-header pipe-bounded line in the section).
    section_start = md.find(_SECTION)
    block = md[section_start:]
    fold_line = next(
        ln
        for ln in block.split("\n")
        if ln.startswith("| fake_binary |") and "| 0 |" in ln
    )
    cells = [c.strip() for c in fold_line.strip("| ").split(" | ")]
    # Expected columns: dataset | fold | metric_mean | metric_ci_lo |
    # metric_ci_hi | ci_method | ci_fallback_reason
    assert cells == ["fake_binary", "0", "-", "-", "-", "bca", "-"]


def test_per_fold_cis_footnote_none_fallback_reason_renders_dash_with_non_null_metrics() -> None:
    """qa-R1-build-I2 closure: pin that ci_fallback_reason=None
    renders as `-` independently from the null-metric path.
    Test #6 covers both Nones together; this isolates the
    `ci_fallback_reason is None` branch when metric values are
    populated."""
    row = _raw_loss_row(
        per_fold_cis=[
            _fold_ci(
                metric_mean=0.215,
                metric_ci_lo=0.200,
                metric_ci_hi=0.230,
                ci_fallback_reason=None,
            )
        ]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    section_start = md.find(_SECTION)
    block = md[section_start:]
    fold_line = next(
        ln
        for ln in block.split("\n")
        if ln.startswith("| fake_binary |") and "| 0 |" in ln
    )
    cells = [c.strip() for c in fold_line.strip("| ").split(" | ")]
    assert cells == ["fake_binary", "0", "0.2150", "0.2000", "0.2300", "bca", "-"]


def test_per_fold_cis_footnote_float_format_4_decimals() -> None:
    row = _raw_loss_row(per_fold_cis=[_fold_ci(metric_mean=0.21567)])
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert "| 0.2157 |" in md


def test_per_fold_cis_footnote_sorts_rows_by_group_columns_first_key() -> None:
    rows = [
        _raw_loss_row(model_name="z_late", per_fold_cis=[_fold_ci()]),
        _raw_loss_row(model_name="a_early", per_fold_cis=[_fold_ci()]),
    ]
    # Use model_name (the differing field) as the sort key.
    md = render_per_fold_cis_footnote(
        rows,
        group_columns=("model_name",),
        header_labels=("Model",),
    )
    early_idx = md.find("| a_early |")
    late_idx = md.find("| z_late |")
    assert early_idx >= 0
    assert late_idx >= 0
    assert early_idx < late_idx


def test_per_fold_cis_footnote_sorts_fold_rows_by_fold_index_ascending_when_input_unsorted() -> None:
    """arch-R1-C1 + qa-R1-I1 closure: out-of-order fold input
    [2, 0, 1] must render ascending 0, 1, 2."""
    row = _raw_loss_row(
        per_fold_cis=[
            _fold_ci(fold_index=2),
            _fold_ci(fold_index=0),
            _fold_ci(fold_index=1),
        ]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    idx_0 = md.find("| 0 |")
    idx_1 = md.find("| 1 |")
    idx_2 = md.find("| 2 |")
    assert idx_0 >= 0
    assert idx_1 >= 0
    assert idx_2 >= 0
    assert idx_0 < idx_1 < idx_2


def test_per_fold_cis_footnote_cell_data_reads_correct_six_fields() -> None:
    """qa-R1-C2 closure: assert all 6 rendered FoldCI fields
    appear in the rendered output. Uses non-None
    ci_fallback_reason so the cell is the literal token, not
    the `-` placeholder (which would mask a hardwired-`-` bug)."""
    row = _raw_loss_row(
        per_fold_cis=[
            _fold_ci(
                fold_index=7,
                metric_mean=0.215,
                metric_ci_lo=0.200,
                metric_ci_hi=0.230,
                ci_method="bca",
                ci_fallback_reason="p0_at_edge",
            )
        ]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert "| 7 |" in md
    assert "0.2150" in md
    assert "0.2000" in md
    assert "0.2300" in md
    assert "| bca |" in md
    assert "| p0_at_edge |" in md


def test_per_fold_cis_footnote_does_not_surface_n_seeds_or_n_entities() -> None:
    """qa-R1-C2 + qa-R1-I2 closure: n_seeds and n_entities are
    audit-only fields and must NOT appear in the rendered
    output. Uses fixture values 99 and 88 (chosen to not
    collide with any other rendered numeric)."""
    row = _raw_loss_row(
        per_fold_cis=[_fold_ci(n_seeds=99, n_entities=88)]
    )
    md = render_per_fold_cis_footnote(
        [row], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    section_idx = md.find(_SECTION)
    assert section_idx >= 0
    per_fold_block = md[section_idx:]
    assert "n_seeds" not in per_fold_block
    assert "n_entities" not in per_fold_block
    assert "99" not in per_fold_block
    assert "88" not in per_fold_block


def test_per_fold_cis_footnote_ci_fallback_reason_source_binding() -> None:
    """qa-R1-C1 closure: two fixtures differing only in
    `ci_fallback_reason`; assert each rendered output contains
    its literal AND does NOT contain the other. Kills a wiring
    bug that hardwires the cell value."""
    row_p0 = _raw_loss_row(per_fold_cis=[_fold_ci(ci_fallback_reason="p0_at_edge")])
    row_a = _raw_loss_row(per_fold_cis=[_fold_ci(ci_fallback_reason="a_overshoot")])
    md_p0 = render_per_fold_cis_footnote(
        [row_p0], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    md_a = render_per_fold_cis_footnote(
        [row_a], group_columns=("dataset_name",), header_labels=("Dataset",)
    )
    assert "p0_at_edge" in md_p0
    assert "a_overshoot" not in md_p0
    assert "a_overshoot" in md_a
    assert "p0_at_edge" not in md_a


# =============================================================================
# B25.3.2 Per-renderer present emission (R-B25-1 parity)
# =============================================================================


def test_raw_loss_renderer_emits_per_fold_cis_when_present() -> None:
    manifest = _raw_loss_manifest()
    rollup = [_raw_loss_row(per_fold_cis=[_fold_ci()])]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert (
        "| Dataset | Model | fold | metric_mean | metric_ci_lo | "
        "metric_ci_hi | ci_method | ci_fallback_reason |" in md
    )


def test_pairwise_renderer_emits_per_fold_cis_when_present() -> None:
    manifest = _pairwise_manifest()
    rollup = [_pairwise_row(per_fold_cis=[_fold_ci()])]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert (
        "| Dataset | Model A | Model B | fold | metric_mean | metric_ci_lo | "
        "metric_ci_hi | ci_method | ci_fallback_reason |" in md
    )


def test_training_time_renderer_emits_per_fold_cis_when_present() -> None:
    manifest = _training_time_manifest()
    rollup = [_training_time_row(per_fold_cis=[_fold_ci()])]
    md = render_training_time_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert (
        "| Dataset | Model | Hardware tier | fold | metric_mean | metric_ci_lo | "
        "metric_ci_hi | ci_method | ci_fallback_reason |" in md
    )


def test_hpo_uplift_renderer_emits_per_fold_cis_when_present() -> None:
    manifest = _hpo_uplift_manifest()
    rollup = [_hpo_uplift_row(per_fold_cis=[_fold_ci()])]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert (
        "| Dataset | Model | fold | metric_mean | metric_ci_lo | "
        "metric_ci_hi | ci_method | ci_fallback_reason |" in md
    )


def test_ensemble_lift_renderer_emits_per_fold_cis_when_present() -> None:
    result = _ensemble_lift_result()
    rollup = [_ensemble_lift_row(per_fold_cis=[_fold_ci()])]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert _SECTION in md
    assert (
        "| Dataset | fold | metric_mean | metric_ci_lo | "
        "metric_ci_hi | ci_method | ci_fallback_reason |" in md
    )


# =============================================================================
# B25.3.3 Per-renderer silent-when-absent
# =============================================================================


def test_raw_loss_renderer_silent_when_per_fold_absent() -> None:
    manifest = _raw_loss_manifest()
    rollup = [_raw_loss_row(per_fold_cis=None)]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_pairwise_renderer_silent_when_per_fold_absent() -> None:
    manifest = _pairwise_manifest()
    rollup = [_pairwise_row(per_fold_cis=None)]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_training_time_renderer_silent_when_per_fold_absent() -> None:
    manifest = _training_time_manifest()
    rollup = [_training_time_row(per_fold_cis=None)]
    md = render_training_time_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_hpo_uplift_renderer_silent_when_per_fold_absent() -> None:
    manifest = _hpo_uplift_manifest()
    rollup = [_hpo_uplift_row(per_fold_cis=None)]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_ensemble_lift_renderer_silent_when_per_fold_absent() -> None:
    result = _ensemble_lift_result()
    rollup = [_ensemble_lift_row(per_fold_cis=None)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert _SECTION not in md


# =============================================================================
# B25.3.4 Mixed-rollup (parametrized over 5 renderers)
# =============================================================================


def _render_raw_loss_mixed_per_fold() -> str:
    manifest = _raw_loss_manifest()
    rollup = [
        _raw_loss_row(model_name="m1", per_fold_cis=[_fold_ci()]),
        _raw_loss_row(model_name="m2", per_fold_cis=None),
    ]
    return render_leaderboard_markdown_with_ci(manifest, rollup)


def _render_pairwise_mixed_per_fold() -> str:
    manifest = _pairwise_manifest()
    rollup = [
        _pairwise_row(model_a="model_a", model_b="model_b", per_fold_cis=[_fold_ci()]),
        _pairwise_row(model_a="model_c", model_b="model_d", per_fold_cis=None),
    ]
    return render_pairwise_markdown_with_ci(manifest, rollup)


def _render_training_time_mixed_per_fold() -> str:
    manifest = _training_time_manifest()
    rollup = [
        _training_time_row(model_name="model_a", per_fold_cis=[_fold_ci()]),
        _training_time_row(model_name="model_b", per_fold_cis=None),
    ]
    return render_training_time_markdown_with_ci(manifest, rollup)


def _render_hpo_uplift_mixed_per_fold() -> str:
    manifest = _hpo_uplift_manifest()
    rollup = [
        _hpo_uplift_row(model_name="model_a", per_fold_cis=[_fold_ci()]),
        _hpo_uplift_row(model_name="model_b", per_fold_cis=None),
    ]
    return render_hpo_uplift_markdown_with_ci(manifest, rollup)


def _render_ensemble_lift_mixed_per_fold() -> str:
    result = _ensemble_lift_result(["ds_one", "ds_two"])
    rollup = [
        _ensemble_lift_row(dataset_name="ds_one", per_fold_cis=[_fold_ci()]),
        _ensemble_lift_row(dataset_name="ds_two", per_fold_cis=None),
    ]
    return render_ensemble_lift_markdown_with_ci(result, rollup)


@pytest.mark.parametrize(
    ("render_fn", "present_id", "absent_id"),
    [
        (_render_raw_loss_mixed_per_fold, "| m1 |", "| m2 |"),
        (_render_pairwise_mixed_per_fold, "| model_a | model_b |", "| model_c | model_d |"),
        (_render_training_time_mixed_per_fold, "| model_a |", "| model_b |"),
        (_render_hpo_uplift_mixed_per_fold, "| model_a |", "| model_b |"),
        (_render_ensemble_lift_mixed_per_fold, "| ds_one |", "| ds_two |"),
    ],
)
def test_renderer_emits_only_per_fold_rows_in_mixed_rollup(
    render_fn: Callable[[], str], present_id: str, absent_id: str
) -> None:
    md = render_fn()
    section_idx = md.find(_SECTION)
    assert section_idx >= 0, f"Per-fold section missing from output: {md[:400]!r}"
    per_fold_block = md[section_idx:]
    assert present_id in per_fold_block
    assert absent_id not in per_fold_block


# =============================================================================
# B25.3.5 Empty list edge case
# =============================================================================


def test_renderer_treats_empty_per_fold_cis_list_as_absent() -> None:
    """per_fold_cis=[] (empty list, not None) should NOT trigger
    the footnote per the caller pre-filter contract."""
    manifest = _raw_loss_manifest()
    rollup = [_raw_loss_row(per_fold_cis=[])]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


# =============================================================================
# B25.3.6 Section ordering (qa-R1-I3 closure)
# =============================================================================


def _render_raw_loss_both_sections() -> str:
    manifest = _raw_loss_manifest()
    rollup = [
        _raw_loss_row(
            bootstrap_ci_fallback_reason="p0_at_edge",
            per_fold_cis=[_fold_ci()],
        )
    ]
    return render_leaderboard_markdown_with_ci(manifest, rollup)


def _render_pairwise_both_sections() -> str:
    manifest = _pairwise_manifest()
    rollup = [
        _pairwise_row(
            bootstrap_ci_fallback_reason="p0_at_edge",
            per_fold_cis=[_fold_ci()],
        )
    ]
    return render_pairwise_markdown_with_ci(manifest, rollup)


def _render_training_time_both_sections() -> str:
    manifest = _training_time_manifest()
    rollup = [
        _training_time_row(
            bootstrap_ci_fallback_reason="p0_at_edge",
            per_fold_cis=[_fold_ci()],
        )
    ]
    return render_training_time_markdown_with_ci(manifest, rollup)


def _render_hpo_uplift_both_sections() -> str:
    manifest = _hpo_uplift_manifest()
    rollup = [
        _hpo_uplift_row(
            bootstrap_ci_fallback_reason="p0_at_edge",
            per_fold_cis=[_fold_ci()],
        )
    ]
    return render_hpo_uplift_markdown_with_ci(manifest, rollup)


def _render_ensemble_lift_both_sections() -> str:
    result = _ensemble_lift_result()
    rollup = [
        _ensemble_lift_row(
            bootstrap_ci_fallback_reason="p0_at_edge",
            per_fold_cis=[_fold_ci()],
        )
    ]
    return render_ensemble_lift_markdown_with_ci(result, rollup)


@pytest.mark.parametrize(
    "render_fn",
    [
        _render_raw_loss_both_sections,
        _render_pairwise_both_sections,
        _render_training_time_both_sections,
        _render_hpo_uplift_both_sections,
        _render_ensemble_lift_both_sections,
    ],
)
def test_per_fold_cis_footnote_lands_after_bca_health_footnote(
    render_fn: Callable[[], str],
) -> None:
    """qa-R1-I3 closure: per-fold section must land AFTER the BCa
    health section. Catches a mis-ordered insertion."""
    md = render_fn()
    bca_idx = md.find(_BCA_SECTION)
    per_fold_idx = md.find(_SECTION)
    assert bca_idx >= 0, f"BCa health section missing: {md[:400]!r}"
    assert per_fold_idx >= 0, f"Per-fold section missing: {md[:400]!r}"
    assert bca_idx < per_fold_idx
