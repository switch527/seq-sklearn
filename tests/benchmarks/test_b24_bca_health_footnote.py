"""Phase B24 BCa health footnote bundle tests (22 named; 26 collected).

Closes the D-B21.1 (per-renderer Bootstrap CI method footnote)
and D-B23.1 (oracle partial-coverage footnote ci_method +
oracle_fallback_reason columns) deferrals.

Test layout:
- B24.4.1: shared helper (#1-#4b)
- B24.4.2: per-renderer present-fallback emission (#5-#9)
- B24.4.3: per-renderer silent-when-no-fallback (#10-#14)
- B24.4.4: parametrized mixed-rollup omission (#15)
- B24.4.5: oracle footnote extension (#16-#20)
"""

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
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
from benchmarks.report._bootstrap_render import render_bca_health_footnote
from benchmarks.report.ensemble import render_pairwise_markdown_with_ci
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown_with_ci
from benchmarks.report.hpo_uplift import render_hpo_uplift_markdown_with_ci
from benchmarks.report.raw_loss import render_leaderboard_markdown_with_ci
from benchmarks.report.training_time import render_training_time_markdown_with_ci

_SECTION = "### Bootstrap CI method"


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
    bootstrap_ci_method: str = "bca",
    bootstrap_ci_fallback_reason: str | None = None,
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
        bootstrap_ci_method=bootstrap_ci_method,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
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
                "run_id": "b24-test",
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
    bootstrap_ci_method: str = "bca",
    bootstrap_ci_fallback_reason: str | None = None,
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
        bootstrap_ci_method=bootstrap_ci_method,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
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
    bootstrap_ci_method: str = "bca",
    bootstrap_ci_fallback_reason: str | None = None,
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
        bootstrap_ci_method=bootstrap_ci_method,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
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
    bootstrap_ci_method: str = "bca",
    bootstrap_ci_fallback_reason: str | None = None,
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
        bootstrap_ci_method=bootstrap_ci_method,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="c" * 64,
    )


def _ensemble_lift_result(dataset_names: list[str] | None = None) -> EnsembleLiftExperimentResult:
    names = dataset_names or ["ds_one"]
    return EnsembleLiftExperimentResult(
        run_id="b24-test",
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
    n_cells_paired: int = 2,
    n_pair_grid: int = 2,
    n_oracle_cells_paired: int = 2,
    bootstrap_ci_method: str = "bca",
    bootstrap_ci_fallback_reason: str | None = None,
    bootstrap_oracle_ci_fallback_reason: str | None = None,
) -> EnsembleLiftRollupRow:
    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=1,
        n_cells_paired=n_cells_paired,
        n_pair_grid=n_pair_grid,
        n_oracle_cells_paired=n_oracle_cells_paired,
        n_skipped_cells=0,
        primary_metric_mean=0.20,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=0.10,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_ci_method=bootstrap_ci_method,
        bootstrap_ci_fallback_reason=bootstrap_ci_fallback_reason,
        bootstrap_oracle_ci_fallback_reason=bootstrap_oracle_ci_fallback_reason,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="e" * 64,
    )


# =============================================================================
# B24.4.1 Shared helper tests (R-B24-1 core)
# =============================================================================


def test_bca_health_footnote_section_heading() -> None:
    """Helper emits the canonical section heading on non-empty input."""
    row = _raw_loss_row(bootstrap_ci_fallback_reason="p0_at_edge")
    md = render_bca_health_footnote(
        [row],
        group_columns=("dataset_name", "model_name"),
        header_labels=("Dataset", "Model"),
    )
    assert _SECTION in md


def test_bca_health_footnote_unequal_lengths_raises() -> None:
    """Mismatched group_columns and header_labels raise ValueError."""
    with pytest.raises(ValueError, match=r"equal length"):
        render_bca_health_footnote(
            [],
            group_columns=("dataset_name", "model_name"),
            header_labels=("Dataset",),
        )


def test_bca_health_footnote_long_reason_exact_boundary() -> None:
    """qa-R1-C2 closure: 121-char reason truncates to
    `reason[:117] + "..."`. Positive substring asserts the
    exact truncation prefix; negative asserts the full reason
    is NOT present. Kills mutations that truncate at 100 chars
    or omit the ellipsis."""
    long_reason = "x" * 121
    row = _raw_loss_row(bootstrap_ci_fallback_reason=long_reason)
    md = render_bca_health_footnote(
        [row],
        group_columns=("dataset_name",),
        header_labels=("Dataset",),
    )
    assert "x" * 117 + "..." in md
    assert long_reason not in md


def test_bca_health_footnote_empty_input_returns_empty_string() -> None:
    """qa-R1-I1 closure: empty input returns "" (mirrors the
    oracle partial-coverage footnote's early-return)."""
    result = render_bca_health_footnote(
        [],
        group_columns=("dataset_name",),
        header_labels=("Dataset",),
    )
    assert result == ""


def test_bca_health_footnote_cell_data_reads_correct_fields() -> None:
    """qa-R2-I1 closure: assert the actual values of
    `bootstrap_ci_method` and `bootstrap_ci_fallback_reason`
    appear in the rendered cells. Kills a getattr-wrong-attr
    bug that would silently render empty cells while still
    passing the heading + header + truncation tests."""
    row = _raw_loss_row(
        bootstrap_ci_method="bca",
        bootstrap_ci_fallback_reason="p0_at_edge",
    )
    md = render_bca_health_footnote(
        [row],
        group_columns=("dataset_name",),
        header_labels=("Dataset",),
    )
    assert "bca" in md
    assert "p0_at_edge" in md


def test_bca_health_footnote_120_char_reason_not_truncated() -> None:
    """qa-R2-N1 closure: 120-char reason passes through
    unmodified. Pins the `> 120` guard against an off-by-one
    `>= 120` mutation that tests using 121-char input cannot
    catch alone."""
    boundary_reason = "y" * 120
    row = _raw_loss_row(bootstrap_ci_fallback_reason=boundary_reason)
    md = render_bca_health_footnote(
        [row],
        group_columns=("dataset_name",),
        header_labels=("Dataset",),
    )
    assert boundary_reason in md


# =============================================================================
# B24.4.2 Per-renderer present-fallback emission (R-B24-1 parity)
# =============================================================================


def test_raw_loss_renderer_emits_bca_health_when_fallback_present() -> None:
    manifest = _raw_loss_manifest()
    rollup = [_raw_loss_row(bootstrap_ci_fallback_reason="p0_at_edge")]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert "| Dataset | Model | ci_method | fallback_reason |" in md


def test_pairwise_renderer_emits_bca_health_when_fallback_present() -> None:
    manifest = _pairwise_manifest()
    rollup = [_pairwise_row(bootstrap_ci_fallback_reason="a_overshoot")]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert "| Dataset | Model A | Model B | ci_method | fallback_reason |" in md


def test_training_time_renderer_emits_bca_health_when_fallback_present() -> None:
    manifest = _training_time_manifest()
    rollup = [_training_time_row(bootstrap_ci_fallback_reason="p0_at_edge")]
    md = render_training_time_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert "| Dataset | Model | Hardware tier | ci_method | fallback_reason |" in md


def test_hpo_uplift_renderer_emits_bca_health_when_fallback_present() -> None:
    manifest = _hpo_uplift_manifest()
    rollup = [_hpo_uplift_row(bootstrap_ci_fallback_reason="a_overshoot")]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert _SECTION in md
    assert "| Dataset | Model | ci_method | fallback_reason |" in md


def test_ensemble_lift_renderer_emits_bca_health_when_main_fallback_present() -> None:
    result = _ensemble_lift_result()
    rollup = [_ensemble_lift_row(bootstrap_ci_fallback_reason="p0_at_edge")]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert _SECTION in md
    assert "| Dataset | ci_method | fallback_reason |" in md


# =============================================================================
# B24.4.3 Per-renderer silent-when-no-fallback (qa-R1-C1 closure)
# =============================================================================


def test_raw_loss_renderer_silent_when_no_fallback() -> None:
    manifest = _raw_loss_manifest()
    rollup = [_raw_loss_row(bootstrap_ci_fallback_reason=None)]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_pairwise_renderer_silent_when_no_fallback() -> None:
    manifest = _pairwise_manifest()
    rollup = [_pairwise_row(bootstrap_ci_fallback_reason=None)]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_training_time_renderer_silent_when_no_fallback() -> None:
    manifest = _training_time_manifest()
    rollup = [_training_time_row(bootstrap_ci_fallback_reason=None)]
    md = render_training_time_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_hpo_uplift_renderer_silent_when_no_fallback() -> None:
    manifest = _hpo_uplift_manifest()
    rollup = [_hpo_uplift_row(bootstrap_ci_fallback_reason=None)]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert _SECTION not in md


def test_ensemble_lift_renderer_silent_when_main_fallback_absent() -> None:
    result = _ensemble_lift_result()
    rollup = [_ensemble_lift_row(bootstrap_ci_fallback_reason=None)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert _SECTION not in md


# =============================================================================
# B24.4.4 Mixed-rollup per renderer (qa-R1-I2 closure)
# =============================================================================


def _render_raw_loss_mixed() -> str:
    manifest = _raw_loss_manifest()
    rollup = [
        _raw_loss_row(model_name="m1", bootstrap_ci_fallback_reason="p0_at_edge"),
        _raw_loss_row(model_name="m2", bootstrap_ci_fallback_reason=None),
    ]
    return render_leaderboard_markdown_with_ci(manifest, rollup)


def _render_pairwise_mixed() -> str:
    manifest = _pairwise_manifest()
    rollup = [
        _pairwise_row(model_a="model_a", model_b="model_b", bootstrap_ci_fallback_reason="p0_at_edge"),
        _pairwise_row(model_a="model_c", model_b="model_d", bootstrap_ci_fallback_reason=None),
    ]
    return render_pairwise_markdown_with_ci(manifest, rollup)


def _render_training_time_mixed() -> str:
    manifest = _training_time_manifest()
    rollup = [
        _training_time_row(model_name="model_a", bootstrap_ci_fallback_reason="p0_at_edge"),
        _training_time_row(model_name="model_b", bootstrap_ci_fallback_reason=None),
    ]
    return render_training_time_markdown_with_ci(manifest, rollup)


def _render_hpo_uplift_mixed() -> str:
    manifest = _hpo_uplift_manifest()
    rollup = [
        _hpo_uplift_row(model_name="model_a", bootstrap_ci_fallback_reason="p0_at_edge"),
        _hpo_uplift_row(model_name="model_b", bootstrap_ci_fallback_reason=None),
    ]
    return render_hpo_uplift_markdown_with_ci(manifest, rollup)


def _render_ensemble_lift_mixed() -> str:
    result = _ensemble_lift_result(["ds_one", "ds_two"])
    rollup = [
        _ensemble_lift_row(dataset_name="ds_one", bootstrap_ci_fallback_reason="p0_at_edge"),
        _ensemble_lift_row(dataset_name="ds_two", bootstrap_ci_fallback_reason=None),
    ]
    return render_ensemble_lift_markdown_with_ci(result, rollup)


@pytest.mark.parametrize(
    ("render_fn", "present_id", "absent_id"),
    [
        (_render_raw_loss_mixed, "| m1 |", "| m2 |"),
        (_render_pairwise_mixed, "| model_a | model_b |", "| model_c | model_d |"),
        (_render_training_time_mixed, "| model_a |", "| model_b |"),
        (_render_hpo_uplift_mixed, "| model_a |", "| model_b |"),
        (_render_ensemble_lift_mixed, "| ds_one |", "| ds_two |"),
    ],
)
def test_renderer_emits_only_fallback_rows_in_mixed_rollup(
    render_fn: object, present_id: str, absent_id: str
) -> None:
    """qa-R1-I2 closure: a wrong-field pre-filter (e.g. on
    `bootstrap_skipped_reason` instead of
    `bootstrap_ci_fallback_reason`) would either include or
    exclude both rows; this test catches both directions per
    renderer."""
    md = render_fn()  # type: ignore[operator]
    section_idx = md.find(_SECTION)
    assert section_idx >= 0, f"BCa health section missing from output: {md[:400]!r}"
    bca_block = md[section_idx:]
    assert present_id in bca_block
    assert absent_id not in bca_block


# =============================================================================
# B24.4.5 Oracle footnote extension (R-B24-2)
# =============================================================================


def test_oracle_partial_coverage_footnote_includes_ci_method_column() -> None:
    """The oracle partial-coverage footnote now carries the
    bootstrap_ci_method value in a `ci_method` column."""
    result = _ensemble_lift_result()
    rollup = [
        _ensemble_lift_row(
            n_cells_paired=4,
            n_pair_grid=4,
            n_oracle_cells_paired=3,
            bootstrap_ci_method="bca",
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Oracle partial-coverage footnotes" in md
    assert "| bca |" in md


def test_oracle_partial_coverage_footnote_includes_fallback_column_when_set() -> None:
    """The `oracle_fallback_reason` column surfaces the value
    of `bootstrap_oracle_ci_fallback_reason`."""
    result = _ensemble_lift_result()
    rollup = [
        _ensemble_lift_row(
            n_pair_grid=4,
            n_oracle_cells_paired=2,
            bootstrap_oracle_ci_fallback_reason="a_overshoot",
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "| a_overshoot |" in md


def test_oracle_partial_coverage_footnote_dash_when_oracle_fallback_none() -> None:
    """When `bootstrap_oracle_ci_fallback_reason` is None, the
    column reads `-` (literal hyphen) so a reader distinguishes
    'no fallback' from 'field absent'."""
    result = _ensemble_lift_result()
    rollup = [
        _ensemble_lift_row(
            n_pair_grid=4,
            n_oracle_cells_paired=2,
            bootstrap_oracle_ci_fallback_reason=None,
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "| - |" in md


def test_oracle_partial_coverage_footnote_long_oracle_reason_exact_boundary() -> None:
    """qa-R1-C2 closure: same boundary pin as test #3, applied
    to the oracle reason path. 121-char input truncates to
    `reason[:117] + "..."`; full input is NOT in output."""
    long_reason = "z" * 121
    result = _ensemble_lift_result()
    rollup = [
        _ensemble_lift_row(
            n_pair_grid=4,
            n_oracle_cells_paired=2,
            bootstrap_oracle_ci_fallback_reason=long_reason,
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "z" * 117 + "..." in md
    assert long_reason not in md


def test_oracle_partial_coverage_footnote_ci_method_column_reads_bootstrap_ci_method_field() -> None:
    """qa-R1-I3 closure: mutate `bootstrap_ci_method` between
    two fixtures with otherwise identical partial-oracle-coverage
    shape; the rendered `ci_method` column in the oracle
    footnote must change. A wiring bug reading a fixed string
    or the wrong attribute survives the individual value tests
    (#16-#18) but fails this source-binding pin."""
    result = _ensemble_lift_result()
    rollup_percentile = [
        _ensemble_lift_row(
            n_pair_grid=4,
            n_oracle_cells_paired=2,
            bootstrap_ci_method="percentile",
        )
    ]
    rollup_bca = [
        _ensemble_lift_row(
            n_pair_grid=4,
            n_oracle_cells_paired=2,
            bootstrap_ci_method="bca",
        )
    ]
    md_percentile = render_ensemble_lift_markdown_with_ci(result, rollup_percentile)
    md_bca = render_ensemble_lift_markdown_with_ci(result, rollup_bca)
    oracle_section_p = md_percentile[md_percentile.find("Oracle partial-coverage footnotes") :]
    oracle_section_b = md_bca[md_bca.find("Oracle partial-coverage footnotes") :]
    assert "| percentile |" in oracle_section_p
    assert "| percentile |" not in oracle_section_b
    assert "| bca |" in oracle_section_b
