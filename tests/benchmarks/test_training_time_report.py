"""Phase B7 training-time report tests (B6.3).

The B5 manifest captures `wall_seconds`, `peak_rss_bytes`, and
`peak_cuda_bytes` on every successful row. B7's report aggregates
these into a per-(dataset, model, hardware_tier) table. These tests
exercise the aggregator + renderer directly on hand-crafted
manifest DataFrames; the e2e test in
`test_training_time_experiment.py` drives the full B5 -> B7 flow.
"""

import math
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    TrainingTimeRollupRow as TrainingTimeRollupRowType,
)
from benchmarks.manifest import results_dir
from benchmarks.report.training_time import (
    TrainingTimeSummary,
    _format_bytes,  # pyright: ignore[reportPrivateUsage]
    _format_value,  # pyright: ignore[reportPrivateUsage]
    _render_dataset_block,  # pyright: ignore[reportPrivateUsage]
    aggregate_training_time,
    render_from_dir,
    render_training_time_markdown,
)


def _row(
    *,
    dataset_name: str = "ds_a",
    model_name: str = "model_x",
    hardware_tier: str = "cpu",
    task_type: str = "binary",
    seed: int = 0,
    fold_index: int = 0,
    skipped_reason: str | None = None,
    wall_seconds: float | None = 1.0,
    peak_rss_bytes: float | None = 1_000_000.0,
    peak_cuda_bytes: float | None = None,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "hardware_tier": hardware_tier,
        "task_type": task_type,
        "seed": seed,
        "fold_index": fold_index,
        "skipped_reason": skipped_reason,
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_cuda_bytes": peak_cuda_bytes,
    }


# --- aggregate_training_time -------------------------------------------------


def test_aggregate_training_time_returns_empty_on_empty_manifest() -> None:
    assert aggregate_training_time(pd.DataFrame()) == []


def test_aggregate_training_time_one_group_one_cell() -> None:
    manifest = pd.DataFrame([_row(wall_seconds=2.5)])
    summaries = aggregate_training_time(manifest)
    assert len(summaries) == 1
    s = summaries[0]
    assert isinstance(s, TrainingTimeSummary)
    assert s.dataset_name == "ds_a"
    assert s.model_name == "model_x"
    assert s.hardware_tier == "cpu"
    assert s.n_cells_evaluated == 1
    assert s.n_skipped == 0
    assert s.wall_seconds_mean == pytest.approx(2.5)
    # Single-sample std is defined as 0.0 (population std with one
    # element collapses); pin the convention.
    assert s.wall_seconds_std == pytest.approx(0.0)
    # peak_cuda_bytes is None for cpu runs.
    assert s.peak_cuda_bytes_mean is None
    assert s.peak_cuda_bytes_max is None


def test_aggregate_training_time_multi_cell_mean_and_std() -> None:
    manifest = pd.DataFrame(
        [
            _row(fold_index=0, wall_seconds=1.0),
            _row(fold_index=1, wall_seconds=2.0),
            _row(fold_index=2, wall_seconds=3.0),
        ]
    )
    summaries = aggregate_training_time(manifest)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_cells_evaluated == 3
    assert s.wall_seconds_mean == pytest.approx(2.0)
    # ddof=0 population std of [1, 2, 3] about 2 is sqrt(2/3).
    assert s.wall_seconds_std == pytest.approx(math.sqrt(2.0 / 3.0))


def test_aggregate_training_time_splits_by_hardware_tier() -> None:
    manifest = pd.DataFrame(
        [
            _row(hardware_tier="cpu", wall_seconds=10.0, peak_cuda_bytes=None),
            _row(hardware_tier="gpu_single", wall_seconds=2.0, peak_cuda_bytes=5e8),
        ]
    )
    summaries = aggregate_training_time(manifest)
    assert len(summaries) == 2
    tiers = sorted(s.hardware_tier for s in summaries)
    assert tiers == ["cpu", "gpu_single"]
    gpu = next(s for s in summaries if s.hardware_tier == "gpu_single")
    assert gpu.wall_seconds_mean == pytest.approx(2.0)
    assert gpu.peak_cuda_bytes_mean == pytest.approx(5e8)


def test_aggregate_training_time_skipped_cells_excluded_from_mean() -> None:
    manifest = pd.DataFrame(
        [
            _row(fold_index=0, wall_seconds=1.0),
            _row(
                fold_index=1,
                skipped_reason="adapter_error: ...",
                wall_seconds=None,
            ),
        ]
    )
    summaries = aggregate_training_time(manifest)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_cells_evaluated == 1
    assert s.n_skipped == 1
    # Mean is over the OK cell only.
    assert s.wall_seconds_mean == pytest.approx(1.0)


def test_aggregate_training_time_all_skipped_yields_sentinel_row() -> None:
    manifest = pd.DataFrame(
        [
            _row(skipped_reason="task_type_mismatch", wall_seconds=None),
            _row(fold_index=1, skipped_reason="task_type_mismatch", wall_seconds=None),
        ]
    )
    summaries = aggregate_training_time(manifest)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_cells_evaluated == 0
    assert s.n_skipped == 2
    assert s.wall_seconds_mean is None
    assert s.wall_seconds_std is None


def test_aggregate_training_time_sorts_within_dataset_by_wall_seconds() -> None:
    manifest = pd.DataFrame(
        [
            _row(model_name="slow", wall_seconds=10.0),
            _row(model_name="fast", wall_seconds=1.0),
            _row(model_name="mid", wall_seconds=5.0),
        ]
    )
    summaries = aggregate_training_time(manifest)
    names_in_order = [s.model_name for s in summaries]
    assert names_in_order == ["fast", "mid", "slow"]


def test_aggregate_training_time_sentinel_row_sinks_to_bottom() -> None:
    # A fully-skipped group has `wall_seconds_mean=None`; the sort
    # key uses `+inf` so it lands at the end of its dataset block.
    manifest = pd.DataFrame(
        [
            _row(model_name="skipped", skipped_reason="x", wall_seconds=None),
            _row(model_name="fast", wall_seconds=1.0),
        ]
    )
    summaries = aggregate_training_time(manifest)
    names_in_order = [s.model_name for s in summaries]
    assert names_in_order == ["fast", "skipped"]


# --- render_training_time_markdown -------------------------------------------


def test_render_training_time_markdown_empty_manifest_returns_no_results() -> None:
    md = render_training_time_markdown(pd.DataFrame())
    assert md.startswith("# Training-time report")
    assert "_No results" in md


def test_render_training_time_markdown_includes_dataset_and_columns() -> None:
    manifest = pd.DataFrame([_row(wall_seconds=2.5)])
    md = render_training_time_markdown(manifest)
    assert "Training-time report" in md
    assert "ds_a" in md
    assert "model_x" in md
    # The mean column header is present.
    assert "wall_seconds_mean" in md
    # 3-decimal formatting.
    assert "2.500" in md


def test_render_training_time_markdown_emits_footnote_for_all_skipped() -> None:
    manifest = pd.DataFrame(
        [
            _row(model_name="ok_model", wall_seconds=1.0),
            _row(
                model_name="all_skipped_model",
                skipped_reason="adapter_error: ...",
                wall_seconds=None,
            ),
        ]
    )
    md = render_training_time_markdown(manifest)
    # Healthy model appears in the dataset block.
    assert "ok_model" in md
    # All-skipped group lands in the footnote.
    assert "Skipped groups" in md
    assert "all_skipped_model" in md


def test_render_training_time_markdown_omits_footnote_when_nothing_skipped() -> None:
    manifest = pd.DataFrame([_row(wall_seconds=1.0)])
    md = render_training_time_markdown(manifest)
    assert "Skipped groups" not in md


def test_render_training_time_markdown_splits_block_per_task_type_for_same_dataset() -> None:
    """Gemini final-pass IMP-2 regression guard: if the manifest
    accrues two task_types for the same dataset (e.g., config
    drift across runs), the renderer must NOT crash the
    `_render_dataset_block` homogeneity assert. It must split into
    one block per (dataset_name, task_type) combo and render both
    cleanly."""
    manifest = pd.DataFrame(
        [
            _row(
                model_name="m_binary",
                task_type="binary",
                wall_seconds=1.0,
            ),
            _row(
                model_name="m_regression",
                task_type="regression_point",
                wall_seconds=2.0,
            ),
        ]
    )
    md = render_training_time_markdown(manifest)
    # Both blocks present, one for each task_type, with the
    # task_type appearing in the block header.
    assert "ds_a (binary," in md
    assert "ds_a (regression_point," in md
    # Both models still surface.
    assert "m_binary" in md
    assert "m_regression" in md


# --- _format_value / _format_bytes branch coverage ---------------------------


def test_format_value_returns_empty_string_for_none() -> None:
    assert _format_value(None) == ""


def test_format_value_returns_three_decimal_string_for_float() -> None:
    assert _format_value(2.5) == "2.500"


def test_format_value_returns_empty_string_for_float_nan() -> None:
    assert _format_value(float("nan")) == ""


def test_format_value_returns_str_for_non_float() -> None:
    # The fallback branch stringifies anything that's neither None
    # nor a float (e.g., an int n_cells, a model_name string).
    assert _format_value(7) == "7"
    assert _format_value("cpu") == "cpu"


def test_format_bytes_returns_empty_string_for_none() -> None:
    assert _format_bytes(None) == ""


def test_format_bytes_returns_int_string_for_float() -> None:
    # Floats render as their truncated-int form (no humanizing suffix
    # at B7 per the module docstring).
    assert _format_bytes(1_500_000.0) == "1500000"


def test_format_bytes_returns_empty_string_for_float_nan() -> None:
    assert _format_bytes(float("nan")) == ""


def test_format_bytes_returns_str_for_non_float() -> None:
    assert _format_bytes(1_500_000) == "1500000"
    assert _format_bytes("cpu") == "cpu"


# --- _render_dataset_block invariants ----------------------------------------


def _summary(
    *,
    model_name: str = "m",
    task_type: str = "binary",
    dataset_name: str = "ds_a",
    hardware_tier: str = "cpu",
) -> TrainingTimeSummary:
    return TrainingTimeSummary(
        dataset_name=dataset_name,
        model_name=model_name,
        hardware_tier=hardware_tier,
        task_type=task_type,
        n_cells_evaluated=1,
        n_skipped=0,
        wall_seconds_mean=1.0,
        wall_seconds_std=0.0,
        peak_rss_bytes_mean=1.0,
        peak_rss_bytes_max=1.0,
        peak_cuda_bytes_mean=None,
        peak_cuda_bytes_max=None,
    )


def test_render_dataset_block_raises_on_heterogeneous_task_types() -> None:
    """The aggregator's group key includes `task_type`, so every
    block reaching `_render_dataset_block` has one task_type by
    construction. The defensive assert at
    `report/training_time.py` guards against future grouping drift;
    pin that it actually fires when violated."""
    block = [
        _summary(model_name="m_binary", task_type="binary"),
        _summary(model_name="m_regression", task_type="regression_point"),
    ]
    with pytest.raises(AssertionError, match="heterogeneous task_types"):
        _render_dataset_block("ds_a", block)


# --- render_from_dir reads the manifest on disk ------------------------------


def _write_manifest_shard(root: Path, rows: list[dict[str, object]]) -> None:
    """Write a single parquet shard under `root/results/` so
    `load_run(root)` returns the given rows. Keeps the test
    independent of `write_cell`'s sentinel contract.
    """
    target = results_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(target / "test_shard.parquet", index=False)


def test_render_from_dir_reads_manifest_and_returns_markdown(tmp_path: Path) -> None:
    """qa-IMP-1: pin that `render_from_dir` round-trips an on-disk
    manifest through `load_run` and emits the same Markdown the
    aggregate+render pair produces. The CLI calls this helper, so
    breaking it silently skips the report write."""
    root = tmp_path / "out"
    _write_manifest_shard(
        root,
        [
            _row(model_name="m_fast", wall_seconds=1.0),
            _row(model_name="m_slow", wall_seconds=5.0),
        ],
    )
    md = render_from_dir(root)
    assert "# Training-time report" in md
    assert "ds_a" in md
    assert "m_fast" in md
    assert "m_slow" in md


def test_render_from_dir_empty_results_dir_returns_no_results_block(
    tmp_path: Path,
) -> None:
    """A `root` with no shards under `results/` must still render a
    legal Markdown document (the "_No results in manifest_" block)
    instead of raising."""
    root = tmp_path / "out"
    root.mkdir(parents=True)
    md = render_from_dir(root)
    assert md.startswith("# Training-time report")
    assert "_No results" in md


# --- structural ---


def test_training_time_summary_is_frozen_and_extra_forbid() -> None:
    from pydantic import ValidationError

    s = TrainingTimeSummary(
        dataset_name="ds_a",
        model_name="m",
        hardware_tier="cpu",
        task_type="binary",
        n_cells_evaluated=1,
        n_skipped=0,
        wall_seconds_mean=1.0,
        wall_seconds_std=0.0,
        peak_rss_bytes_mean=1.0,
        peak_rss_bytes_max=1.0,
        peak_cuda_bytes_mean=None,
        peak_cuda_bytes_max=None,
    )
    # Frozen: assignment raises.
    with pytest.raises(ValidationError):
        s.wall_seconds_mean = 2.0  # pyright: ignore[reportAttributeAccessIssue]
    # extra="forbid": stray kwarg raises.
    with pytest.raises(ValidationError):
        TrainingTimeSummary(
            dataset_name="ds_a",
            model_name="m",
            hardware_tier="cpu",
            task_type="binary",
            n_cells_evaluated=1,
            n_skipped=0,
            wall_seconds_mean=1.0,
            wall_seconds_std=0.0,
            peak_rss_bytes_mean=1.0,
            peak_rss_bytes_max=1.0,
            peak_cuda_bytes_mean=None,
            peak_cuda_bytes_max=None,
            not_a_real_field=1.0,  # pyright: ignore[reportCallIssue]
        )


# --- Phase B14: CI variant renderer tests ---------------------------------


def _b14_manifest() -> pd.DataFrame:
    rows = []
    for s in (0, 1):
        rows.append(
            {
                "dataset_name": "fake_binary",
                "model_name": "fake_constant_binary",
                "hardware_tier": "cpu",
                "task_type": "binary",
                "seed": s,
                "fold_index": 0,
                "log_loss": 0.22,
                "wall_seconds": 1.0 + 0.1 * s,
                "peak_rss_bytes": 1000.0,
                "peak_cuda_bytes": None,
                "skipped_reason": None,
            }
        )
    return pd.DataFrame(rows)


def _b14_rollup_row(
    *,
    primary_loss_mean: float | None = 1.05,
    primary_loss_ci_lo: float | None = 0.98,
    primary_loss_ci_hi: float | None = 1.12,
    bootstrap_skipped_reason: str | None = None,
    manifest_fingerprint: str = "cafef00d" * 8,
    model_name: str = "fake_constant_binary",
    n_cells_evaluated: int = 2,
    n_seeds: int = 2,
) -> TrainingTimeRollupRowType:
    from benchmarks.bootstrap_manifest import TrainingTimeRollupRow

    return TrainingTimeRollupRow(
        dataset_name="fake_binary",
        model_name=model_name,
        hardware_tier="cpu",
        task_type="binary",
        primary_metric="wall_seconds",
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells_evaluated,
        n_skipped_cells=0,
        primary_loss_mean=primary_loss_mean,
        primary_loss_ci_lo=primary_loss_ci_lo,
        primary_loss_ci_hi=primary_loss_ci_hi,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_confidence=0.95,
        bootstrap_rng_algorithm="PCG64",
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


def test_b14_render_training_time_with_ci_renders_mean_and_interval() -> None:
    """Happy path: rollup + matching fingerprint -> CI variant."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    md = render_training_time_markdown_with_ci(
        _b14_manifest(),
        [_b14_rollup_row()],
        expected_manifest_fingerprint="cafef00d" * 8,
    )
    assert "wall_seconds [95% CI]" in md
    assert "1.0500 [0.9800, 1.1200]" in md


def test_b14_render_training_time_without_ci_falls_back_to_scalar() -> None:
    """Empty rollup -> std variant silently."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    md = render_training_time_markdown_with_ci(_b14_manifest(), [])
    assert "wall_seconds [95% CI]" not in md
    assert "wall_seconds_mean" in md


def test_b14_render_training_time_with_ci_drops_old_mean_and_std_headers_when_ci_present() -> None:
    """Stage-1 qa-N2 closure: BOTH `wall_seconds_mean` AND
    `wall_seconds_std` column-headers are ABSENT from the
    dataset-block table header when CI active. The section
    heading text 'ranked by wall_seconds_mean' is preserved
    (it's the section-title prose, not a table header)."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    md = render_training_time_markdown_with_ci(_b14_manifest(), [_b14_rollup_row()])
    assert "wall_seconds [95% CI]" in md
    # The table-header row uses the CI column; the bare-mean
    # column header must NOT appear in any TABLE header (lines
    # starting with `| model |`).
    lines = md.splitlines()
    table_header_lines = [line for line in lines if line.startswith("| model |")]
    assert table_header_lines
    for line in table_header_lines:
        assert "wall_seconds_mean" not in line
        assert "wall_seconds_std" not in line
        assert "wall_seconds [95% CI]" in line


def test_b14_render_training_time_with_ci_falls_back_on_manifest_fingerprint_mismatch() -> None:
    """Stale rollup -> std + footnote."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    md = render_training_time_markdown_with_ci(
        _b14_manifest(),
        [_b14_rollup_row(manifest_fingerprint="oldfingerprint")],
        expected_manifest_fingerprint="newfingerprint",
    )
    assert "Bootstrap rollup is stale" in md
    assert "wall_seconds [95% CI]" not in md


def test_b14_render_training_time_with_ci_surfaces_aggregator_failed_footnote() -> None:
    """aggregator_error_class -> std + footnote."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    md = render_training_time_markdown_with_ci(
        _b14_manifest(), [], aggregator_error_class="RawRollupError"
    )
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_b14_render_training_time_with_ci_manifest_unreadable_appends_freshness_footnote() -> None:
    """manifest_unreadable=True -> CI body + freshness-skipped footnote."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    md = render_training_time_markdown_with_ci(
        _b14_manifest(), [_b14_rollup_row()], manifest_unreadable=True
    )
    assert "wall_seconds [95% CI]" in md
    assert "Bootstrap freshness check skipped" in md


def test_b14_render_training_time_with_ci_marks_partial_fold_cell_with_asterisk() -> None:
    """Stage-4 qa-C2 mirror for B7: when
    rollup_row.n_cells_evaluated < n_seeds * n_folds, the CI cell
    ends with `*`."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    # Two folds per group -> n_seeds * n_folds = 1 * 2 = 2; rollup
    # has n_cells_evaluated=1 -> partial=True.
    rows = []
    for f in (0, 1):
        rows.append(
            {
                "dataset_name": "fake_binary",
                "model_name": "fake_constant_binary",
                "hardware_tier": "cpu",
                "task_type": "binary",
                "seed": 0,
                "fold_index": f,
                "log_loss": 0.22,
                "wall_seconds": 1.0,
                "peak_rss_bytes": 1000.0,
                "peak_cuda_bytes": None,
                "skipped_reason": None,
            }
        )
    partial_manifest = pd.DataFrame(rows)
    rollup = [_b14_rollup_row(n_cells_evaluated=1, n_seeds=1)]
    md = render_training_time_markdown_with_ci(partial_manifest, rollup)
    assert "1.0500 [0.9800, 1.1200]*" in md


def test_b14_render_training_time_with_ci_renders_no_ci_when_rollup_missing_for_group() -> None:
    """Stage-4 code-I2 mirror for B7: manifest has a (dataset,
    model, hardware_tier) group but the rollup row is missing
    -> `(no CI)` cell sentinel."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    rollup = [_b14_rollup_row(model_name="other_model")]
    md = render_training_time_markdown_with_ci(_b14_manifest(), rollup)
    assert "(no CI)" in md
    assert "wall_seconds [95% CI]" in md


def test_b14_render_training_time_with_ci_rollup_skipped_emits_separate_footnote() -> None:
    """Sentinel rollup row -> 'Bootstrap skipped' footnote."""
    from benchmarks.report.training_time import render_training_time_markdown_with_ci

    rollup = [
        _b14_rollup_row(),
        _b14_rollup_row(
            model_name="bad_model",
            primary_loss_mean=None,
            primary_loss_ci_lo=None,
            primary_loss_ci_hi=None,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
            n_cells_evaluated=0,
        ),
    ]
    md = render_training_time_markdown_with_ci(_b14_manifest(), rollup)
    assert "Bootstrap skipped" in md
    assert "all_cells_skipped_in_manifest" in md
