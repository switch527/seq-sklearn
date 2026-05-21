"""Phase B7 training-time report tests (B6.3).

The B5 manifest captures `wall_seconds`, `peak_rss_bytes`, and
`peak_cuda_bytes` on every successful row. B7's report aggregates
these into a per-(dataset, model, hardware_tier) table. These tests
exercise the aggregator + renderer directly on hand-crafted
manifest DataFrames; the e2e test in
`test_training_time_experiment.py` drives the full B5 -> B7 flow.
"""

import math

import pandas as pd
import pytest
from benchmarks.report.training_time import (
    TrainingTimeSummary,
    aggregate_training_time,
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
