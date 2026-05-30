"""Per-(dataset, model) training-time report (Phase B7 / B6.3).

The B5 raw-loss driver records `wall_seconds`, `peak_rss_bytes`, and
`peak_cuda_bytes` on every successful `ResultRow` via
`measure_fit`. B7 is a report-only pass over that manifest: it
aggregates the fit-time + memory measurements per (dataset, model,
hardware_tier) and renders a Markdown table.

Design alignment (B6.3):

- "`fit` wall-clock per (dataset, model, profile) at fixed config":
  reuses B5's `wall_seconds`. Cells with `skipped_reason != None`
  are excluded from the mean and counted under `n_skipped`.
- "deep models report single-GPU time; baselines report CPU":
  the report groups by `(dataset_name, model_name, hardware_tier)`
  so a model that produced both CPU and GPU cells (e.g., across
  reruns on different machines) renders as two rows. cuML GPU
  baselines fall into the same `hardware_tier="gpu_single"` bucket
  by design and round-trip transparently.

Scaling-curve deferral: design B6.3 names "reported against dataset
size so the scaling curve is visible". The B5 manifest does not
carry a per-row dataset-size column today; a B7-followup
back-ports the count once the schema is extended. B7 ships the
flat per-(dataset, model) table; the scaling-curve plot follows.
"""

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from benchmarks.bootstrap_manifest import (
    TrainingTimeRollupRow,
    load_training_time_rollup,
    training_time_aggregator_failed_sentinel_path,
    training_time_rollup_path,
)
from benchmarks.manifest import load_run
from benchmarks.report._bootstrap_render import (
    folds_per_group,
    format_ci_cell,
    render_bca_health_footnote,
    render_rollup_skipped_footnote,
)
from benchmarks.run_manifest import load_run_manifest, run_manifest_path

logger = logging.getLogger(__name__)


class TrainingTimeSummary(BaseModel):
    """One row of the per-(dataset, model, hardware_tier) training-
    time aggregation.

    All fit-time stats are computed on the OK cells only; the
    `n_skipped` field reports how many cells in the group were
    excluded (any non-None `skipped_reason`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    hardware_tier: str
    task_type: str
    n_cells_evaluated: int
    n_skipped: int
    wall_seconds_mean: float | None
    wall_seconds_std: float | None
    peak_rss_bytes_mean: float | None
    peak_rss_bytes_max: float | None
    peak_cuda_bytes_mean: float | None
    peak_cuda_bytes_max: float | None


def aggregate_training_time(manifest: pd.DataFrame) -> list[TrainingTimeSummary]:
    """Aggregate B5 manifest rows to per-(dataset, model, hardware_
    tier) training-time summaries.

    Returns rows sorted by `(dataset_name ASC, wall_seconds_mean ASC)`
    so the fastest model per dataset surfaces first.
    """
    if manifest.empty:
        return []

    out: list[TrainingTimeSummary] = []
    grouped = manifest.groupby(
        ["dataset_name", "model_name", "hardware_tier", "task_type"], sort=True
    )
    for group_key, block in grouped:
        dataset_name, model_name, hardware_tier, task_type = cast(
            tuple[str, str, str, str], group_key
        )
        ok = block.loc[block["skipped_reason"].isna()]
        n_skipped = int(block.shape[0] - ok.shape[0])
        if ok.empty:
            # Every cell in this group is skipped; surface a sentinel
            # row so the report's footnote can pick it up.
            out.append(
                TrainingTimeSummary(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    hardware_tier=hardware_tier,
                    task_type=task_type,
                    n_cells_evaluated=0,
                    n_skipped=n_skipped,
                    wall_seconds_mean=None,
                    wall_seconds_std=None,
                    peak_rss_bytes_mean=None,
                    peak_rss_bytes_max=None,
                    peak_cuda_bytes_mean=None,
                    peak_cuda_bytes_max=None,
                )
            )
            continue

        def _mean(col: str, _ok: pd.DataFrame = ok) -> float | None:
            values = _ok[col].dropna()
            if values.empty:
                return None
            return float(values.astype(float).mean())

        def _max(col: str, _ok: pd.DataFrame = ok) -> float | None:
            values = _ok[col].dropna()
            if values.empty:
                return None
            return float(values.astype(float).max())

        wall_seconds_series = ok["wall_seconds"].dropna().astype(float)
        if len(wall_seconds_series) > 1:
            wall_std: float | None = float(wall_seconds_series.std(ddof=0))
        elif len(wall_seconds_series) == 1:
            wall_std = 0.0
        else:
            wall_std = None

        out.append(
            TrainingTimeSummary(
                dataset_name=dataset_name,
                model_name=model_name,
                hardware_tier=hardware_tier,
                task_type=task_type,
                n_cells_evaluated=int(ok.shape[0]),
                n_skipped=n_skipped,
                wall_seconds_mean=_mean("wall_seconds"),
                wall_seconds_std=wall_std,
                peak_rss_bytes_mean=_mean("peak_rss_bytes"),
                peak_rss_bytes_max=_max("peak_rss_bytes"),
                peak_cuda_bytes_mean=_mean("peak_cuda_bytes"),
                peak_cuda_bytes_max=_max("peak_cuda_bytes"),
            )
        )

    out.sort(
        key=lambda s: (
            s.dataset_name,
            float("inf") if s.wall_seconds_mean is None else s.wall_seconds_mean,
        )
    )
    return out


def _format_value(value: Any) -> str:
    """Render one cell of the training-time Markdown table.

    `wall_seconds_*` values render to 3 decimals; byte columns
    render in plain int form (no humanizing K/M/G suffix at B7;
    the B7-followup adds that if the report needs it).
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def _format_bytes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{int(value)}"
    return str(value)


def _render_dataset_block(dataset_name: str, summaries: list[TrainingTimeSummary]) -> str:
    """Markdown for one dataset's training-time block."""
    if not summaries:
        return ""
    task_type = summaries[0].task_type
    # The caller groups by `dataset_name` BEFORE calling this helper,
    # and the aggregator's group key includes `task_type`, so all
    # summaries in a single dataset block share one task_type by
    # construction. Pinning the invariant here surfaces any future
    # regression where the grouping shape drifts.
    assert all(s.task_type == task_type for s in summaries), (
        f"dataset {dataset_name!r} has heterogeneous task_types: "
        f"{sorted({s.task_type for s in summaries})}"
    )
    header_cells = [
        "model",
        "hardware_tier",
        "n_cells",
        "wall_seconds_mean",
        "wall_seconds_std",
        "peak_rss_bytes_mean",
        "peak_rss_bytes_max",
        "peak_cuda_bytes_mean",
        "peak_cuda_bytes_max",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type}, ranked by wall_seconds_mean)",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for summary in summaries:
        row_cells = [
            summary.model_name,
            summary.hardware_tier,
            str(summary.n_cells_evaluated),
            _format_value(summary.wall_seconds_mean),
            _format_value(summary.wall_seconds_std),
            _format_bytes(summary.peak_rss_bytes_mean),
            _format_bytes(summary.peak_rss_bytes_max),
            _format_bytes(summary.peak_cuda_bytes_mean),
            _format_bytes(summary.peak_cuda_bytes_max),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_skipped_footnote(summaries: list[TrainingTimeSummary]) -> str:
    """Footnote listing groups where every cell was skipped."""
    fully_skipped = [s for s in summaries if s.n_cells_evaluated == 0]
    if not fully_skipped:
        return ""
    lines = ["### Skipped groups", ""]
    lines.append("| Dataset | Model | Hardware tier | n_skipped |")
    lines.append("| --- | --- | --- | --- |")
    for s in fully_skipped:
        lines.append(f"| {s.dataset_name} | {s.model_name} | {s.hardware_tier} | {s.n_skipped} |")
    lines.append("")
    return "\n".join(lines)


def render_training_time_markdown(manifest: pd.DataFrame) -> str:
    """Render the full per-dataset training-time table + footnote.

    Empty manifest -> a single-line "no results" Markdown block so
    the caller's downstream pipeline still receives valid Markdown.
    """
    if manifest.empty:
        return (
            "# Training-time report\n\n"
            "_No results in manifest; run "
            "`python -m benchmarks.run --experiment=raw_loss --config "
            "<config.toml>` first._\n"
        )
    summaries = aggregate_training_time(manifest)
    # Key by `(dataset_name, task_type)`, not by `dataset_name`
    # alone: if a manifest accrues two task_types for the same
    # dataset across runs (e.g., the dataset's task_type changed
    # in config.toml between runs), keying by name alone would
    # mix them into one block and trip the homogeneity assert in
    # `_render_dataset_block`. Splitting on task_type also gives
    # the heterogeneous case sensible output: one block per
    # (dataset, task_type) combo, with the task_type already in
    # the block header.
    by_dataset: dict[tuple[str, str], list[TrainingTimeSummary]] = {}
    for s in summaries:
        # Drop fully-skipped groups from the dataset block; they
        # land in the footnote.
        if s.n_cells_evaluated == 0:
            continue
        by_dataset.setdefault((s.dataset_name, s.task_type), []).append(s)

    parts = ["# Training-time report", ""]
    for dataset_name, _task_type in sorted(by_dataset):
        parts.append(_render_dataset_block(dataset_name, by_dataset[(dataset_name, _task_type)]))
    footnote = _render_skipped_footnote(summaries)
    if footnote:
        parts.append(footnote)
    return "\n".join(parts)


def render_training_time_markdown_with_ci(
    manifest: pd.DataFrame,
    rollup: list[TrainingTimeRollupRow],
    *,
    expected_manifest_fingerprint: str | None = None,
    aggregator_error_class: str | None = None,
    manifest_unreadable: bool = False,
) -> str:
    """Render the training-time report with B14 CIs.

    Replaces the existing `wall_seconds_mean` + `wall_seconds_std`
    pair with a single `wall_seconds [95% CI]` column. Footnote
    precedence mirrors the B5/B6 CI variants (B14.4).
    """
    if aggregator_error_class is not None:
        std_body = render_training_time_markdown(manifest)
        footnote = (
            "\n### Bootstrap aggregator failed\n\n"
            f"_The B14 training-time bootstrap-rollup aggregator raised "
            f"`{aggregator_error_class}`; the table above is rendered "
            "in the legacy mean+std shape. See the run logs._\n"
        )
        return std_body + footnote

    if not rollup:
        return render_training_time_markdown(manifest)

    if expected_manifest_fingerprint is not None and any(
        row.manifest_fingerprint != expected_manifest_fingerprint for row in rollup
    ):
        std_body = render_training_time_markdown(manifest)
        footnote = (
            "\n### Bootstrap rollup is stale\n\n"
            "_The B14 training-time bootstrap-rollup file's "
            "manifest_fingerprint does not match the current manifest's; "
            "the CI variant was suppressed for safety._\n"
        )
        return std_body + footnote

    body = _render_with_ci_training_time(manifest, rollup)
    if manifest_unreadable:
        body += (
            "\n### Bootstrap freshness check skipped\n\n"
            "_The B14 training-time freshness check could not run because "
            "`run_manifest.json` is corrupt or schema-mismatched._\n"
        )
    return body


def _render_with_ci_training_time(
    manifest: pd.DataFrame, rollup: list[TrainingTimeRollupRow]
) -> str:
    """Joins manifest + rollup; emits the per-dataset table with
    the CI column replacing wall_seconds_mean + wall_seconds_std."""
    rollup_index: dict[tuple[str, str, str, str], TrainingTimeRollupRow] = {}
    for row in rollup:
        key = (row.dataset_name, row.model_name, row.hardware_tier, row.task_type)
        rollup_index[key] = row

    folds_index = folds_per_group(
        manifest,
        group_columns=("dataset_name", "model_name", "hardware_tier"),
    )

    summaries = aggregate_training_time(manifest)
    by_dataset: dict[tuple[str, str], list[TrainingTimeSummary]] = {}
    for s in summaries:
        if s.n_cells_evaluated == 0:
            continue
        by_dataset.setdefault((s.dataset_name, s.task_type), []).append(s)

    parts = ["# Training-time report", ""]
    for dataset_name, _task_type in sorted(by_dataset):
        parts.append(
            _render_dataset_block_with_ci(
                dataset_name,
                by_dataset[(dataset_name, _task_type)],
                rollup_index,
                folds_index,
            )
        )

    footnote = _render_skipped_footnote(summaries)
    if footnote:
        parts.append(footnote)

    rollup_skipped = [r for r in rollup if r.bootstrap_skipped_reason is not None]
    if rollup_skipped:
        parts.append(
            render_rollup_skipped_footnote(
                rollup_skipped,
                group_columns=("dataset_name", "model_name", "hardware_tier"),
                header_labels=("Dataset", "Model", "Hardware tier"),
            )
        )

    # B24 / D-B21.1: BCa health footnote.
    rollup_with_fallback = [r for r in rollup if r.bootstrap_ci_fallback_reason is not None]
    if rollup_with_fallback:
        parts.append(
            render_bca_health_footnote(
                rollup_with_fallback,
                group_columns=("dataset_name", "model_name", "hardware_tier"),
                header_labels=("Dataset", "Model", "Hardware tier"),
            )
        )
    return "\n".join(parts)


def _render_dataset_block_with_ci(
    dataset_name: str,
    summaries: list[TrainingTimeSummary],
    rollup_index: dict[tuple[str, str, str, str], TrainingTimeRollupRow],
    folds_index: dict[tuple[str, ...], int],
) -> str:
    """CI variant of the per-dataset training-time block. Replaces
    `wall_seconds_mean` + `wall_seconds_std` with a single
    `wall_seconds [95% CI]` column."""
    if not summaries:
        return ""
    task_type = summaries[0].task_type
    assert all(s.task_type == task_type for s in summaries), (
        f"dataset {dataset_name!r} has heterogeneous task_types: "
        f"{sorted({s.task_type for s in summaries})}"
    )
    header_cells = [
        "model",
        "hardware_tier",
        "n_cells",
        "wall_seconds [95% CI]",
        "peak_rss_bytes_mean",
        "peak_rss_bytes_max",
        "peak_cuda_bytes_mean",
        "peak_cuda_bytes_max",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type}, ranked by wall_seconds_mean)",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for summary in summaries:
        key = (summary.dataset_name, summary.model_name, summary.hardware_tier, summary.task_type)
        rollup_row = rollup_index.get(key)
        if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
            ci_cell = "(no CI)"
        else:
            n_folds = folds_index.get(
                (summary.dataset_name, summary.model_name, summary.hardware_tier), 0
            )
            expected = rollup_row.n_seeds * n_folds
            partial = rollup_row.n_cells_evaluated < expected and expected > 0
            ci_cell = format_ci_cell(
                rollup_row.primary_metric_mean,
                rollup_row.primary_metric_ci_lo,
                rollup_row.primary_metric_ci_hi,
                partial=partial,
            )
        row_cells = [
            summary.model_name,
            summary.hardware_tier,
            str(summary.n_cells_evaluated),
            ci_cell,
            _format_bytes(summary.peak_rss_bytes_mean),
            _format_bytes(summary.peak_rss_bytes_max),
            _format_bytes(summary.peak_cuda_bytes_mean),
            _format_bytes(summary.peak_cuda_bytes_max),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_from_dir(output_root: Path) -> str:
    """Convenience: load the B5 manifest under `output_root` and
    render the training-time table.

    The CI variant kicks in when the training-time rollup parquet
    exists AND the rollup's `manifest_fingerprint` matches the
    live run manifest's (B14.4 freshness check). When the CLI
    wrapper caught a `RawRollupError` and wrote the failure
    sentinel, the renderer surfaces the std + footnote.
    """
    manifest = load_run(output_root)

    failure_sentinel = training_time_aggregator_failed_sentinel_path(output_root)
    if failure_sentinel.exists():
        error_class = failure_sentinel.read_text(encoding="utf-8").strip()
        return render_training_time_markdown_with_ci(
            manifest, [], aggregator_error_class=error_class
        )

    if training_time_rollup_path(output_root).exists():
        rollup = load_training_time_rollup(output_root)
        if rollup:
            expected: str | None = None
            manifest_unreadable = False
            if run_manifest_path(output_root).exists():
                try:
                    run_manifest = load_run_manifest(output_root)
                except (FileNotFoundError, ValueError, ValidationError):
                    run_manifest = None
                    manifest_unreadable = True
                if run_manifest is not None:
                    expected = run_manifest.fingerprint()
            return render_training_time_markdown_with_ci(
                manifest,
                rollup,
                expected_manifest_fingerprint=expected,
                manifest_unreadable=manifest_unreadable,
            )
    return render_training_time_markdown(manifest)
