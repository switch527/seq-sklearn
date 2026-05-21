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
from pydantic import BaseModel, ConfigDict

from benchmarks.manifest import load_run

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
    by_dataset: dict[str, list[TrainingTimeSummary]] = {}
    for s in summaries:
        # Drop fully-skipped groups from the dataset block; they
        # land in the footnote.
        if s.n_cells_evaluated == 0:
            continue
        by_dataset.setdefault(s.dataset_name, []).append(s)

    parts = ["# Training-time report", ""]
    for dataset_name in sorted(by_dataset):
        parts.append(_render_dataset_block(dataset_name, by_dataset[dataset_name]))
    footnote = _render_skipped_footnote(summaries)
    if footnote:
        parts.append(footnote)
    return "\n".join(parts)


def render_from_dir(output_root: Path) -> str:
    """Convenience: load the B5 manifest under `output_root` and
    render the training-time table.

    The training-time experiment driver calls this after every
    B5-manifest read; no separate `training_time/` shard layout is
    produced because the data is already in `results/`.
    """
    manifest = load_run(output_root)
    return render_training_time_markdown(manifest)
