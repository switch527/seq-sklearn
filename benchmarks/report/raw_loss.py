"""Per-dataset raw-loss leaderboard renderer (Phase B5 / B6.1).

Reads the manifest written by `benchmarks.experiments.run_raw_loss`,
aggregates seeds/folds per (dataset, model), and produces a
Markdown table ranked by the B5 primary loss with the secondary
metric table beside it.

The primary loss is task-type-dependent:

- `binary` / `multiclass`: `log_loss` (lower is better)
- `regression_point`: `rmse` (lower is better)
- `regression_quantile`: deferred to B5-followup (the pinball
  columns are not yet captured); the leaderboard reports the row as
  "skipped" with a `regression_quantile_b5_followup` footnote.

Aggregation policy:
- Per (dataset, model), report the mean+std across seeds + folds
  for the primary loss; report the mean only for the visible
  secondary metrics (per-metric std deferred to B6 when a CD
  diagram lands alongside).
- Skipped cells (any non-None `skipped_reason`) are excluded from
  the rank but listed in a "Skipped cells" footnote so a reader
  knows whether a model "lost" or was "skipped".

The renderer is intentionally text-only (Markdown). Plotting
(critical-difference diagrams, etc.) lands in a later phase tied
to the design B7.5 statistics block.
"""

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from benchmarks.bootstrap_manifest import (
    aggregator_failed_sentinel_path,
    load_rollup,
    rollup_path,
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


# Primary-loss column per task type (B5 / loss-first). Lower is
# better for every entry here.
_PRIMARY_LOSS: dict[str, str] = {
    "binary": "log_loss",
    "multiclass": "log_loss",
    "regression_point": "rmse",
}


# Secondary metric columns reported alongside the primary. Per the
# B5 delta R4 ordering: identity columns first, primary loss, then
# the rest of the applicable metric set.
_CLASSIFICATION_SECONDARY: tuple[str, ...] = (
    "accuracy",
    "precision_zd0",
    "recall_zd0",
    "f1_zd0",
    "precision_macro_zd0",
    "precision_weighted_zd0",
    "recall_macro_zd0",
    "recall_weighted_zd0",
    "f1_macro_zd0",
    "f1_weighted_zd0",
    "roc_auc",
    "pr_auc",
    "roc_auc_macro_ovr",
    "pr_auc_macro_ovr",
    "balanced_accuracy",
    "mcc",
    "brier",
    "brier_multiclass_mean",
    "ece_q15",
)


_REGRESSION_POINT_SECONDARY: tuple[str, ...] = (
    "mae",
    "r2",
    "mape",
    "mape_skip_reason",
)


_RESOURCE_COLUMNS: tuple[str, ...] = (
    "wall_seconds",
    "median_predict_ms",
    "p95_predict_ms",
)


class LeaderboardEntry(BaseModel):
    """One row of the per-dataset leaderboard.

    The `primary_loss_mean` / `primary_loss_std` pair is the rank
    key; the `n_seeds` / `n_cells_evaluated` pair reports the
    aggregation denominator so a reader can spot a thin sample.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    task_type: str
    primary_metric: str  # "log_loss" | "rmse"
    primary_loss_mean: float
    primary_loss_std: float
    n_seeds: int
    n_cells_evaluated: int  # excluding skipped cells
    n_skipped: int  # cells with a non-None skipped_reason


def rank_by_primary_loss(manifest: pd.DataFrame) -> list[LeaderboardEntry]:
    """Aggregate manifest rows to per-(dataset, model) leaderboard
    entries ranked by the primary loss within each dataset.

    Skipped cells (rows with a non-None `skipped_reason`) are
    counted into `n_skipped` and excluded from the rank. A model
    with ALL cells skipped on a dataset is dropped from that
    dataset's leaderboard but included in the skipped-cells
    footnote.

    Returns rows in `(dataset_name, primary_loss_mean ASC)` order;
    `primary_loss_mean` is `nan` when every cell was skipped.
    """
    if manifest.empty:
        return []
    entries: list[LeaderboardEntry] = []
    grouped = manifest.groupby(["dataset_name", "model_name", "task_type"], sort=True)
    # The MultiIndex group key types as `Hashable` to pyright; the
    # cast narrows it to the actual 3-tuple shape.
    for group_key, block in grouped:
        dataset_name, model_name, task_type = cast(tuple[str, str, str], group_key)
        if task_type not in _PRIMARY_LOSS:
            # regression_quantile cells are deferred to a B5-followup.
            continue
        primary_col = _PRIMARY_LOSS[task_type]
        ok_mask = block["skipped_reason"].isna()
        n_skipped = int((~ok_mask).sum())
        ok = block.loc[ok_mask]
        if ok.empty:
            # Every cell skipped; report a sentinel row so the
            # footnote can pick it up.
            entries.append(
                LeaderboardEntry(
                    dataset_name=str(dataset_name),
                    model_name=str(model_name),
                    task_type=str(task_type),
                    primary_metric=primary_col,
                    primary_loss_mean=float("nan"),
                    primary_loss_std=float("nan"),
                    n_seeds=int(block["seed"].nunique()),
                    n_cells_evaluated=0,
                    n_skipped=n_skipped,
                )
            )
            continue
        primary_series = ok[primary_col].astype(float)
        entries.append(
            LeaderboardEntry(
                dataset_name=str(dataset_name),
                model_name=str(model_name),
                task_type=str(task_type),
                primary_metric=primary_col,
                primary_loss_mean=float(primary_series.mean()),
                primary_loss_std=float(primary_series.std(ddof=0))
                if len(primary_series) > 1
                else 0.0,
                n_seeds=int(ok["seed"].nunique()),
                n_cells_evaluated=len(ok),
                n_skipped=n_skipped,
            )
        )
    # Sort by (dataset_name, primary_loss_mean); nan sinks to the bottom.
    entries.sort(
        key=lambda e: (
            e.dataset_name,
            float("inf") if pd.isna(e.primary_loss_mean) else e.primary_loss_mean,
        )
    )
    return entries


def _format_metric(value: Any) -> str:
    """Render one metric cell for the Markdown table."""
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        # 4-decimal display is enough to resolve the typical
        # multi-seed ordering without overflowing the column.
        return f"{value:.4f}"
    return str(value)


def _secondary_columns_for(task_type: str) -> tuple[str, ...]:
    if task_type in {"binary", "multiclass"}:
        return _CLASSIFICATION_SECONDARY
    if task_type == "regression_point":
        return _REGRESSION_POINT_SECONDARY
    return ()


def _render_dataset_block(
    dataset_name: str,
    entries: list[LeaderboardEntry],
    manifest: pd.DataFrame,
) -> str:
    """Markdown for one dataset's leaderboard block."""
    if not entries:
        return ""
    # All entries here share the same task_type per the group key.
    task_type = entries[0].task_type
    primary_metric = entries[0].primary_metric
    secondary_cols = _secondary_columns_for(task_type)
    header_cells = [
        "Rank",
        "Model",
        f"{primary_metric} (mean ± std)",
        "n_cells",
        *secondary_cols,
        *_RESOURCE_COLUMNS,
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type}, ranked by {primary_metric})",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for rank, entry in enumerate(entries, start=1):
        # Pull the mean per secondary column across the OK cells.
        ok_block = manifest.loc[
            (manifest["dataset_name"] == dataset_name)
            & (manifest["model_name"] == entry.model_name)
            & manifest["skipped_reason"].isna()
        ]
        row_cells = [
            str(rank),
            entry.model_name,
            (
                f"{entry.primary_loss_mean:.4f} ± {entry.primary_loss_std:.4f}"
                if not pd.isna(entry.primary_loss_mean)
                else "(all skipped)"
            ),
            str(entry.n_cells_evaluated),
        ]
        for col in secondary_cols:
            if col not in ok_block.columns:
                row_cells.append("")
                continue
            if col == "mape_skip_reason":
                # Categorical column; take the most-common non-null value.
                vals = ok_block[col].dropna()
                row_cells.append(str(vals.iloc[0]) if not vals.empty else "")
                continue
            row_cells.append(_format_metric(ok_block[col].astype(float).mean()))
        for col in _RESOURCE_COLUMNS:
            if col not in ok_block.columns:
                row_cells.append("")
                continue
            row_cells.append(_format_metric(ok_block[col].astype(float).mean()))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_skipped_footnote(manifest: pd.DataFrame) -> str:
    """List skipped cells grouped by reason category."""
    skipped = manifest.loc[manifest["skipped_reason"].notna()]
    if skipped.empty:
        return ""
    lines = ["### Skipped cells", ""]
    grouped = skipped.groupby(["dataset_name", "model_name", "skipped_reason"]).size()
    lines.append("| Dataset | Model | Reason | Count |")
    lines.append("| --- | --- | --- | --- |")
    for group_key, count in grouped.items():
        dataset_name, model_name, reason = cast(tuple[str, str, str], group_key)
        # Truncate long reason strings (adapter tracebacks) so the
        # table stays readable.
        reason_short = str(reason)
        if len(reason_short) > 120:
            reason_short = reason_short[:117] + "..."
        lines.append(f"| {dataset_name} | {model_name} | {reason_short} | {int(count)} |")
    lines.append("")
    return "\n".join(lines)


def render_leaderboard_markdown(manifest: pd.DataFrame) -> str:
    """Render the full per-dataset leaderboard + skipped footnote
    as Markdown.

    Empty manifest -> a single-line "no results" Markdown block so
    the caller's downstream pipeline still receives valid Markdown.
    """
    if manifest.empty:
        return (
            "# Raw-loss leaderboard\n\n"
            "_No results in manifest; run "
            "`python -m benchmarks.run --experiment=raw_loss --config "
            "<config.toml>` first._\n"
        )
    entries = rank_by_primary_loss(manifest)
    by_dataset: dict[str, list[LeaderboardEntry]] = {}
    for entry in entries:
        by_dataset.setdefault(entry.dataset_name, []).append(entry)

    parts = ["# Raw-loss leaderboard", ""]
    for dataset_name in sorted(by_dataset):
        parts.append(_render_dataset_block(dataset_name, by_dataset[dataset_name], manifest))
    footnote = _render_skipped_footnote(manifest)
    if footnote:
        parts.append(footnote)
    return "\n".join(parts)


def render_leaderboard_markdown_with_ci(
    manifest: pd.DataFrame,
    rollup: list[Any],
    *,
    expected_manifest_fingerprint: str | None = None,
    aggregator_error_class: str | None = None,
    manifest_unreadable: bool = False,
) -> str:
    """Render the leaderboard with B13 bootstrap CIs.

    The CI variant joins the manifest with the rollup table on
    `(dataset_name, model_name, task_type)` and replaces the
    primary-loss `mean ± std` column with `mean [ci_lo, ci_hi]`.

    Footnote-source precedence (evaluation order; each is
    short-circuiting unless noted):

    1. `aggregator_error_class is not None` (Gemini-C1/C2 wrapper
       caught a `RawRollupError`): std variant + "Bootstrap
       aggregator failed: <class>" footnote. Nothing else
       evaluated.
    2. `not rollup` (no rollup ran; opt-out or absent file): std
       variant; no footnote.
    3. `expected_manifest_fingerprint` given AND any rollup row's
       `manifest_fingerprint` mismatches (Gemini-C3 freshness):
       std variant + "rollup is stale" footnote. Nothing else
       evaluated.
    4. CI variant body renders; if `manifest_unreadable=True`
       (R2 arch-I2), append the "freshness check skipped"
       footnote to the CI body (NOT short-circuiting, the CI
       variant is what the reader sees).

    Args:
        manifest: the B5 manifest DataFrame (from `load_run`).
        rollup: the B13 rollup rows (from `load_rollup`).
        expected_manifest_fingerprint: the live run manifest's
            fingerprint; the renderer asserts every rollup row
            matches before joining.
        aggregator_error_class: when set, the renderer falls back
            to the std variant + emits the failure footnote.
        manifest_unreadable: when True, the renderer appends a
            "freshness check skipped: manifest unreadable" footnote.
    """
    if aggregator_error_class is not None:
        std_body = render_leaderboard_markdown(manifest)
        footnote = (
            "\n### Bootstrap aggregator failed\n\n"
            f"_The B13 bootstrap-rollup aggregator raised "
            f"`{aggregator_error_class}`; the leaderboard above is "
            "rendered in the legacy `mean ± std` shape. See the "
            "run logs for the underlying cause._\n"
        )
        return std_body + footnote

    if not rollup:
        # No rollup ran (opt-out via ExperimentSpec, or smoke-tier
        # config that disabled it). Fall back silently.
        return render_leaderboard_markdown(manifest)

    if expected_manifest_fingerprint is not None and any(
        row.manifest_fingerprint != expected_manifest_fingerprint for row in rollup
    ):
        std_body = render_leaderboard_markdown(manifest)
        footnote = (
            "\n### Bootstrap rollup is stale\n\n"
            "_The B13 bootstrap-rollup file's manifest_fingerprint "
            "does not match the current manifest's; the CI variant "
            "was suppressed for safety. Re-run with "
            "`--experiment=raw_loss` to refresh the rollup._\n"
        )
        return std_body + footnote

    body = _render_with_ci(manifest, rollup)
    if manifest_unreadable:
        body += (
            "\n### Bootstrap freshness check skipped\n\n"
            "_The B13 freshness check could not run because "
            "`run_manifest.json` is corrupt or schema-mismatched; the "
            "CI variant above was NOT verified against the live "
            "manifest's fingerprint. Re-run `--experiment=raw_loss` "
            "to refresh both the manifest and the rollup._\n"
        )
    return body


def _render_with_ci(manifest: pd.DataFrame, rollup: list[Any]) -> str:
    # Index the rollup by (dataset, model, task_type) for the join.
    rollup_index: dict[tuple[str, str, str], Any] = {}
    for row in rollup:
        key = (row.dataset_name, row.model_name, row.task_type)
        rollup_index[key] = row

    # n_folds per (dataset, model) = the number of distinct
    # fold_index values in the manifest's OK rows for that group.
    # Used for the partial-fold `*` flag (Gemini-I3).
    folds_index = folds_per_group(
        manifest,
        group_columns=("dataset_name", "model_name"),
    )

    entries = rank_by_primary_loss(manifest)
    by_dataset: dict[str, list[LeaderboardEntry]] = {}
    for entry in entries:
        by_dataset.setdefault(entry.dataset_name, []).append(entry)

    parts = ["# Raw-loss leaderboard", ""]
    rollup_skipped: list[Any] = []
    for dataset_name in sorted(by_dataset):
        block = _render_dataset_block_with_ci(
            dataset_name,
            by_dataset[dataset_name],
            manifest,
            rollup_index,
            folds_index,
        )
        parts.append(block)

    rollup_skipped = [row for row in rollup if row.bootstrap_skipped_reason is not None]
    cell_footnote = _render_skipped_footnote(manifest)
    if cell_footnote:
        parts.append(cell_footnote)
    if rollup_skipped:
        parts.append(
            render_rollup_skipped_footnote(
                rollup_skipped,
                group_columns=("dataset_name", "model_name"),
                header_labels=("Dataset", "Model"),
            )
        )

    # B24 / D-B21.1: BCa health footnote.
    rollup_with_fallback = [r for r in rollup if r.bootstrap_ci_fallback_reason is not None]
    if rollup_with_fallback:
        parts.append(
            render_bca_health_footnote(
                rollup_with_fallback,
                group_columns=("dataset_name", "model_name"),
                header_labels=("Dataset", "Model"),
            )
        )
    return "\n".join(parts)


def _render_dataset_block_with_ci(
    dataset_name: str,
    entries: list[LeaderboardEntry],
    manifest: pd.DataFrame,
    rollup_index: dict[tuple[str, str, str], Any],
    folds_index: dict[tuple[str, ...], int],
) -> str:
    if not entries:
        return ""
    task_type = entries[0].task_type
    primary_metric = entries[0].primary_metric
    secondary_cols = _secondary_columns_for(task_type)
    header_cells = [
        "Rank",
        "Model",
        f"{primary_metric} [95% CI]",
        "n_cells",
        *secondary_cols,
        *_RESOURCE_COLUMNS,
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type}, ranked by {primary_metric})",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for rank, entry in enumerate(entries, start=1):
        key = (entry.dataset_name, entry.model_name, entry.task_type)
        rollup_row = rollup_index.get(key)
        if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
            ci_cell = "(no CI)"
        else:
            n_folds = folds_index.get((entry.dataset_name, entry.model_name), 0)
            expected = entry.n_seeds * n_folds
            partial = rollup_row.n_cells_evaluated < expected and expected > 0
            ci_cell = format_ci_cell(
                rollup_row.primary_metric_mean,
                rollup_row.primary_metric_ci_lo,
                rollup_row.primary_metric_ci_hi,
                partial=partial,
            )
        ok_block = manifest.loc[
            (manifest["dataset_name"] == dataset_name)
            & (manifest["model_name"] == entry.model_name)
            & manifest["skipped_reason"].isna()
        ]
        row_cells = [str(rank), entry.model_name, ci_cell, str(entry.n_cells_evaluated)]
        for col in secondary_cols:
            if col not in ok_block.columns:
                row_cells.append("")
                continue
            if col == "mape_skip_reason":
                vals = ok_block[col].dropna()
                row_cells.append(str(vals.iloc[0]) if not vals.empty else "")
                continue
            row_cells.append(_format_metric(ok_block[col].astype(float).mean()))
        for col in _RESOURCE_COLUMNS:
            if col not in ok_block.columns:
                row_cells.append("")
                continue
            row_cells.append(_format_metric(ok_block[col].astype(float).mean()))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_from_dir(output_root: Path) -> str:
    """Convenience: load the manifest under `output_root` and render.

    The CLI calls this after `run_raw_loss` finishes so the
    leaderboard markdown is written next to the manifest shards.
    The CI variant kicks in when a rollup file is present AND the
    rollup's `manifest_fingerprint` matches the live run manifest's
    (Gemini-C3 freshness check). When the CLI wrapper caught a
    `RawRollupError` and wrote `bootstrap_aggregator_failed.txt`,
    the renderer surfaces the "Bootstrap aggregator failed"
    footnote in the std fallback (B13.0 + Gemini-C2 wrapper case).
    """
    manifest = load_run(output_root)

    # Gemini-C2 wrapper case: a failed aggregator dropped a sentinel
    # file naming the exception class. Render std + the failure
    # footnote.
    failure_sentinel = aggregator_failed_sentinel_path(output_root)
    if failure_sentinel.exists():
        error_class = failure_sentinel.read_text(encoding="utf-8").strip()
        return render_leaderboard_markdown_with_ci(manifest, [], aggregator_error_class=error_class)

    if rollup_path(output_root).exists():
        rollup = load_rollup(output_root)
        if rollup:
            # Gemini-C3 freshness check: pass the live manifest's
            # fingerprint so a stale rollup falls back to the std
            # variant with a "rollup is stale" footnote.
            expected: str | None = None
            manifest_unreadable = False
            if run_manifest_path(output_root).exists():
                try:
                    run_manifest = load_run_manifest(output_root)
                except (
                    FileNotFoundError,
                    ValueError,
                    ValidationError,
                ):
                    # R2 arch-I2: the manifest file exists but is
                    # corrupt or schema-mismatched. Surface a
                    # "freshness check skipped" footnote rather
                    # than silently rendering an unverified CI.
                    run_manifest = None
                    manifest_unreadable = True
                if run_manifest is not None:
                    expected = run_manifest.fingerprint()
            return render_leaderboard_markdown_with_ci(
                manifest,
                rollup,
                expected_manifest_fingerprint=expected,
                manifest_unreadable=manifest_unreadable,
            )
    return render_leaderboard_markdown(manifest)
