"""Phase B11 ensemble-lift report (B6.2.5) + Phase B16 CI variant (D-B13.4).

Renders the `EnsembleLiftExperimentResult` produced by
`benchmarks.experiments.ensemble_lift.run_ensemble_lift` as the
`ensemble_lift.md` deliverable: a per-dataset Δloss table, the
oracle upper bound, and the Wilcoxon signed-rank + Holm-adjusted
significance call.

The CI variant (B16) replaces the bare `Δloss` + `Δstd` columns
with a single `Δloss [95% CI]` column when the
`bootstrap_ensemble_lift_rollup.parquet` shard is present and
fresh. `render_from_dir(output_root, *, lift_result)` dispatches
between the std and CI variants following B15's five-source
footnote precedence.
"""

import logging
from pathlib import Path

from pydantic import ValidationError

from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    ensemble_lift_aggregator_failed_sentinel_path,
    ensemble_lift_rollup_path,
    load_ensemble_lift_rollup,
)
from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    PerDatasetLift,
)
from benchmarks.report._bootstrap_render import (
    format_ci_cell,
    render_rollup_skipped_footnote,
)
from benchmarks.run_manifest import load_run_manifest, run_manifest_path

logger = logging.getLogger(__name__)

__all__ = [
    "render_ensemble_lift_markdown",
    "render_ensemble_lift_markdown_with_ci",
    "render_from_dir",
]


def _format_value(value: float | None, *, decimals: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def _format_pvalue(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _sort_complete(rows: list[PerDatasetLift]) -> list[PerDatasetLift]:
    """Sort by (task_type ASC, delta_loss_mean DESC); ties on Δ
    are stable. Caller passes pre-filtered complete rows."""
    return sorted(
        rows,
        key=lambda r: (
            r.task_type,
            # Negate so larger Δ (more lift) sorts first.
            -(r.delta_loss_mean or 0.0),
        ),
    )


def _render_complete_table_std(rows: list[PerDatasetLift]) -> str:
    """Markdown for the complete-rows table (std variant).

    Two columns `Δloss` + `Δstd` ride together. Caller passes
    rows with `delta_loss_mean is not None`."""
    if not rows:
        return ""
    header_cells = [
        "dataset",
        "task",
        "primary_loss",
        "n_cells",
        "loss(GBM)",
        "loss(GBM+seq)",
        "Δloss",
        "Δstd",
        "loss(oracle)",
        "Δloss(oracle)",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for row in _sort_complete(rows):
        row_cells = [
            row.dataset_name,
            row.task_type,
            row.primary_loss_column,
            str(row.n_cells_paired),
            _format_value(row.loss_gbm_only_mean),
            _format_value(row.loss_gbm_plus_seq_mean),
            _format_value(row.delta_loss_mean),
            _format_value(row.delta_loss_std),
            _format_value(row.oracle_loss_mean),
            _format_value(row.oracle_delta_loss_mean),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_complete_table_with_ci(
    rows: list[PerDatasetLift],
    rollup_index: dict[str, EnsembleLiftRollupRow],
) -> str:
    """Markdown for the complete-rows table (CI variant).

    Replaces the bare `Δloss` + `Δstd` columns with a single
    `Δloss [95% CI]` column. When the rollup row for a dataset
    is absent, has a sentinel `bootstrap_skipped_reason`, or
    is missing its `primary_loss_mean`, the CI cell renders as
    `(no CI)` and the per-dataset row still surfaces in the
    table with the existing loss / oracle columns intact."""
    if not rows:
        return ""
    header_cells = [
        "dataset",
        "task",
        "primary_loss",
        "n_cells",
        "loss(GBM)",
        "loss(GBM+seq)",
        "Δloss [95% CI]",
        "loss(oracle)",
        "Δloss(oracle)",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for row in _sort_complete(rows):
        rollup_row = rollup_index.get(row.dataset_name)
        if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
            ci_cell = "(no CI)"
        else:
            expected = rollup_row.n_seeds * rollup_row.n_folds
            partial = rollup_row.n_cells_paired < expected and expected > 0
            ci_cell = format_ci_cell(
                rollup_row.primary_loss_mean,
                rollup_row.primary_loss_ci_lo,
                rollup_row.primary_loss_ci_hi,
                partial=partial,
            )
        row_cells = [
            row.dataset_name,
            row.task_type,
            row.primary_loss_column,
            str(row.n_cells_paired),
            _format_value(row.loss_gbm_only_mean),
            _format_value(row.loss_gbm_plus_seq_mean),
            ci_cell,
            _format_value(row.oracle_loss_mean),
            _format_value(row.oracle_delta_loss_mean),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_incomplete_block(rows: list[PerDatasetLift]) -> str:
    """Footnote table listing per-dataset incomplete groups.

    Shared by std and CI variants (the B11 incomplete-block
    surface is unchanged in B16; the new
    `all_cells_skipped_in_manifest` rollup sentinel is
    surfaced separately by the CI variant)."""
    incomplete = [r for r in rows if r.delta_loss_mean is None]
    if not incomplete:
        return ""
    lines = ["### Incomplete datasets", ""]
    lines.append("| Dataset | Task | Reason |")
    lines.append("| --- | --- | --- |")
    for row in incomplete:
        if row.no_gbm_predictions and row.no_seq_predictions:
            reason = "neither GBM nor seq cells in manifest"
        elif row.no_gbm_predictions:
            reason = "no GBM cells in manifest (only seq)"
        elif row.no_seq_predictions:
            reason = "no seq cells in manifest (only GBM)"
        else:
            reason = (
                "GBM + seq cells present but no (seed, fold) pairs "
                "yielded both ensembles (predictions shards missing "
                "or proba-column mismatch)"
            )
        lines.append(f"| {row.dataset_name} | {row.task_type} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def _render_dataset_table(rows: list[PerDatasetLift]) -> str:
    """Std-variant per-dataset Δloss block.

    Composes `_render_complete_table_std` + `_render_incomplete_block`.
    Sorted by (task_type ASC, delta_loss_mean DESC) so the
    largest lifts surface first within each task family. Rows
    with `delta_loss_mean=None` (incomplete pairings) land in the
    footnote.
    """
    complete = [r for r in rows if r.delta_loss_mean is not None]
    parts = ["## Per-dataset Δloss", ""]
    if not complete:
        parts.append("_No paired (GBM-only, GBM+seq) cells in this manifest._")
        parts.append("")
    else:
        parts.append(_render_complete_table_std(complete))
    incomplete_md = _render_incomplete_block(rows)
    if incomplete_md:
        parts.append(incomplete_md)
    return "\n".join(parts)


def _render_wilcoxon_block(result: EnsembleLiftExperimentResult) -> str:
    """Significance test block.

    Quotes the Wilcoxon statistic, the raw p-value, and the
    Holm-adjusted p-value (equal to raw when family_size=1).
    """
    wilcoxon = result.wilcoxon
    parts = ["## Significance: Wilcoxon signed-rank", ""]
    parts.append(
        f"Pair: `{result.seq_family}` deep model vs `{result.baseline_family}` "
        "baseline ensemble. Paired over datasets on per-dataset Δloss "
        "(positive Δ means adding the deep model lowered the loss)."
    )
    parts.append("")
    if wilcoxon.statistic is None or wilcoxon.p_value is None:
        parts.append(
            f"_Wilcoxon skipped: n_datasets={wilcoxon.n_datasets} "
            "(paired signed-rank needs >= 2 datasets for inference; "
            "the single-dataset Δloss above is informational)._"
        )
        parts.append("")
        return "\n".join(parts)
    parts.append(
        f"- Statistic: `{wilcoxon.statistic:.4f}`\n"
        f"- Raw p-value: `{_format_pvalue(wilcoxon.p_value)}`\n"
        f"- Holm-adjusted p-value (family_size={wilcoxon.family_size}): "
        f"`{_format_pvalue(wilcoxon.holm_adjusted_p_value)}`\n"
        f"- n_datasets paired: {wilcoxon.n_datasets}"
    )
    parts.append("")
    return "\n".join(parts)


def _render_header(result: EnsembleLiftExperimentResult) -> str:
    return "\n".join(
        [
            "# Ensemble-lift report (B6.2.5)",
            "",
            (
                f"_The `{result.baseline_family}` family forms a baseline ensemble "
                f"(equal-weight average of its registered cells per fold); the "
                f"`{result.baseline_family} + {result.seq_family}` ensemble adds the deep "
                "model to the same pool. Δloss is the seed-mean of "
                "`loss(baseline) - loss(baseline+seq)` per (seed, fold) pair; "
                "positive Δ means the deep model lowered the loss. The "
                "per-sample-best oracle row quotes the ceiling._"
            ),
            "",
        ]
    )


def render_ensemble_lift_markdown(
    result: EnsembleLiftExperimentResult,
) -> str:
    """Render the full ensemble-lift Markdown report from the
    driver's structured result.

    The renderer is filesystem-free; the CLI passes the result of
    `run_ensemble_lift` straight in.
    """
    parts: list[str] = [_render_header(result)]
    parts.append(_render_dataset_table(list(result.rows)))
    parts.append(_render_wilcoxon_block(result))
    return "\n".join(parts)


def render_ensemble_lift_markdown_with_ci(
    result: EnsembleLiftExperimentResult,
    rollup: list[EnsembleLiftRollupRow],
    *,
    expected_manifest_fingerprint: str | None = None,
    aggregator_error_class: str | None = None,
    manifest_unreadable: bool = False,
) -> str:
    """Render the ensemble-lift report with B16 paired-Δ CIs.

    Joins the result's per-dataset rows with the rollup table on
    `dataset_name` and replaces the bare `Δloss` + `Δstd`
    columns with `Δloss [95% CI]`.

    Footnote-source precedence (B16.4, mirrors B15.4):

    1. `aggregator_error_class is not None`: std variant +
       "Bootstrap aggregator failed: <class>" footnote.
    2. `not rollup`: std variant silently.
    3. `expected_manifest_fingerprint` set AND any row mismatch:
       std variant + "rollup is stale" footnote.
    4. CI variant body renders; if `manifest_unreadable=True`,
       append "freshness check skipped" footnote.
    5. CI body iteration: if ANY rollup row has
       `n_skipped_cells > 0`, append a "Partial coverage"
       footnote. The new `all_cells_skipped_in_manifest`
       sentinel rows surface via the shared
       `render_rollup_skipped_footnote` helper. The two
       pre-existing B11 sentinels (`no_gbm_predictions`,
       `no_seq_predictions`) flow through the existing
       incomplete-block via the `PerDatasetLift` flags
       (B16.4 arch-I1 closure).
    """
    if aggregator_error_class is not None:
        std_body = render_ensemble_lift_markdown(result)
        footnote = (
            "\n### Bootstrap aggregator failed\n\n"
            f"_The B16 ensemble-lift bootstrap-rollup aggregator raised "
            f"`{aggregator_error_class}`; the report above is rendered "
            "in the legacy mean-only shape. See the run logs._\n"
        )
        return std_body + footnote

    if not rollup:
        return render_ensemble_lift_markdown(result)

    if expected_manifest_fingerprint is not None and any(
        row.manifest_fingerprint != expected_manifest_fingerprint for row in rollup
    ):
        std_body = render_ensemble_lift_markdown(result)
        footnote = (
            "\n### Bootstrap rollup is stale\n\n"
            "_The B16 ensemble-lift bootstrap-rollup file's "
            "manifest_fingerprint does not match the current manifest's; "
            "the CI variant was suppressed for safety._\n"
        )
        return std_body + footnote

    body = _render_with_ci(result, rollup)
    if manifest_unreadable:
        body += (
            "\n### Bootstrap freshness check skipped\n\n"
            "_The B16 ensemble-lift freshness check could not run because "
            "`run_manifest.json` is corrupt or schema-mismatched._\n"
        )
    return body


def _render_with_ci(
    result: EnsembleLiftExperimentResult,
    rollup: list[EnsembleLiftRollupRow],
) -> str:
    """Compose the CI-variant body."""
    rollup_index: dict[str, EnsembleLiftRollupRow] = {row.dataset_name: row for row in rollup}
    parts: list[str] = [_render_header(result)]

    complete = [r for r in result.rows if r.delta_loss_mean is not None]
    table_parts: list[str] = ["## Per-dataset Δloss", ""]
    if not complete:
        table_parts.append("_No paired (GBM-only, GBM+seq) cells in this manifest._")
        table_parts.append("")
    else:
        table_parts.append(_render_complete_table_with_ci(complete, rollup_index))
    incomplete_md = _render_incomplete_block(list(result.rows))
    if incomplete_md:
        table_parts.append(incomplete_md)
    parts.append("\n".join(table_parts))
    parts.append(_render_wilcoxon_block(result))

    # Source 5a: partial-coverage footnote on any rollup row with
    # n_skipped_cells > 0.
    partial = [r for r in rollup if r.bootstrap_skipped_reason is None and r.n_skipped_cells > 0]
    if partial:
        parts.append(_render_partial_coverage_footnote(partial))

    # Source 5b: NEW all_cells_skipped_in_manifest sentinel via the
    # shared helper. The two pre-existing B11 incomplete-block
    # sentinels (no_gbm_predictions, no_seq_predictions) are
    # already surfaced through `_render_incomplete_block` via the
    # PerDatasetLift flags (B16.4 arch-I1 closure: the renderer-side
    # incomplete block keeps the pre-existing sentinel surface; the
    # shared helper takes only the new B16-specific sentinel).
    all_skipped = [
        r for r in rollup if r.bootstrap_skipped_reason == "all_cells_skipped_in_manifest"
    ]
    if all_skipped:
        parts.append(
            render_rollup_skipped_footnote(
                all_skipped,
                group_columns=("dataset_name",),
                header_labels=("Dataset",),
            )
        )

    return "\n".join(parts)


def _render_partial_coverage_footnote(rollup: list[EnsembleLiftRollupRow]) -> str:
    """Footnote listing per-dataset n_skipped_cells / n_cells_paired
    when any non-sentinel rollup row has a positive skipped count."""
    lines = ["### Partial coverage", ""]
    lines.append("| Dataset | n_skipped_cells / n_cells_paired |")
    lines.append("| --- | --- |")
    for r in rollup:
        lines.append(f"| {r.dataset_name} | {r.n_skipped_cells} / {r.n_cells_paired} |")
    lines.append("")
    return "\n".join(lines)


def render_from_dir(
    output_root: Path,
    *,
    lift_result: EnsembleLiftExperimentResult,
) -> str:
    """Dispatch between the std and CI variants from the contents
    of `output_root`.

    Same shape as B15's `render_from_dir`: the caller passes the
    `EnsembleLiftExperimentResult` (since the B11 driver requires
    the structured `BenchmarkConfig` to rebuild it from scratch)
    AND the output root path. The function inspects the failure
    sentinel + rollup file + run manifest to choose the variant.

    Dispatch:
    - Failure-sentinel present → CI variant with
      `aggregator_error_class` set to the sentinel's content.
    - Rollup absent OR empty → std variant silently.
    - Rollup present + freshness OK → CI variant.
    - Run manifest unreadable → CI variant with the freshness
      check skipped footnote.
    """
    failure_sentinel = ensemble_lift_aggregator_failed_sentinel_path(output_root)
    if failure_sentinel.exists():
        error_class = failure_sentinel.read_text(encoding="utf-8").strip()
        return render_ensemble_lift_markdown_with_ci(
            lift_result, [], aggregator_error_class=error_class
        )

    if ensemble_lift_rollup_path(output_root).exists():
        rollup = load_ensemble_lift_rollup(output_root)
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
            return render_ensemble_lift_markdown_with_ci(
                lift_result,
                rollup,
                expected_manifest_fingerprint=expected,
                manifest_unreadable=manifest_unreadable,
            )
    return render_ensemble_lift_markdown(lift_result)
