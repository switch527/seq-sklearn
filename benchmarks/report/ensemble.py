"""Per-dataset pairwise ensemble report (Phase B6 / B6.2).

Reads the pairwise manifest written by
`benchmarks.experiments.run_ensemble`, aggregates across seeds and
folds per (dataset, model_a, model_b), and renders a Markdown
table plus a top-N "most complementary" pairs summary.

Ranking criterion: the design B6.2 names diversity-without-loss
("seq model error is decorrelated from GBM error AND adding it
lowers ensemble loss"). The B6.2.5 complementarity ensemble that
would prove the second half is a B6-followup. For B6 proper, the
top-N ranking uses a single proxy: LOW `pearson_error_corr` (error
decorrelation) combined with HIGH `disagreement_rate` (hard-
prediction independence). A composite "complementarity score" of
`(1 - error_corr) + disagreement` aggregates the two; B6-followup
ledgers the formal stacked-meta-learner ΔLogLoss as the rank key
once the GBM ensemble lands.
"""

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from benchmarks.bootstrap_manifest import (
    PairwiseRollupRow,
    load_pairwise_rollup,
    pairwise_aggregator_failed_sentinel_path,
    pairwise_rollup_path,
)
from benchmarks.experiments.ensemble import load_pairwise
from benchmarks.report._bootstrap_render import (
    folds_per_group,
    format_ci_cell,
    render_bca_health_footnote,
    render_per_fold_cis_footnote,
    render_rollup_skipped_footnote,
)
from benchmarks.run_manifest import load_run_manifest, run_manifest_path

logger = logging.getLogger(__name__)


# Classification cells carry the full diversity-stat set; regression
# cells only carry the cross-task correlations.
_CLASSIFICATION_STATS: tuple[str, ...] = (
    "n_samples",
    "yule_q",
    "phi",
    "disagreement_rate",
    "double_fault_rate",
    "pearson_pred_corr",
    "spearman_pred_corr",
    "pearson_error_corr",
)
_REGRESSION_POINT_STATS: tuple[str, ...] = (
    "n_samples",
    "pearson_pred_corr",
    "spearman_pred_corr",
    "pearson_error_corr",
)


class PairwiseSummary(BaseModel):
    """One row of the per-dataset pairwise aggregation table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_a: str
    model_b: str
    task_type: str
    n_cells_evaluated: int
    n_skipped: int
    # Each stat carries the mean across folds + seeds.
    yule_q_mean: float | None
    phi_mean: float | None
    disagreement_mean: float | None
    double_fault_mean: float | None
    pearson_pred_mean: float | None
    spearman_pred_mean: float | None
    pearson_error_mean: float | None
    complementarity_score: float | None  # `(1 - error_corr) + disagreement`


def aggregate_pairs(manifest: pd.DataFrame) -> list[PairwiseSummary]:
    """Aggregate pairwise rows to per-(dataset, model_a, model_b)
    summaries.

    Skipped rows are excluded from the mean; their count is
    reported in `n_skipped`. A pair with zero OK rows is dropped
    from the table.
    """
    if manifest.empty:
        return []
    out: list[PairwiseSummary] = []
    grouped = manifest.groupby(["dataset_name", "model_a", "model_b", "task_type"], sort=True)
    for group_key, block in grouped:
        dataset_name, model_a, model_b, task_type = cast(tuple[str, str, str, str], group_key)
        ok = block.loc[block["skipped_reason"].isna()]
        n_skipped = int(block.shape[0] - ok.shape[0])
        if ok.empty:
            continue

        def _mean(col: str, _ok: pd.DataFrame = ok) -> float | None:
            values = _ok[col].dropna()
            if values.empty:
                return None
            return float(values.astype(float).mean())

        error_mean = _mean("pearson_error_corr")
        disagreement_mean = _mean("disagreement_rate")
        # Complementarity score (B6 proxy): higher = better
        # ensemble candidate. Defined only when both inputs are
        # present (classification cells); regression rows carry
        # `disagreement_mean=None` and report `None`.
        if error_mean is not None and disagreement_mean is not None:
            complementarity = (1.0 - error_mean) + disagreement_mean
        else:
            complementarity = None
        out.append(
            PairwiseSummary(
                dataset_name=dataset_name,
                model_a=model_a,
                model_b=model_b,
                task_type=task_type,
                n_cells_evaluated=int(ok.shape[0]),
                n_skipped=n_skipped,
                yule_q_mean=_mean("yule_q"),
                phi_mean=_mean("phi"),
                disagreement_mean=disagreement_mean,
                double_fault_mean=_mean("double_fault_rate"),
                pearson_pred_mean=_mean("pearson_pred_corr"),
                spearman_pred_mean=_mean("spearman_pred_corr"),
                pearson_error_mean=error_mean,
                complementarity_score=complementarity,
            )
        )
    return out


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _stat_columns_for(task_type: str) -> tuple[str, ...]:
    if task_type in {"binary", "multiclass"}:
        return _CLASSIFICATION_STATS
    if task_type == "regression_point":
        return _REGRESSION_POINT_STATS
    return ()


def _render_dataset_block(
    dataset_name: str, summaries: list[PairwiseSummary], manifest: pd.DataFrame
) -> str:
    if not summaries:
        return ""
    task_type = summaries[0].task_type
    cols = _stat_columns_for(task_type)
    header_cells = [
        "model_a",
        "model_b",
        "n_cells",
        *cols,
        "complementarity_score",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type})",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    # Sort by complementarity_score descending (NaN sinks to bottom).
    sorted_summaries = sorted(
        summaries,
        key=lambda s: -float("inf") if s.complementarity_score is None else s.complementarity_score,
        reverse=True,
    )
    for summary in sorted_summaries:
        ok_block = manifest.loc[
            (manifest["dataset_name"] == dataset_name)
            & (manifest["model_a"] == summary.model_a)
            & (manifest["model_b"] == summary.model_b)
            & manifest["skipped_reason"].isna()
        ]
        row_cells = [
            summary.model_a,
            summary.model_b,
            str(summary.n_cells_evaluated),
        ]
        for col in cols:
            if col == "n_samples":
                # The post-strip sample count: total across folds.
                if col in ok_block.columns:
                    total = ok_block["n_samples"].dropna()
                    row_cells.append(str(int(total.astype(int).sum())) if not total.empty else "")
                else:
                    row_cells.append("")
                continue
            stat_field = {
                "yule_q": summary.yule_q_mean,
                "phi": summary.phi_mean,
                "disagreement_rate": summary.disagreement_mean,
                "double_fault_rate": summary.double_fault_mean,
                "pearson_pred_corr": summary.pearson_pred_mean,
                "spearman_pred_corr": summary.spearman_pred_mean,
                "pearson_error_corr": summary.pearson_error_mean,
            }.get(col)
            row_cells.append(_format_float(stat_field))
        row_cells.append(_format_float(summary.complementarity_score))
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_top_n_summary(summaries: list[PairwiseSummary], *, top_n: int = 5) -> str:
    # `complementarity_score is None` only on regression-only pairs;
    # the filter drops them so the sort key dereferences a definite
    # `float`. The `assert` narrows the type for pyright; the
    # defensive `None` arm the lambda used to carry is unreachable
    # after the filter.
    ranked: list[PairwiseSummary] = [s for s in summaries if s.complementarity_score is not None]
    if not ranked:
        return ""

    def _score(s: PairwiseSummary) -> float:
        assert s.complementarity_score is not None
        return s.complementarity_score

    ranked.sort(key=_score, reverse=True)
    top = ranked[:top_n]
    lines = [f"### Top-{len(top)} most complementary pairs", ""]
    lines.append("| Rank | Dataset | model_a | model_b | complementarity_score |")
    lines.append("| --- | --- | --- | --- | --- |")
    for rank, summary in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {summary.dataset_name} | {summary.model_a} | "
            f"{summary.model_b} | {_format_float(summary.complementarity_score)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_pairwise_markdown(manifest: pd.DataFrame) -> str:
    """Render the full per-dataset pairwise table + top-N summary."""
    if manifest.empty:
        return (
            "# Pairwise ensemble-complementarity report\n\n"
            "_No pairwise rows in manifest; run "
            "`python -m benchmarks.run --experiment=ensemble --config "
            "<config.toml>` first._\n"
        )
    summaries = aggregate_pairs(manifest)
    by_dataset: dict[str, list[PairwiseSummary]] = {}
    for s in summaries:
        by_dataset.setdefault(s.dataset_name, []).append(s)
    parts = ["# Pairwise ensemble-complementarity report", ""]
    for dataset_name in sorted(by_dataset):
        parts.append(_render_dataset_block(dataset_name, by_dataset[dataset_name], manifest))
    top_n = _render_top_n_summary(summaries)
    if top_n:
        parts.append(top_n)
    return "\n".join(parts)


def render_pairwise_markdown_with_ci(
    manifest: pd.DataFrame,
    rollup: list[PairwiseRollupRow],
    *,
    expected_manifest_fingerprint: str | None = None,
    aggregator_error_class: str | None = None,
    manifest_unreadable: bool = False,
) -> str:
    """Render the pairwise complementarity report with B14 CIs.

    Joins the pairwise manifest with the rollup table on
    `(dataset_name, model_a, model_b, task_type)` and replaces
    the bare `complementarity_score` column with
    `complementarity_score [95% CI]`.

    Footnote-source precedence (B14.4 mirror of B13):

    1. `aggregator_error_class is not None`: std variant +
       "Bootstrap aggregator failed: <class>" footnote.
    2. `not rollup`: std variant silently.
    3. `expected_manifest_fingerprint` set AND any row mismatch:
       std variant + "rollup is stale" footnote.
    4. CI variant body renders; if `manifest_unreadable=True`,
       append "freshness check skipped" footnote.
    """
    if aggregator_error_class is not None:
        std_body = render_pairwise_markdown(manifest)
        footnote = (
            "\n### Bootstrap aggregator failed\n\n"
            f"_The B14 pairwise bootstrap-rollup aggregator raised "
            f"`{aggregator_error_class}`; the pairwise report above is "
            "rendered in the legacy mean-only shape. See the run logs "
            "for the underlying cause._\n"
        )
        return std_body + footnote

    if not rollup:
        return render_pairwise_markdown(manifest)

    if expected_manifest_fingerprint is not None and any(
        row.manifest_fingerprint != expected_manifest_fingerprint for row in rollup
    ):
        std_body = render_pairwise_markdown(manifest)
        footnote = (
            "\n### Bootstrap rollup is stale\n\n"
            "_The B14 pairwise bootstrap-rollup file's manifest_fingerprint "
            "does not match the current manifest's; the CI variant was "
            "suppressed for safety. Re-run with `--experiment=ensemble` "
            "to refresh the rollup._\n"
        )
        return std_body + footnote

    body = _render_with_ci_pairwise(manifest, rollup)
    if manifest_unreadable:
        body += (
            "\n### Bootstrap freshness check skipped\n\n"
            "_The B14 pairwise freshness check could not run because "
            "`run_manifest.json` is corrupt or schema-mismatched; the CI "
            "variant above was NOT verified against the live manifest's "
            "fingerprint. Re-run `--experiment=ensemble` to refresh both._\n"
        )
    return body


def _render_with_ci_pairwise(manifest: pd.DataFrame, rollup: list[PairwiseRollupRow]) -> str:
    """B14.4 pairwise CI body: joins manifest + rollup and emits
    the per-dataset table with the CI column replacing the bare
    complementarity_score."""
    rollup_index: dict[tuple[str, str, str, str], PairwiseRollupRow] = {}
    for row in rollup:
        key = (row.dataset_name, row.model_a, row.model_b, row.task_type)
        rollup_index[key] = row

    folds_index = folds_per_group(
        manifest,
        group_columns=("dataset_name", "model_a", "model_b"),
    )

    summaries = aggregate_pairs(manifest)
    by_dataset: dict[str, list[PairwiseSummary]] = {}
    for s in summaries:
        by_dataset.setdefault(s.dataset_name, []).append(s)

    parts = ["# Pairwise ensemble-complementarity report", ""]
    for dataset_name in sorted(by_dataset):
        parts.append(
            _render_dataset_block_with_ci(
                dataset_name,
                by_dataset[dataset_name],
                manifest,
                rollup_index,
                folds_index,
            )
        )
    top_n = _render_top_n_summary(summaries)
    if top_n:
        parts.append(top_n)

    rollup_skipped = [r for r in rollup if r.bootstrap_skipped_reason is not None]
    if rollup_skipped:
        parts.append(
            render_rollup_skipped_footnote(
                rollup_skipped,
                group_columns=("dataset_name", "model_a", "model_b"),
                header_labels=("Dataset", "Model A", "Model B"),
            )
        )

    # B24 / D-B21.1: BCa health footnote.
    rollup_with_fallback = [r for r in rollup if r.bootstrap_ci_fallback_reason is not None]
    if rollup_with_fallback:
        parts.append(
            render_bca_health_footnote(
                rollup_with_fallback,
                group_columns=("dataset_name", "model_a", "model_b"),
                header_labels=("Dataset", "Model A", "Model B"),
            )
        )

    # B25 / D-B22.1: per-fold CIs footnote.
    rollup_with_per_fold = [
        r for r in rollup if r.per_fold_cis is not None and len(r.per_fold_cis) > 0
    ]
    if rollup_with_per_fold:
        parts.append(
            render_per_fold_cis_footnote(
                rollup_with_per_fold,
                group_columns=("dataset_name", "model_a", "model_b"),
                header_labels=("Dataset", "Model A", "Model B"),
            )
        )
    return "\n".join(parts)


def _render_dataset_block_with_ci(
    dataset_name: str,
    summaries: list[PairwiseSummary],
    manifest: pd.DataFrame,
    rollup_index: dict[tuple[str, str, str, str], PairwiseRollupRow],
    folds_index: dict[tuple[str, ...], int],
) -> str:
    """Mirror of `_render_dataset_block` with the CI column."""
    if not summaries:
        return ""
    task_type = summaries[0].task_type
    cols = _stat_columns_for(task_type)
    header_cells = [
        "model_a",
        "model_b",
        "n_cells",
        *cols,
        "complementarity_score [95% CI]",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type})",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    sorted_summaries = sorted(
        summaries,
        key=lambda s: -float("inf") if s.complementarity_score is None else s.complementarity_score,
        reverse=True,
    )
    for summary in sorted_summaries:
        key = (summary.dataset_name, summary.model_a, summary.model_b, summary.task_type)
        rollup_row = rollup_index.get(key)
        if rollup_row is None or rollup_row.bootstrap_skipped_reason is not None:
            ci_cell = "(no CI)"
        else:
            n_folds = folds_index.get((summary.dataset_name, summary.model_a, summary.model_b), 0)
            expected = rollup_row.n_seeds * n_folds
            partial = rollup_row.n_cells_evaluated < expected and expected > 0
            ci_cell = format_ci_cell(
                rollup_row.primary_metric_mean,
                rollup_row.primary_metric_ci_lo,
                rollup_row.primary_metric_ci_hi,
                partial=partial,
            )
        ok_block = manifest.loc[
            (manifest["dataset_name"] == dataset_name)
            & (manifest["model_a"] == summary.model_a)
            & (manifest["model_b"] == summary.model_b)
            & manifest["skipped_reason"].isna()
        ]
        row_cells = [summary.model_a, summary.model_b, str(summary.n_cells_evaluated)]
        for col in cols:
            if col == "n_samples":
                if col in ok_block.columns:
                    total = ok_block["n_samples"].dropna()
                    row_cells.append(str(int(total.astype(int).sum())) if not total.empty else "")
                else:
                    row_cells.append("")
                continue
            stat_field = {
                "yule_q": summary.yule_q_mean,
                "phi": summary.phi_mean,
                "disagreement_rate": summary.disagreement_mean,
                "double_fault_rate": summary.double_fault_mean,
                "pearson_pred_corr": summary.pearson_pred_mean,
                "spearman_pred_corr": summary.spearman_pred_mean,
                "pearson_error_corr": summary.pearson_error_mean,
            }.get(col)
            row_cells.append(_format_float(stat_field))
        row_cells.append(ci_cell)
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_from_dir(output_root: Path) -> str:
    """Convenience: load the pairwise manifest under `output_root`
    and render. The CI variant kicks in when a rollup file is
    present AND the rollup's `manifest_fingerprint` matches the
    live run manifest's (B14.4 freshness check). When the CLI
    wrapper caught a `RawRollupError` and wrote the failure
    sentinel, the renderer surfaces the std + footnote."""
    manifest = load_pairwise(output_root)

    failure_sentinel = pairwise_aggregator_failed_sentinel_path(output_root)
    if failure_sentinel.exists():
        error_class = failure_sentinel.read_text(encoding="utf-8").strip()
        return render_pairwise_markdown_with_ci(manifest, [], aggregator_error_class=error_class)

    if pairwise_rollup_path(output_root).exists():
        rollup = load_pairwise_rollup(output_root)
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
            return render_pairwise_markdown_with_ci(
                manifest,
                rollup,
                expected_manifest_fingerprint=expected,
                manifest_unreadable=manifest_unreadable,
            )
    return render_pairwise_markdown(manifest)
