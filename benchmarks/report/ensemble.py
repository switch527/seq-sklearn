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
from pydantic import BaseModel, ConfigDict

from benchmarks.experiments.ensemble import load_pairwise

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


def render_from_dir(output_root: Path) -> str:
    """Convenience: load the pairwise manifest under `output_root`
    and render."""
    manifest = load_pairwise(output_root)
    return render_pairwise_markdown(manifest)
