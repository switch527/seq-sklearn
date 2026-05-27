"""Phase B11 ensemble-lift report (B6.2.5).

Renders the `EnsembleLiftExperimentResult` produced by
`benchmarks.experiments.ensemble_lift.run_ensemble_lift` as the
`ensemble_lift.md` deliverable: a per-dataset Δloss table, the
oracle upper bound, and the Wilcoxon signed-rank + Holm-adjusted
significance call.

Unlike the B5/B6/B7/B8 renderers, this module does NOT expose a
`render_from_dir(output_root)` convenience because building the
ensemble-lift result requires the structured `BenchmarkConfig`
(the driver inspects `config.experiments` for the `ensemble_lift`
spec). The CLI path calls `run_ensemble_lift(config, env=env,
output_root=output_root)` and pipes the result into
`render_ensemble_lift_markdown(result)`.
"""

import logging

from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    PerDatasetLift,
)

logger = logging.getLogger(__name__)

__all__ = [
    "render_ensemble_lift_markdown",
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


def _render_dataset_table(rows: list[PerDatasetLift]) -> str:
    """Per-dataset Δloss block.

    Sorted by (task_type ASC, delta_loss_mean DESC) so the
    largest lifts surface first within each task family. Rows
    with `delta_loss_mean=None` (incomplete pairings) land in the
    footnote.
    """
    complete = [r for r in rows if r.delta_loss_mean is not None]
    incomplete = [r for r in rows if r.delta_loss_mean is None]

    parts = ["## Per-dataset Δloss", ""]
    if not complete:
        parts.append("_No paired (GBM-only, GBM+seq) cells in this manifest._")
        parts.append("")
    else:
        complete.sort(
            key=lambda r: (
                r.task_type,
                # Negate so larger Δ (more lift) sorts first.
                -(r.delta_loss_mean or 0.0),
            )
        )
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
        parts.append("| " + " | ".join(header_cells) + " |")
        parts.append("| " + " | ".join(sep_cells) + " |")
        for row in complete:
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
            parts.append("| " + " | ".join(row_cells) + " |")
        parts.append("")

    if incomplete:
        parts.append("### Incomplete datasets")
        parts.append("")
        parts.append("| Dataset | Task | Reason |")
        parts.append("| --- | --- | --- |")
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
            parts.append(f"| {row.dataset_name} | {row.task_type} | {reason} |")
        parts.append("")

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


def render_ensemble_lift_markdown(
    result: EnsembleLiftExperimentResult,
) -> str:
    """Render the full ensemble-lift Markdown report from the
    driver's structured result.

    The renderer is filesystem-free; the CLI passes the result of
    `run_ensemble_lift` straight in.
    """
    parts: list[str] = [
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
    parts.append(_render_dataset_table(list(result.rows)))
    parts.append(_render_wilcoxon_block(result))
    return "\n".join(parts)
