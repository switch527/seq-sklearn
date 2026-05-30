"""Shared bootstrap-CI render helpers (Phase B14 extraction).

Houses the cell formatter, the rollup-skipped-footnote
renderer, the BCa health footnote renderer (B24 / D-B21.1),
the per-fold CIs footnote renderer (B25 / D-B22.1), and the
partial-fold denominator helper. Hoisted from
`benchmarks/report/raw_loss.py` by B14 so the renderers share
one source of truth.

Package-internal (`_` prefix): consumed by the five
`*_markdown_with_ci` renderers under `benchmarks/report/`
(`raw_loss.py`, `ensemble.py`, `training_time.py`,
`hpo_uplift.py`, `ensemble_lift.py`).
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd


def format_ci_cell(
    mean: float | None,
    ci_lo: float | None,
    ci_hi: float | None,
    *,
    partial: bool,
) -> str:
    """`mean [ci_lo, ci_hi]` with 4 decimal places; appends `*` when
    `partial=True` (rollup ran on fewer cells than `n_seeds *
    n_folds`)."""
    if mean is None or ci_lo is None or ci_hi is None:
        return "(no CI)"
    star = "*" if partial else ""
    return f"{mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]{star}"


def render_rollup_skipped_footnote(
    rollup_skipped: Sequence[Any],
    *,
    group_columns: Sequence[str] = ("dataset_name", "model_name"),
    header_labels: Sequence[str] = ("Dataset", "Model"),
) -> str:
    """Render the 'Bootstrap skipped' footnote table.

    `rollup_skipped` is the list of rollup rows whose
    `bootstrap_skipped_reason` is populated. `group_columns` names
    the row-attribute names to read; `header_labels` names the
    Markdown header text (the two lists must agree in length). B5
    uses columns `("dataset_name", "model_name")` with labels
    `("Dataset", "Model")`; B6 uses
    `("dataset_name", "model_a", "model_b")` with
    `("Dataset", "Model A", "Model B")`; B7 uses
    `("dataset_name", "model_name", "hardware_tier")` with
    `("Dataset", "Model", "Hardware tier")`.
    """
    if len(group_columns) != len(header_labels):
        raise ValueError(
            f"group_columns and header_labels must have equal length; "
            f"got {len(group_columns)} vs {len(header_labels)}"
        )
    lines = ["### Bootstrap skipped", ""]
    header = "| " + " | ".join([*header_labels, "Reason"]) + " |"
    sep = "| " + " | ".join(["---"] * (len(group_columns) + 1)) + " |"
    lines.append(header)
    lines.append(sep)
    for row in rollup_skipped:
        reason = str(getattr(row, "bootstrap_skipped_reason", None) or "")
        if len(reason) > 120:
            reason = reason[:117] + "..."
        cells = [str(getattr(row, col, "")) for col in group_columns]
        cells.append(reason)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_bca_health_footnote(
    rollup_with_fallback: Sequence[Any],
    *,
    group_columns: Sequence[str] = ("dataset_name", "model_name"),
    header_labels: Sequence[str] = ("Dataset", "Model"),
) -> str:
    """B24 / D-B21.1: render the 'Bootstrap CI method' footnote.

    Caller pre-filters: pass only rows whose
    `bootstrap_ci_fallback_reason` is non-None. Empty input
    returns "" (mirrors `_render_oracle_partial_coverage_footnote`).

    Each row must carry `bootstrap_ci_method: str` and
    `bootstrap_ci_fallback_reason: str | None`. All 5 v1
    RollupRow classes carry both fields; a future row type
    lacking them would silently render empty cells via
    `getattr(row, name, "")` rather than raise. The 120-char
    truncation matches `render_rollup_skipped_footnote`. Rows
    are sorted deterministically by `group_columns[0]` for
    reproducible report bytes.
    """
    if len(group_columns) != len(header_labels):
        raise ValueError(
            f"group_columns and header_labels must have equal length; "
            f"got {len(group_columns)} vs {len(header_labels)}"
        )
    if not rollup_with_fallback:
        return ""
    sort_key = group_columns[0]
    sorted_rows = sorted(rollup_with_fallback, key=lambda r: str(getattr(r, sort_key, "")))
    lines = ["### Bootstrap CI method", ""]
    header = "| " + " | ".join([*header_labels, "ci_method", "fallback_reason"]) + " |"
    sep = "| " + " | ".join(["---"] * (len(group_columns) + 2)) + " |"
    lines.append(header)
    lines.append(sep)
    for row in sorted_rows:
        ci_method = str(getattr(row, "bootstrap_ci_method", ""))
        reason = str(getattr(row, "bootstrap_ci_fallback_reason", None) or "")
        if len(reason) > 120:
            reason = reason[:117] + "..."
        cells = [str(getattr(row, col, "")) for col in group_columns]
        cells.extend([ci_method, reason])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_per_fold_cis_footnote(
    rollup_with_per_fold: Sequence[Any],
    *,
    group_columns: Sequence[str] = ("dataset_name", "model_name"),
    header_labels: Sequence[str] = ("Dataset", "Model"),
) -> str:
    """B25 / D-B22.1: render the 'Per-fold CIs' footnote.

    Caller pre-filter contract: pass only rows whose
    `per_fold_cis` is non-None AND non-empty. The helper's
    outer empty-Sequence early return returns "" for an
    empty `rollup_with_per_fold`; the per-row empty check is
    the caller's responsibility.

    The helper reads ONLY 6 FoldCI fields per fold:
    `fold_index, metric_mean, metric_ci_lo, metric_ci_hi,
    ci_method, ci_fallback_reason`. `n_seeds` and
    `n_entities` are NOT rendered (parquet-shard-audit only;
    D-B25.3). Rows missing `per_fold_cis` render with no
    fold rows via `getattr(row, "per_fold_cis", []) or []`.

    Rows are sorted by `group_columns[0]`; folds within each
    row are sorted defensively by `fold_index` ascending.

    Nullable metric cells render as `-` (literal hyphen);
    `ci_fallback_reason=None` renders as `-`. The 120-char
    truncation on `ci_fallback_reason` matches the sibling
    helpers.

    Raises:
        ValueError: when `group_columns` and `header_labels`
            differ in length.
    """
    if len(group_columns) != len(header_labels):
        raise ValueError(
            f"group_columns and header_labels must have equal length; "
            f"got {len(group_columns)} vs {len(header_labels)}"
        )
    if not rollup_with_per_fold:
        return ""
    sort_key = group_columns[0]
    sorted_rows = sorted(rollup_with_per_fold, key=lambda r: str(getattr(r, sort_key, "")))
    lines = ["### Per-fold CIs", ""]
    header = (
        "| "
        + " | ".join(
            [
                *header_labels,
                "fold",
                "metric_mean",
                "metric_ci_lo",
                "metric_ci_hi",
                "ci_method",
                "ci_fallback_reason",
            ]
        )
        + " |"
    )
    sep = "| " + " | ".join(["---"] * (len(group_columns) + 6)) + " |"
    lines.append(header)
    lines.append(sep)
    for row in sorted_rows:
        fold_cis: Sequence[Any] = getattr(row, "per_fold_cis", None) or []
        sorted_folds = sorted(
            fold_cis, key=lambda f: int(getattr(f, "fold_index", 0))
        )
        id_cells = [str(getattr(row, col, "")) for col in group_columns]
        for fci in sorted_folds:
            mean = getattr(fci, "metric_mean", None)
            ci_lo = getattr(fci, "metric_ci_lo", None)
            ci_hi = getattr(fci, "metric_ci_hi", None)
            mean_cell = "-" if mean is None else f"{mean:.4f}"
            lo_cell = "-" if ci_lo is None else f"{ci_lo:.4f}"
            hi_cell = "-" if ci_hi is None else f"{ci_hi:.4f}"
            ci_method = str(getattr(fci, "ci_method", ""))
            reason_raw = getattr(fci, "ci_fallback_reason", None)
            reason_cell = "-" if reason_raw is None else str(reason_raw)
            if len(reason_cell) > 120:
                reason_cell = reason_cell[:117] + "..."
            row_cells = [
                *id_cells,
                str(int(getattr(fci, "fold_index", 0))),
                mean_cell,
                lo_cell,
                hi_cell,
                ci_method,
                reason_cell,
            ]
            lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def folds_per_group(
    manifest: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    fold_column: str = "fold_index",
    skipped_column: str = "skipped_reason",
) -> dict[tuple[str, ...], int]:
    """Count distinct fold indices per group on the OK rows.

    The B5 renderer uses `("dataset_name", "model_name")` as the
    group key; B6 uses `("dataset_name", "model_a", "model_b")`;
    B7 uses `("dataset_name", "model_name", "hardware_tier")`.
    The returned dict's keys are tuples of the group-column values
    (cast to `str`); the values are the unique-fold counts.
    """
    if manifest.empty or fold_column not in manifest.columns:
        return {}
    ok = manifest.loc[manifest[skipped_column].isna()]
    if ok.empty:
        return {}
    out: dict[tuple[str, ...], int] = {}
    for key, block in ok.groupby(list(group_columns)):
        normalized = tuple(str(v) for v in key) if isinstance(key, tuple) else (str(key),)
        out[normalized] = int(block[fold_column].nunique())
    return out


__all__ = [
    "folds_per_group",
    "format_ci_cell",
    "render_bca_health_footnote",
    "render_per_fold_cis_footnote",
    "render_rollup_skipped_footnote",
]
