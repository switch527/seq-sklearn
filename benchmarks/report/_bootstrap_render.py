"""Shared bootstrap-CI render helpers (Phase B14 extraction).

Houses the cell formatter, the rollup-skipped-footnote renderer,
and the partial-fold denominator helper used by all three
bootstrap-CI rollup renderers (B5 raw-loss, B6 pairwise, B7
training-time). Hoisted from `benchmarks/report/raw_loss.py` by
B14 so the three renderers share one source of truth.

Package-internal (`_` prefix): consumed only by the three
`bootstrap_*.py` and `raw_loss.py` modules under
`benchmarks/report/`.
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
    "render_rollup_skipped_footnote",
]
