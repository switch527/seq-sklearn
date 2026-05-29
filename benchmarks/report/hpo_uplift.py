"""HPO-uplift report (Phase B8 / B6.4 deliverable).

Reads the manifest under `output_root`, joins `variant="default"`
rows (the B5 raw-loss arm) against `variant="tuned"` rows (this
phase's arm) per (dataset, model, seed, fold), computes the Δ on
the primary loss, and renders:

- per-(dataset, model) Δ table with the search-space parity
  disclosure (B6.4.0) and the (n_trials, timeout, time-to-best)
  budget columns (B7.4)
- matrix-level Friedman omnibus + Holm-adjusted pairwise
  comparisons against a designated reference model (B7.5)
- skipped-groups footnote for cells whose tuned arm failed

Design alignment:

- "two variants per (model, dataset): `default` config and `tuned`
  under a fixed budget tier" (B6.4.1): the join requires both
  variants to be present for a Δ to render. Cells with only one
  arm fall into the footnote.
- "report Δ(primary loss) default→tuned and time-to-best-trial"
  (B6.4.1): Δ is computed on the primary loss (B6.1 contract:
  log_loss for classification, rmse for regression_point, mean
  pinball for regression_quantile).
- "search-space size next to it" (B6.4.0): the renderer pulls
  `hpo_search_space_size` from the tuned row and displays it.
- "Friedman + Holm correction across the matrix" (B7.5): the
  matrix is one cell per (model, dataset) equal to the seed-mean
  primary loss on the tuned arm. The omnibus uses
  `scipy.stats.friedmanchisquare`; Holm-adjusted pairwise
  Wilcoxon signed-rank comparisons are emitted against the
  configured reference model.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchmarks.bootstrap_manifest import (
    HPOUpliftRollupRow,
    hpo_uplift_aggregator_failed_sentinel_path,
    hpo_uplift_rollup_path,
    load_hpo_uplift_rollup,
)
from benchmarks.manifest import load_run
from benchmarks.report._bootstrap_render import (
    format_ci_cell,
    render_rollup_skipped_footnote,
)
from benchmarks.run_manifest import load_run_manifest, run_manifest_path
from benchmarks.stats.friedman import FriedmanHolmResult, friedman_holm

logger = logging.getLogger(__name__)

__all__ = [
    "HPOUpliftReport",
    "PrimaryLossSelector",
    "UpliftRow",
    "aggregate_hpo_uplift",
    "build_hpo_uplift_report",
    "render_from_dir",
    "render_hpo_uplift_markdown",
]


_DEFAULT_REFERENCE_MODEL = "tft_classifier"
_DEFAULT_VARIANT = "default"
_HPO_VARIANT = "tuned"
_PRIMARY_LOSS_BY_TASK: dict[str, str] = {
    "binary": "log_loss",
    "multiclass": "log_loss",
    "regression_point": "rmse",
    # `regression_quantile` resolves to the `pinball_json` column;
    # the selector unpacks JSON and means across keys.
    "regression_quantile": "pinball_json",
}


class PrimaryLossSelector(BaseModel):
    """Resolved primary-loss column + task-type for one (dataset,
    model) group.

    `column` is the manifest column to read for binary / multiclass
    / regression_point cells. `task_type` is reported so the
    pinball-mean branch is handled uniformly with the simple-column
    branches.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: str
    column: str


class UpliftRow(BaseModel):
    """One row of the per-(dataset, model) Δ aggregation.

    Δ = mean_default - mean_tuned (positive Δ means the tuned arm
    improved). Means are taken seed-and-fold first, then averaged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    task_type: str
    primary_loss_column: str
    n_cells_paired: int = Field(ge=0)
    default_mean: float | None = None
    tuned_mean: float | None = None
    delta: float | None = None
    delta_std: float | None = None
    hpo_search_space_size: int | None = None
    hpo_tier: str | None = None
    hpo_n_trials_completed_mean: float | None = None
    hpo_time_to_best_seconds_mean: float | None = None
    # Footnote dispatch flags. `default_only` and `tuned_only` mean
    # only one arm produced cells in the manifest;
    # `paired_but_no_valid_loss` means cells from both arms exist
    # but every paired (seed, fold) row had a missing primary loss
    # (e.g., adapter crashed during refit). All three carry
    # `delta is None`.
    default_only: bool = False
    tuned_only: bool = False
    paired_but_no_valid_loss: bool = False


class HPOUpliftReport(BaseModel):
    """Bundled Δ rows + Friedman/Holm result.

    `rows` is sorted by (dataset_name ASC, delta DESC) so the best
    uplift per dataset surfaces first. `friedman` is None when the
    matrix has fewer than 2 models and 2 datasets after dropping
    NaN columns.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[UpliftRow, ...]
    friedman: FriedmanHolmResult | None
    reference_model: str


def _primary_loss_value(row: pd.Series, selector: PrimaryLossSelector) -> float | None:
    """Read the primary loss from one manifest row.

    For `regression_quantile`, the manifest's `pinball_json` column
    is a JSON dict `{quantile_label: loss}`; the mean of the values
    is the primary loss per B6.1.
    """
    if selector.task_type == "regression_quantile":
        raw = row.get("pinball_json")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        from benchmarks.manifest import pinball_from_json

        pinball = pinball_from_json(str(raw))
        if not pinball:
            return None
        return float(np.mean(list(pinball.values())))
    value = row.get(selector.column)
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return float(value)


def _resolve_selector(task_type: Any) -> PrimaryLossSelector:
    """Map a `task_type` string to its primary-loss column."""
    task_str = str(task_type)
    column = _PRIMARY_LOSS_BY_TASK.get(task_str)
    if column is None:
        raise ValueError(
            f"_resolve_selector: unsupported task_type={task_str!r}; "
            f"known: {sorted(_PRIMARY_LOSS_BY_TASK)}"
        )
    return PrimaryLossSelector(task_type=task_str, column=column)


def aggregate_hpo_uplift(
    manifest: pd.DataFrame,
) -> list[UpliftRow]:
    """Join default + tuned manifest rows per (dataset, model, seed,
    fold), compute the Δ on the primary loss, and aggregate per
    (dataset, model).

    Cells with only the default variant present (e.g., a model
    family with no HPO space registered) yield a `default_only=True`
    sentinel row so the footnote can list them. Cells with only the
    tuned variant present yield a `tuned_only=True` sentinel.

    Returns rows sorted by (dataset_name ASC, delta DESC). Sentinel
    rows (no Δ) sort to the end.
    """
    if manifest.empty:
        return []

    # Filter to OK cells (skipped cells carry no primary loss).
    ok = manifest.loc[manifest["skipped_reason"].isna()].copy()
    out: list[UpliftRow] = []
    grouped = ok.groupby(["dataset_name", "model_name", "task_type"], sort=True)
    for group_key, block in grouped:
        dataset_name, model_name, task_type = group_key
        try:
            selector = _resolve_selector(task_type)
        except ValueError as exc:
            logger.warning(
                "aggregate_hpo_uplift: skipping group %s/%s task=%s: %s",
                dataset_name,
                model_name,
                task_type,
                exc,
            )
            continue
        default_block = block.loc[block["variant"] == _DEFAULT_VARIANT]
        tuned_block = block.loc[block["variant"] == _HPO_VARIANT]
        if default_block.empty and tuned_block.empty:
            continue
        if default_block.empty:
            # tuned_only group: the Friedman matrix still benefits
            # from the tuned-arm seed-mean loss (design B7.5), so
            # compute `tuned_mean` from the tuned_block even though
            # no Δ is reportable. Without this, `_build_loss_matrix`
            # drops the (model, dataset) cell and the Friedman
            # omnibus understates its sample size (arch-I3).
            tuned_losses = [_primary_loss_value(row, selector) for _, row in tuned_block.iterrows()]
            tuned_losses_clean = [v for v in tuned_losses if v is not None]
            tuned_mean = float(np.mean(tuned_losses_clean)) if tuned_losses_clean else None
            out.append(
                UpliftRow(
                    dataset_name=str(dataset_name),
                    model_name=str(model_name),
                    task_type=str(task_type),
                    primary_loss_column=selector.column,
                    n_cells_paired=0,
                    tuned_mean=tuned_mean,
                    tuned_only=True,
                )
            )
            continue
        if tuned_block.empty:
            out.append(
                UpliftRow(
                    dataset_name=str(dataset_name),
                    model_name=str(model_name),
                    task_type=str(task_type),
                    primary_loss_column=selector.column,
                    n_cells_paired=0,
                    default_only=True,
                )
            )
            continue

        # Inner-join on (seed, fold): only paired cells contribute
        # to the Δ. A cell with default but no tuned (or vice
        # versa) is dropped from the Δ but tallied in
        # `n_cells_paired` so the report knows how many cells the
        # mean is over.
        default_keyed = default_block.set_index(["seed", "fold_index"])
        tuned_keyed = tuned_block.set_index(["seed", "fold_index"])
        common_keys = default_keyed.index.intersection(tuned_keyed.index)
        if len(common_keys) == 0:
            out.append(
                UpliftRow(
                    dataset_name=str(dataset_name),
                    model_name=str(model_name),
                    task_type=str(task_type),
                    primary_loss_column=selector.column,
                    n_cells_paired=0,
                )
            )
            continue

        per_cell_default: list[float] = []
        per_cell_tuned: list[float] = []
        per_cell_delta: list[float] = []
        per_cell_n_trials: list[float] = []
        per_cell_time_to_best: list[float] = []
        for key_tuple in common_keys:
            default_loss = _primary_loss_value(default_keyed.loc[key_tuple], selector)
            tuned_row = tuned_keyed.loc[key_tuple]
            tuned_loss = _primary_loss_value(tuned_row, selector)
            if default_loss is None or tuned_loss is None:
                continue
            per_cell_default.append(default_loss)
            per_cell_tuned.append(tuned_loss)
            per_cell_delta.append(default_loss - tuned_loss)
            n_trials = tuned_row.get("hpo_n_trials_completed")
            if n_trials is not None and not (isinstance(n_trials, float) and pd.isna(n_trials)):
                per_cell_n_trials.append(float(n_trials))
            time_to_best = tuned_row.get("hpo_time_to_best_seconds")
            if time_to_best is not None and not (
                isinstance(time_to_best, float) and pd.isna(time_to_best)
            ):
                per_cell_time_to_best.append(float(time_to_best))

        if not per_cell_delta:
            # Cells exist in both arms but every paired row had a
            # missing primary loss. Surface the group via the
            # footnote (`paired_but_no_valid_loss=True`) so a
            # silent omission doesn't hide the data loss (arch-I4).
            out.append(
                UpliftRow(
                    dataset_name=str(dataset_name),
                    model_name=str(model_name),
                    task_type=str(task_type),
                    primary_loss_column=selector.column,
                    n_cells_paired=len(common_keys),
                    paired_but_no_valid_loss=True,
                )
            )
            continue

        delta_arr = np.asarray(per_cell_delta, dtype=float)
        # Sample std (`ddof=1`) is the convention for paired-cell
        # spread in benchmark reports; a single paired cell has no
        # spread so the std is None (mathematically undefined),
        # not 0.0 (which would understate uncertainty). code-I3.
        delta_std: float | None = float(delta_arr.std(ddof=1)) if delta_arr.size > 1 else None

        # Tuned-arm metadata: pull the search-space size + hpo_tier
        # from the FIRST tuned row in the group; both should be
        # constant within a (dataset, model) group by construction
        # (a single run produces one tier per cell).
        first_tuned = tuned_block.iloc[0]
        search_space_size_raw = first_tuned.get("hpo_search_space_size")
        if search_space_size_raw is None or (
            isinstance(search_space_size_raw, float) and pd.isna(search_space_size_raw)
        ):
            search_space_size: int | None = None
        else:
            search_space_size = int(search_space_size_raw)
        hpo_tier_raw = first_tuned.get("hpo_tier")
        hpo_tier: str | None
        if hpo_tier_raw is None or (isinstance(hpo_tier_raw, float) and pd.isna(hpo_tier_raw)):
            hpo_tier = None
        else:
            hpo_tier = str(hpo_tier_raw)

        out.append(
            UpliftRow(
                dataset_name=str(dataset_name),
                model_name=str(model_name),
                task_type=str(task_type),
                primary_loss_column=selector.column,
                n_cells_paired=len(per_cell_delta),
                default_mean=float(np.mean(per_cell_default)),
                tuned_mean=float(np.mean(per_cell_tuned)),
                delta=float(np.mean(per_cell_delta)),
                delta_std=delta_std,
                hpo_search_space_size=search_space_size,
                hpo_tier=hpo_tier,
                hpo_n_trials_completed_mean=(
                    float(np.mean(per_cell_n_trials)) if per_cell_n_trials else None
                ),
                hpo_time_to_best_seconds_mean=(
                    float(np.mean(per_cell_time_to_best)) if per_cell_time_to_best else None
                ),
            )
        )

    def _sort_key(row: UpliftRow) -> tuple[str, float]:
        # Sentinel rows (default_only / tuned_only / unpaired) sort
        # to the end of their dataset block via +inf delta.
        if row.delta is None:
            return (row.dataset_name, float("inf"))
        # Negate so larger Δ (more uplift) sorts first per dataset.
        return (row.dataset_name, -row.delta)

    out.sort(key=_sort_key)
    return out


def _build_loss_matrix(rows: list[UpliftRow]) -> pd.DataFrame:
    """Construct the (model, dataset) seed-mean loss matrix for
    Friedman from the aggregated rows.

    Uses the tuned-arm seed-mean loss per (model, dataset). Cells
    with no tuned arm produce NaN, and the Friedman helper drops
    those columns before testing.
    """
    if not rows:
        return pd.DataFrame()
    records: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.tuned_mean is None:
            continue
        records[(row.model_name, row.dataset_name)] = row.tuned_mean
    if not records:
        return pd.DataFrame()
    models = sorted({m for (m, _d) in records})
    datasets = sorted({d for (_m, d) in records})
    matrix = pd.DataFrame(
        index=pd.Index(models, name="model"),
        columns=pd.Index(datasets, name="dataset"),
        dtype=float,
    )
    for (model_name, dataset_name), value in records.items():
        matrix.at[model_name, dataset_name] = value
    return matrix


def _format_value(value: Any, *, decimals: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.{decimals}f}"
    return str(value)


def _format_int(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{int(value)}"
    return str(value)


def _render_dataset_block(dataset_name: str, summaries: list[UpliftRow]) -> str:
    """Markdown for one dataset's per-model uplift block."""
    if not summaries:
        return ""
    task_type = summaries[0].task_type
    assert all(s.task_type == task_type for s in summaries), (
        f"dataset {dataset_name!r} has heterogeneous task_types: "
        f"{sorted({s.task_type for s in summaries})}"
    )
    primary_loss_column = summaries[0].primary_loss_column
    header_cells = [
        "model",
        "n_cells",
        "default_mean",
        "tuned_mean",
        "delta",
        "delta_std",
        "search_space_size",
        "hpo_tier",
        "n_trials_mean",
        "time_to_best_s_mean",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type}, primary={primary_loss_column}, ranked by Δ DESC)",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for summary in summaries:
        if summary.delta is None:
            # Skip sentinel + unpaired rows in the per-dataset block;
            # they land in the footnote.
            continue
        row_cells = [
            summary.model_name,
            str(summary.n_cells_paired),
            _format_value(summary.default_mean),
            _format_value(summary.tuned_mean),
            _format_value(summary.delta),
            _format_value(summary.delta_std),
            _format_int(summary.hpo_search_space_size),
            summary.hpo_tier or "",
            _format_value(summary.hpo_n_trials_completed_mean, decimals=1),
            _format_value(summary.hpo_time_to_best_seconds_mean, decimals=2),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_footnote(rows: list[UpliftRow]) -> str:
    """Footnote listing groups where the Δ could not be computed."""
    incomplete = [r for r in rows if r.delta is None]
    if not incomplete:
        return ""
    lines = ["### Incomplete groups", ""]
    lines.append("| Dataset | Model | Reason |")
    lines.append("| --- | --- | --- |")
    for row in incomplete:
        if row.default_only:
            reason = "default arm only (no tuned cell in manifest)"
        elif row.tuned_only:
            reason = "tuned arm only (no default cell in manifest)"
        elif row.paired_but_no_valid_loss:
            reason = (
                f"both arms present (n_cells_paired={row.n_cells_paired}) "
                "but every paired row had missing primary loss"
            )
        else:
            reason = "no (seed, fold) intersection between default and tuned"
        lines.append(f"| {row.dataset_name} | {row.model_name} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def _render_friedman_block(friedman: FriedmanHolmResult) -> str:
    """Markdown for the Friedman omnibus + Holm-adjusted pairwise."""
    lines = ["### Matrix-level Friedman + Holm", ""]
    if friedman.friedman_statistic is None or friedman.friedman_p_value is None:
        lines.append(
            f"_Friedman omnibus skipped: n_models={friedman.n_models} "
            f"n_datasets={friedman.n_datasets} (need >= 2 of each)._\n"
        )
        return "\n".join(lines)
    lines.append(
        f"Friedman χ² = {friedman.friedman_statistic:.3f}, "
        f"p = {friedman.friedman_p_value:.4f} "
        f"(n_models={friedman.n_models}, n_datasets={friedman.n_datasets})."
    )
    if friedman.pairwise:
        lines.append("")
        lines.append(
            f"Holm-adjusted pairwise (family_size={friedman.family_size}; "
            f"reference vs each other model):"
        )
        lines.append("")
        lines.append("| Comparison | Wilcoxon stat | raw p | Holm-adj p | n_datasets |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in friedman.pairwise:
            lines.append(
                f"| {p.comparison_name} | {p.wilcoxon_statistic:.3f} | "
                f"{p.raw_p_value:.4f} | {p.holm_adjusted_p_value:.4f} | "
                f"{p.n_datasets} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_hpo_uplift_report(
    manifest: pd.DataFrame,
    *,
    reference_model: str = _DEFAULT_REFERENCE_MODEL,
) -> HPOUpliftReport:
    """Build the structured `HPOUpliftReport` from a manifest.

    Aggregates default+tuned rows per (dataset, model) and runs the
    Friedman/Holm matrix-level test against the configured
    reference model. Returns the typed bundle for callers that want
    structured access (the renderer wraps this for the Markdown
    pass).

    On an empty manifest the report has empty rows and `friedman`
    is None.
    """
    rows = tuple(aggregate_hpo_uplift(manifest))
    matrix = _build_loss_matrix(list(rows))
    friedman: FriedmanHolmResult | None
    if matrix.empty or reference_model not in matrix.index:
        friedman = None
    else:
        friedman = friedman_holm(matrix, reference_model=reference_model)
    return HPOUpliftReport(rows=rows, friedman=friedman, reference_model=reference_model)


def render_hpo_uplift_markdown(
    manifest: pd.DataFrame,
    *,
    reference_model: str = _DEFAULT_REFERENCE_MODEL,
) -> str:
    """Render the full HPO-uplift Markdown report.

    Empty manifest → a single-line "no results" Markdown block so
    the caller's downstream pipeline still receives valid Markdown.
    """
    if manifest.empty:
        return (
            "# HPO-uplift report\n\n"
            "_No results in manifest; run "
            "`python -m benchmarks.run --experiment=raw_loss --config "
            "<config.toml>` followed by `--experiment=hpo_uplift`._\n"
        )
    report = build_hpo_uplift_report(manifest, reference_model=reference_model)
    rows = list(report.rows)
    by_dataset: dict[str, list[UpliftRow]] = {}
    for row in rows:
        if row.delta is None:
            continue
        by_dataset.setdefault(row.dataset_name, []).append(row)
    parts = [
        "# HPO-uplift report",
        "",
        (
            "_Δ = default_mean - tuned_mean on the primary loss "
            "(positive Δ means tuning helped). search_space_size is "
            "the registered dimensionality from `benchmarks.hpo.<family>`; "
            "v1 does not normalize across families per design B6.4.0._"
        ),
        "",
    ]
    for dataset_name in sorted(by_dataset):
        parts.append(_render_dataset_block(dataset_name, by_dataset[dataset_name]))
    if report.friedman is None:
        matrix = _build_loss_matrix(rows)
        if matrix.empty:
            parts.append("_Friedman omnibus skipped: empty loss matrix._\n")
        else:
            parts.append(
                f"_Friedman omnibus skipped: reference_model={reference_model!r} "
                f"absent from matrix (have {sorted(matrix.index)})._\n"
            )
    else:
        parts.append(_render_friedman_block(report.friedman))
    footnote = _render_footnote(rows)
    if footnote:
        parts.append(footnote)
    return "\n".join(parts)


def render_hpo_uplift_markdown_with_ci(
    manifest: pd.DataFrame,
    rollup: list[HPOUpliftRollupRow],
    *,
    reference_model: str = _DEFAULT_REFERENCE_MODEL,
    expected_manifest_fingerprint: str | None = None,
    aggregator_error_class: str | None = None,
    manifest_unreadable: bool = False,
) -> str:
    """Render the HPO-uplift report with B15 paired Δ CIs.

    Joins the manifest's UpliftRow data with the rollup table on
    (dataset_name, model_name, task_type) and replaces the bare
    `delta` column with `Δ [95% CI]`.

    Footnote-source precedence (B15.4):

    1. `aggregator_error_class is not None`: std variant +
       "Bootstrap aggregator failed: <class>" footnote.
    2. `not rollup`: std variant silently.
    3. `expected_manifest_fingerprint` set AND any row mismatch:
       std variant + "rollup is stale" footnote.
    4. CI variant body renders; if `manifest_unreadable=True`,
       append "freshness check skipped" footnote.
    5. CI body iteration: if ANY rollup row has
       `n_skipped_cells > 0`, append a "Partial coverage"
       footnote. Independent from sources 1-4.
    """
    if aggregator_error_class is not None:
        std_body = render_hpo_uplift_markdown(manifest, reference_model=reference_model)
        footnote = (
            "\n### Bootstrap aggregator failed\n\n"
            f"_The B15 HPO-uplift bootstrap-rollup aggregator raised "
            f"`{aggregator_error_class}`; the report above is rendered "
            "in the legacy mean-only shape. See the run logs._\n"
        )
        return std_body + footnote

    if not rollup:
        return render_hpo_uplift_markdown(manifest, reference_model=reference_model)

    if expected_manifest_fingerprint is not None and any(
        row.manifest_fingerprint != expected_manifest_fingerprint for row in rollup
    ):
        std_body = render_hpo_uplift_markdown(manifest, reference_model=reference_model)
        footnote = (
            "\n### Bootstrap rollup is stale\n\n"
            "_The B15 HPO-uplift bootstrap-rollup file's "
            "manifest_fingerprint does not match the current manifest's; "
            "the CI variant was suppressed for safety._\n"
        )
        return std_body + footnote

    body = _render_with_ci_hpo_uplift(
        manifest, rollup, reference_model=reference_model
    )
    if manifest_unreadable:
        body += (
            "\n### Bootstrap freshness check skipped\n\n"
            "_The B15 HPO-uplift freshness check could not run because "
            "`run_manifest.json` is corrupt or schema-mismatched._\n"
        )
    return body


def _render_with_ci_hpo_uplift(
    manifest: pd.DataFrame,
    rollup: list[HPOUpliftRollupRow],
    *,
    reference_model: str,
) -> str:
    """Joins the manifest's UpliftRow data with the rollup; emits
    the per-dataset table with the CI column replacing the bare
    Δ; preserves the Friedman/Holm block unchanged."""
    report = build_hpo_uplift_report(manifest, reference_model=reference_model)
    rollup_index: dict[tuple[str, str, str], HPOUpliftRollupRow] = {}
    for row in rollup:
        rollup_index[(row.dataset_name, row.model_name, row.task_type)] = row

    rows = list(report.rows)
    by_dataset: dict[str, list[UpliftRow]] = {}
    for row in rows:
        if row.delta is None:
            continue
        by_dataset.setdefault(row.dataset_name, []).append(row)

    parts = [
        "# HPO-uplift report",
        "",
        (
            "_Δ = default_mean - tuned_mean on the primary loss "
            "(positive Δ means tuning helped). search_space_size is "
            "the registered dimensionality from `benchmarks.hpo.<family>`; "
            "v1 does not normalize across families per design B6.4.0._"
        ),
        "",
    ]
    for dataset_name in sorted(by_dataset):
        parts.append(
            _render_dataset_block_with_ci(
                dataset_name, by_dataset[dataset_name], rollup_index
            )
        )

    if report.friedman is None:
        matrix = _build_loss_matrix(rows)
        if matrix.empty:
            parts.append("_Friedman omnibus skipped: empty loss matrix._\n")
        else:
            parts.append(
                f"_Friedman omnibus skipped: reference_model={reference_model!r} "
                f"absent from matrix (have {sorted(matrix.index)})._\n"
            )
    else:
        parts.append(_render_friedman_block(report.friedman))

    footnote = _render_footnote(rows)
    if footnote:
        parts.append(footnote)

    # Source 5: partial-coverage footnote (loss-dropout disclosure).
    partial_coverage = _render_partial_coverage_footnote(rollup)
    if partial_coverage:
        parts.append(partial_coverage)

    # New no_paired_cells sentinel rows route via the shared
    # render_rollup_skipped_footnote helper. The three existing
    # B8 sentinels (default_only, tuned_only,
    # paired_but_no_valid_loss) flow through _render_footnote
    # above on the UpliftRow side; we surface ONLY the
    # no_paired_cells sentinel here.
    no_paired = [r for r in rollup if r.bootstrap_skipped_reason == "no_paired_cells"]
    if no_paired:
        parts.append(
            render_rollup_skipped_footnote(
                no_paired,
                group_columns=("dataset_name", "model_name"),
                header_labels=("Dataset", "Model"),
            )
        )

    return "\n".join(parts)


def _render_dataset_block_with_ci(
    dataset_name: str,
    summaries: list[UpliftRow],
    rollup_index: dict[tuple[str, str, str], HPOUpliftRollupRow],
) -> str:
    """CI variant of the per-dataset block. Replaces the bare
    `delta` column with `Δ [95% CI]`. Keeps default_mean +
    tuned_mean + secondary columns as-is."""
    if not summaries:
        return ""
    task_type = summaries[0].task_type
    primary_loss_column = summaries[0].primary_loss_column
    header_cells = [
        "model",
        "n_cells",
        "default_mean",
        "tuned_mean",
        "Δ [95% CI]",
        "search_space_size",
        "hpo_tier",
        "n_trials_mean",
        "time_to_best_s_mean",
    ]
    sep_cells = ["---"] * len(header_cells)
    lines = [
        f"### {dataset_name} ({task_type}, primary={primary_loss_column}, ranked by Δ DESC)",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for summary in summaries:
        if summary.delta is None:
            continue
        key = (summary.dataset_name, summary.model_name, summary.task_type)
        rollup_row = rollup_index.get(key)
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
            summary.model_name,
            str(summary.n_cells_paired),
            _format_value(summary.default_mean),
            _format_value(summary.tuned_mean),
            ci_cell,
            _format_int(summary.hpo_search_space_size),
            summary.hpo_tier or "",
            _format_value(summary.hpo_n_trials_completed_mean, decimals=1),
            _format_value(summary.hpo_time_to_best_seconds_mean, decimals=2),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_partial_coverage_footnote(rollup: list[HPOUpliftRollupRow]) -> str:
    """Source 5: when ANY non-sentinel rollup row has
    n_skipped_cells > 0, surface a "Partial coverage" footnote
    listing per-(dataset, model) the n_skipped_cells /
    n_cells_paired ratio."""
    partial = [
        r
        for r in rollup
        if r.bootstrap_skipped_reason is None and r.n_skipped_cells > 0
    ]
    if not partial:
        return ""
    lines = ["### Partial coverage", ""]
    lines.append("| Dataset | Model | n_skipped_cells / n_cells_paired |")
    lines.append("| --- | --- | --- |")
    for r in partial:
        lines.append(
            f"| {r.dataset_name} | {r.model_name} | "
            f"{r.n_skipped_cells} / {r.n_cells_paired} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_from_dir(
    output_root: Path,
    *,
    reference_model: str = _DEFAULT_REFERENCE_MODEL,
) -> str:
    """Convenience: load the manifest under `output_root` and render.

    The CI variant kicks in when a rollup file is present AND the
    rollup's `manifest_fingerprint` matches the live run manifest's
    (B15.4 freshness check). When the CLI wrapper caught a
    `RawRollupError` and wrote the failure sentinel, the renderer
    surfaces the std + footnote.
    """
    manifest = load_run(output_root)

    failure_sentinel = hpo_uplift_aggregator_failed_sentinel_path(output_root)
    if failure_sentinel.exists():
        error_class = failure_sentinel.read_text(encoding="utf-8").strip()
        return render_hpo_uplift_markdown_with_ci(
            manifest, [], reference_model=reference_model, aggregator_error_class=error_class
        )

    if hpo_uplift_rollup_path(output_root).exists():
        rollup = load_hpo_uplift_rollup(output_root)
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
            return render_hpo_uplift_markdown_with_ci(
                manifest,
                rollup,
                reference_model=reference_model,
                expected_manifest_fingerprint=expected,
                manifest_unreadable=manifest_unreadable,
            )
    return render_hpo_uplift_markdown(manifest, reference_model=reference_model)
