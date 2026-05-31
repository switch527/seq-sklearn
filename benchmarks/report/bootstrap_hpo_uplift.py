"""Phase B15 HPO-uplift bootstrap-CI aggregator (D-B13.3).

Reads the B5 manifest (which the B8 driver writes into with
`variant="tuned"` rows alongside the existing `variant="default"`
rows), pairs cells per (dataset, model, seed, fold_index) via
an inner-join on the variant axis, computes per-cell
Δ = `default_loss - tuned_loss`, and bootstraps the per-cell
deltas using the entity-block primitive with cell-index as
entity-id.

Design contracts (`docs/benchmark_suite_phase_log.md` phase
B15; full delta at the B15 merge commit per the log):

- Reuses B13 `RawRollupError(RuntimeError)` typed failure.
- Reuses `RunManifest.fingerprint()` for freshness.
- Reuses B13 `BOOTSTRAP_ROW_COUNT_CEILING` via shared
  `_bootstrap_aggregate` constants + `resolve_n_resamples`.
- Sentinel reason strings: "default_only", "tuned_only",
  "paired_but_no_valid_loss" (inherited from B8 UpliftRow
  flags) + "no_paired_cells" (new, inner-join yields zero).
- Raise conditions: malformed paired cell (NaN loss + group
  not flagged paired_but_no_valid_loss); duplicate (seed,
  fold) pair in a single variant arm; OOM ceiling.
- Wrapper Gate D: aggregator returns `[]` when the manifest
  has no `variant=tuned` rows (B8 was not run); the wrapper
  treats this as a no-op skip (no failure sentinel).
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from benchmarks.bootstrap_manifest import (
    HPOUpliftRollupRow,
    write_hpo_uplift_rollup,
)
from benchmarks.config import BenchmarkConfig
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.manifest import load_run
from benchmarks.metrics.bootstrap import (
    BOOTSTRAP_RNG_ALGORITHM,
    entity_block_bootstrap_ci,
)
from benchmarks.report import _bootstrap_aggregate
from benchmarks.report._bootstrap_aggregate import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_DEFAULT_SEED,
    BOOTSTRAP_ROW_COUNT_CEILING,
    numpy_version,
    resolve_n_resamples,
)
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import RunManifest

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_bootstrap_hpo_uplift_rollup",
    "hpo_uplift_rollup_output_path",
    "is_hpo_uplift_rollup_enabled",
]

_PRIMARY_METRIC = "delta"
_DEFAULT_VARIANT = "default"
_TUNED_VARIANT = "tuned"

# Map B8 task_type to the primary loss column for audit. B8's
# `regression_quantile` cells are early-skipped upstream so they
# never reach this aggregator (the manifest carries them as
# `regression_quantile_b5_followup` skipped_reason).
_PRIMARY_LOSS_COLUMN_BY_TASK: dict[str, str] = {
    "binary": "log_loss",
    "multiclass": "log_loss",
    "regression_point": "rmse",
}


def is_hpo_uplift_rollup_enabled(config: BenchmarkConfig) -> bool:
    """True iff any ExperimentSpec(kind="hpo_uplift") has
    bootstrap_hpo_uplift_enabled=True."""
    return any(
        spec.kind == "hpo_uplift" and spec.bootstrap_hpo_uplift_enabled
        for spec in config.experiments
    )


def hpo_uplift_rollup_output_path() -> str:
    """Stable filename token for log messages."""
    return "bootstrap_hpo_uplift_rollup.parquet"


def _check_duplicate_pairs(
    block: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
) -> None:
    """Within a single (dataset, model, variant) group, raise on
    duplicate (seed, fold_index) pairs."""
    for variant_value in (_DEFAULT_VARIANT, _TUNED_VARIANT):
        variant_block = block.loc[block["variant"] == variant_value]
        if variant_block.empty:
            continue
        pairs = variant_block[["seed", "fold_index"]]
        if pairs.duplicated().any():
            raise RawRollupError(
                f"aggregate_bootstrap_hpo_uplift_rollup: dataset={dataset_name!r} "
                f"model={model_name!r} variant={variant_value!r} has a duplicate "
                "(seed, fold_index) pair in the manifest"
            )


def _emit_sentinel_row(
    *,
    dataset_name: str,
    model_name: str,
    task_type: str,
    primary_loss_column: str,
    n_seeds: int,
    n_folds: int,
    n_cells_paired: int,
    n_skipped_cells: int,
    bootstrap_skipped_reason: str,
    n_resamples: int,
    manifest_fingerprint: str,
) -> HPOUpliftRollupRow:
    return HPOUpliftRollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        primary_loss_column=primary_loss_column,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells_paired,
        n_skipped_cells=n_skipped_cells,
        primary_metric_mean=None,
        primary_metric_ci_lo=None,
        primary_metric_ci_hi=None,
        bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
        bootstrap_n_resamples=n_resamples,
        bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
        bootstrap_ci_fallback_reason=None,
        per_fold_cis=None,
        bootstrap_numpy_version=numpy_version(),
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


def _build_group_rollup(
    block: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    task_type: str,
    n_resamples: int,
    manifest_fingerprint: str,
    per_fold_enabled: bool,
) -> HPOUpliftRollupRow:
    """One HPO-uplift rollup row per (dataset, model, task_type) group."""
    primary_loss_column = _PRIMARY_LOSS_COLUMN_BY_TASK.get(task_type, "log_loss")
    ok = block.loc[block["skipped_reason"].isna()].copy()
    n_seeds = int(ok["seed"].nunique()) if not ok.empty else 0
    n_folds = int(ok["fold_index"].nunique()) if not ok.empty else 0

    _check_duplicate_pairs(ok, dataset_name=dataset_name, model_name=model_name)

    default_block = ok.loc[ok["variant"] == _DEFAULT_VARIANT]
    tuned_block = ok.loc[ok["variant"] == _TUNED_VARIANT]

    if default_block.empty and tuned_block.empty:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_loss_column=primary_loss_column,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_cells_paired=0,
            n_skipped_cells=0,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )
    if default_block.empty:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_loss_column=primary_loss_column,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_cells_paired=0,
            n_skipped_cells=0,
            bootstrap_skipped_reason="tuned_only",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )
    if tuned_block.empty:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_loss_column=primary_loss_column,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_cells_paired=0,
            n_skipped_cells=0,
            bootstrap_skipped_reason="default_only",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )

    # Inner-join on (seed, fold_index).
    join_cols = ["seed", "fold_index"]
    paired = default_block.merge(
        tuned_block,
        on=join_cols,
        suffixes=("_default", "_tuned"),
        how="inner",
    )
    if paired.empty:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_loss_column=primary_loss_column,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_cells_paired=0,
            n_skipped_cells=0,
            bootstrap_skipped_reason="no_paired_cells",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )

    default_losses = paired[f"{primary_loss_column}_default"].astype(float).to_numpy()
    tuned_losses = paired[f"{primary_loss_column}_tuned"].astype(float).to_numpy()
    pair_nan_mask = np.isnan(default_losses) | np.isnan(tuned_losses)
    n_paired_total = int(paired.shape[0])
    n_skipped_cells = int(pair_nan_mask.sum())

    if pair_nan_mask.all():
        # Every paired cell has NaN; treat as the B8
        # paired_but_no_valid_loss sentinel.
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_loss_column=primary_loss_column,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_cells_paired=n_paired_total,
            n_skipped_cells=n_skipped_cells,
            bootstrap_skipped_reason="paired_but_no_valid_loss",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )
    if pair_nan_mask.any():
        # Mixed NaN: spec mandates raise (B15.2 malformed-paired-cell)
        # because a group with SOME NaN paired cells indicates a
        # corrupt manifest (the B8 driver would have flagged the
        # whole group as paired_but_no_valid_loss if every refit
        # crashed).
        raise RawRollupError(
            f"aggregate_bootstrap_hpo_uplift_rollup: dataset={dataset_name!r} "
            f"model={model_name!r} has a paired cell with NaN primary loss "
            "on one side; the upstream B8 group is not flagged "
            "paired_but_no_valid_loss"
        )

    # All paired cells are clean. Compute per-cell delta and bootstrap.
    deltas = default_losses - tuned_losses
    n_cells = deltas.shape[0]

    if n_cells * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING:
        raise RawRollupError(
            f"aggregate_bootstrap_hpo_uplift_rollup: dataset={dataset_name!r} "
            f"with n_cells={n_cells} * n_resamples={n_resamples} exceeds the "
            f"bootstrap-row-count ceiling ({BOOTSTRAP_ROW_COUNT_CEILING})"
        )

    entity_ids = np.arange(n_cells, dtype=np.int64)
    mean, ci_lo, ci_hi, fallback_reason = entity_block_bootstrap_ci(
        deltas,
        entity_ids,
        n_resamples=n_resamples,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_DEFAULT_SEED,
        ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
    )

    # B22 / D-B16.3: per-fold CIs (opt-in).
    per_fold_cis = None
    if per_fold_enabled:
        per_fold_cis = _bootstrap_aggregate.compute_per_fold_cis(
            losses=deltas,
            entities=entity_ids,
            fold_indices=paired["fold_index"].to_numpy().astype(np.int64),
            seeds=paired["seed"].to_numpy().astype(np.int64),
            n_resamples=n_resamples,
            confidence=BOOTSTRAP_CONFIDENCE,
            base_seed=BOOTSTRAP_DEFAULT_SEED,
            fold_seed_offset=_bootstrap_aggregate.BOOTSTRAP_PER_FOLD_SEED_OFFSET,
            ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
            metric_fn=lambda x: float(np.nanmean(x)),
            dataset_name=dataset_name,
            kind="hpo_uplift",
        )

    return HPOUpliftRollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        primary_loss_column=primary_loss_column,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells,
        n_skipped_cells=n_skipped_cells,
        primary_metric_mean=mean,
        primary_metric_ci_lo=ci_lo,
        primary_metric_ci_hi=ci_hi,
        bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
        bootstrap_n_resamples=n_resamples,
        bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
        bootstrap_ci_fallback_reason=fallback_reason,
        per_fold_cis=per_fold_cis,
        bootstrap_numpy_version=numpy_version(),
        bootstrap_skipped_reason=None,
        manifest_fingerprint=manifest_fingerprint,
    )


def aggregate_bootstrap_hpo_uplift_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[HPOUpliftRollupRow]:
    """Aggregate the HPO-uplift Δ CI rollup from the B5 manifest.

    Gate D: when the manifest has no `variant=tuned` rows (B8 was
    not run), return `[]`. The wrapper treats this as a no-op
    skip (no failure sentinel written).

    Raises:
        RawRollupError: malformed paired cell, duplicate (seed,
            fold) pair, or OOM ceiling.
    """
    df = load_run(output_root)
    if df.empty:
        return []
    # Gate D: no tuned-variant rows means B8 was not run.
    if "variant" not in df.columns or not (df["variant"] == _TUNED_VARIANT).any():
        return []

    n_resamples = resolve_n_resamples(config.experiments, env.profile, kind="hpo_uplift")
    per_fold_enabled = _bootstrap_aggregate.spec_has_per_fold_cis_enabled(
        config.experiments, kind="hpo_uplift"
    )
    manifest_fingerprint = manifest.fingerprint()

    rows: list[HPOUpliftRollupRow] = []
    grouped = df.groupby(["dataset_name", "model_name", "task_type"], sort=True)
    for group_key, block in grouped:
        dataset_name, model_name, task_type = cast(tuple[str, str, str], group_key)
        # Skip regression_quantile cells entirely; they don't have a
        # primary loss column resolution path here. B8 already flags
        # them upstream with the regression_quantile_b5_followup
        # skipped reason, so they're absent from the OK block.
        if task_type not in _PRIMARY_LOSS_COLUMN_BY_TASK:
            continue
        rows.append(
            _build_group_rollup(
                block,
                dataset_name=dataset_name,
                model_name=model_name,
                task_type=task_type,
                n_resamples=n_resamples,
                per_fold_enabled=per_fold_enabled,
                manifest_fingerprint=manifest_fingerprint,
            )
        )

    write_hpo_uplift_rollup(output_root, rows)
    logger.info(
        "aggregate_bootstrap_hpo_uplift_rollup: wrote %d rollup rows to %s",
        len(rows),
        hpo_uplift_rollup_output_path(),
    )
    return rows
