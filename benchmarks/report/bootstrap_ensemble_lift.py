"""Phase B16 ensemble-lift bootstrap-CI aggregator (D-B13.4).

Reads the B5 manifest, partitions cells into seq-family and
baseline-family rosters via the existing B11 helpers
(`model_families`, `build_cells_table`), and for each dataset
calls the shared `compute_per_cell_lift_deltas` helper to compute
the per-(seed, fold) Δloss records. Bootstraps the per-cell
`delta_loss` array using the entity-block primitive with
cell-index as entity-id.

Design contracts (`docs/benchmark_suite_design_b16_delta.md`
sections B16.0, B16.2, B16.5):

- Reuses B13 `RawRollupError(RuntimeError)` typed failure.
- Reuses `RunManifest.fingerprint()` for freshness.
- Reuses `BOOTSTRAP_ROW_COUNT_CEILING` via shared
  `_bootstrap_aggregate` constants + `resolve_n_resamples`.
- THREE sentinel reason strings (matches B11's actual flag
  pair semantics): `"no_gbm_predictions"`,
  `"no_seq_predictions"`, `"all_cells_skipped_in_manifest"`.
  When both B11 flags trip, the aggregator emits the
  lexically-first sentinel (`no_gbm_predictions`) for
  determinism.
- Raise conditions: malformed paired cell (delta None despite
  no skipped_reason); duplicate (seed, fold) pair in either
  family arm; OOM ceiling.
- Wrapper Gate D: aggregator returns `[]` when the manifest
  has zero OK rows mapping to either the seq family OR the
  baseline family. The wrapper treats this as a no-op skip.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    write_ensemble_lift_rollup,
)
from benchmarks.config import BenchmarkConfig
from benchmarks.experiments.ensemble_lift import (
    DEFAULT_BASELINE_FAMILY,
    DEFAULT_SEQ_FAMILY,
    build_cells_table,
    compute_per_cell_lift_deltas,
    model_families,
)
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.manifest import load_run
from benchmarks.metrics.bootstrap import (
    BOOTSTRAP_RNG_ALGORITHM,
    entity_block_bootstrap_ci,
)
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
    "aggregate_bootstrap_ensemble_lift_rollup",
    "ensemble_lift_rollup_output_path",
    "is_ensemble_lift_rollup_enabled",
]

_PRIMARY_METRIC = "delta_loss"

# Map B11 task_type to the primary loss column for audit.
# regression_quantile is not supported (B11 inherits B5's skip).
_PRIMARY_LOSS_COLUMN_BY_TASK: dict[str, str] = {
    "binary": "log_loss",
    "multiclass": "log_loss",
    "regression_point": "rmse",
}


def is_ensemble_lift_rollup_enabled(config: BenchmarkConfig) -> bool:
    """True iff any ExperimentSpec(kind="ensemble_lift") has
    bootstrap_ensemble_lift_enabled=True."""
    return any(
        spec.kind == "ensemble_lift" and spec.bootstrap_ensemble_lift_enabled
        for spec in config.experiments
    )


def ensemble_lift_rollup_output_path() -> str:
    """Stable filename token for log messages."""
    return "bootstrap_ensemble_lift_rollup.parquet"


def _check_duplicate_pairs(
    cells: pd.DataFrame,
    *,
    dataset_name: str,
    family_label: str,
) -> None:
    """Raise on duplicate (seed, fold_index) pairs within a single
    (dataset, family) cell roster.

    The B11 per-cell join machinery rejects duplicate
    `panel_row_index` values inside a single predictions shard,
    but duplicate (seed, fold) rows in the OK manifest would
    silently double-count a cell in the bootstrap. Mirrors B15's
    `_check_duplicate_pairs` (parity, not redundancy)."""
    if cells.empty:
        return
    pairs = cells[["seed", "fold_index"]]
    if pairs.duplicated().any():
        raise RawRollupError(
            f"aggregate_bootstrap_ensemble_lift_rollup: dataset="
            f"{dataset_name!r} family={family_label!r} has a duplicate "
            "(seed, fold_index) pair in the OK manifest"
        )


def _emit_sentinel_row(
    *,
    dataset_name: str,
    task_type: str,
    primary_loss_column: str,
    n_seeds: int,
    n_folds: int,
    n_cells_paired: int,
    n_skipped_cells: int,
    bootstrap_skipped_reason: str,
    n_resamples: int,
    manifest_fingerprint: str,
) -> EnsembleLiftRollupRow:
    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        primary_loss_column=primary_loss_column,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells_paired,
        n_skipped_cells=n_skipped_cells,
        primary_loss_mean=None,
        primary_loss_ci_lo=None,
        primary_loss_ci_hi=None,
        bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
        bootstrap_n_resamples=n_resamples,
        bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_numpy_version=numpy_version(),
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


def _build_dataset_rollup(
    *,
    dataset_name: str,
    task_type: str,
    seed_fold_pairs: list[tuple[int, int]],
    gbm_cells: pd.DataFrame,
    seq_cells: pd.DataFrame,
    output_root: Path,
    n_resamples: int,
    manifest_fingerprint: str,
) -> EnsembleLiftRollupRow:
    """One ensemble-lift rollup row per dataset.

    The caller (`aggregate_bootstrap_ensemble_lift_rollup`) early-
    continues on unsupported `task_type` values, so direct
    subscription here is safe; a missing key indicates a caller
    contract violation, not a runtime fallback case.
    """
    primary_loss_column = _PRIMARY_LOSS_COLUMN_BY_TASK[task_type]
    n_seeds = int(pd.concat([gbm_cells["seed"], seq_cells["seed"]], ignore_index=True).nunique())
    n_folds = int(
        pd.concat([gbm_cells["fold_index"], seq_cells["fold_index"]], ignore_index=True).nunique()
    )

    _check_duplicate_pairs(gbm_cells, dataset_name=dataset_name, family_label="gbm")
    _check_duplicate_pairs(seq_cells, dataset_name=dataset_name, family_label="seq")

    computed = compute_per_cell_lift_deltas(
        dataset_name=dataset_name,
        task_type=task_type,
        seed_fold_pairs=seed_fold_pairs,
        gbm_cells=gbm_cells,
        seq_cells=seq_cells,
        output_root=output_root,
    )

    if not computed.cells:
        # Per design B16.0: when both B11 flags trip, emit the
        # lexically-first sentinel. The two flags map to
        # sentinel strings whose lexical order matches the
        # if/elif priority below (`"no_gbm_predictions"` <
        # `"no_seq_predictions"`), so the cascade is
        # observationally equivalent to a `sorted(...)[0]`
        # selection. A future flag rename that breaks that
        # coincidence must update this cascade in lockstep.
        candidate_reasons: list[str] = []
        if computed.seen_no_gbm:
            candidate_reasons.append("no_gbm_predictions")
        if computed.seen_no_seq:
            candidate_reasons.append("no_seq_predictions")
        reason = (
            sorted(candidate_reasons)[0] if candidate_reasons else "all_cells_skipped_in_manifest"
        )
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            task_type=task_type,
            primary_loss_column=primary_loss_column,
            n_seeds=n_seeds,
            n_folds=n_folds,
            n_cells_paired=0,
            n_skipped_cells=0,
            bootstrap_skipped_reason=reason,
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )

    deltas = np.array([c.delta_loss for c in computed.cells], dtype=float)
    n_cells = deltas.shape[0]

    if not np.isfinite(deltas).all():
        # `compute_per_cell_lift_deltas` only emits finite-loss
        # cells; a non-finite delta here indicates predictions-
        # shard corruption upstream.
        raise RawRollupError(
            f"aggregate_bootstrap_ensemble_lift_rollup: dataset="
            f"{dataset_name!r} has a paired cell with a non-finite "
            "delta_loss; the upstream predictions shard is corrupt"
        )

    if n_cells * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING:
        raise RawRollupError(
            f"aggregate_bootstrap_ensemble_lift_rollup: dataset="
            f"{dataset_name!r} with n_cells={n_cells} * n_resamples="
            f"{n_resamples} exceeds the bootstrap-row-count ceiling "
            f"({BOOTSTRAP_ROW_COUNT_CEILING})"
        )

    entity_ids = np.arange(n_cells, dtype=np.int64)
    mean, ci_lo, ci_hi = entity_block_bootstrap_ci(
        deltas,
        entity_ids,
        n_resamples=n_resamples,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_DEFAULT_SEED,
    )

    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        primary_loss_column=primary_loss_column,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells,
        n_skipped_cells=0,
        primary_loss_mean=mean,
        primary_loss_ci_lo=ci_lo,
        primary_loss_ci_hi=ci_hi,
        bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
        bootstrap_n_resamples=n_resamples,
        bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_numpy_version=numpy_version(),
        bootstrap_skipped_reason=None,
        manifest_fingerprint=manifest_fingerprint,
    )


def aggregate_bootstrap_ensemble_lift_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[EnsembleLiftRollupRow]:
    """Aggregate the ensemble-lift Δloss CI rollup from the B5 manifest.

    Gate D: when the manifest has no OK rows mapping to either
    the seq family OR the baseline family, return `[]`. The
    wrapper treats this as a no-op skip (no failure sentinel
    written).

    Raises:
        RawRollupError: malformed paired cell (non-finite delta
            despite no upstream skipped_reason), duplicate
            (seed, fold) pair in either family arm, or OOM
            ceiling exceeded.
    """
    df = load_run(output_root)
    if df.empty:
        return []

    families = model_families(df)
    gbm_cells_all = build_cells_table(df, families=families, target_family=DEFAULT_BASELINE_FAMILY)
    seq_cells_all = build_cells_table(df, families=families, target_family=DEFAULT_SEQ_FAMILY)

    # Gate D: both families need at least one OK row in the
    # manifest. Either-empty -> no-op skip (no failure sentinel).
    if gbm_cells_all.empty or seq_cells_all.empty:
        return []

    n_resamples = resolve_n_resamples(config.experiments, env.profile, kind="ensemble_lift")
    manifest_fingerprint = manifest.fingerprint()

    rows: list[EnsembleLiftRollupRow] = []
    dataset_names = sorted({str(name) for name in df["dataset_name"].dropna().unique()})
    for dataset_name in dataset_names:
        ds_block = df.loc[df["dataset_name"] == dataset_name]
        task_type_values = ds_block["task_type"].dropna().unique()
        if len(task_type_values) != 1:
            continue
        task_type = str(task_type_values[0])
        if task_type not in _PRIMARY_LOSS_COLUMN_BY_TASK:
            continue
        seed_fold_pairs: list[tuple[int, int]] = sorted(
            {
                (int(s), int(f))
                for s, f in zip(
                    ds_block["seed"].to_numpy(),
                    ds_block["fold_index"].to_numpy(),
                    strict=True,
                )
                if pd.notna(s) and pd.notna(f)
            }
        )
        gbm_ds = gbm_cells_all.loc[gbm_cells_all["dataset_name"] == dataset_name]
        seq_ds = seq_cells_all.loc[seq_cells_all["dataset_name"] == dataset_name]
        rows.append(
            _build_dataset_rollup(
                dataset_name=dataset_name,
                task_type=task_type,
                seed_fold_pairs=seed_fold_pairs,
                gbm_cells=gbm_ds,
                seq_cells=seq_ds,
                output_root=output_root,
                n_resamples=n_resamples,
                manifest_fingerprint=manifest_fingerprint,
            )
        )

    write_ensemble_lift_rollup(output_root, rows)
    logger.info(
        "aggregate_bootstrap_ensemble_lift_rollup: wrote %d rollup rows to %s",
        len(rows),
        ensemble_lift_rollup_output_path(),
    )
    return rows
