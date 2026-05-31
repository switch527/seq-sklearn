"""Phase B14 training-time bootstrap-CI aggregator (D-B13.2).

Reads the B5 raw-loss manifest (NOT a separate training-time
shard: `run_training_time` is itself a report-only thin pass over
the B5 manifest per `benchmarks/experiments/training_time.py`,
so the data the CI rollup needs is the same `wall_seconds`
column `run_raw_loss` already wrote). Groups by
`(dataset_name, model_name, hardware_tier, task_type)`, filters
to OK cells, extracts per-cell `wall_seconds`, calls the entity-
block bootstrap with the cell index as the entity id, and emits
one `TrainingTimeRollupRow` per group.

Design contracts (`docs/benchmark_suite_phase_log.md` phase
B14; full delta at the B14 merge commit per the log):

- Reuses the B13 `RawRollupError(RuntimeError)` typed failure
  surface; the CLI wrapper at
  `benchmarks/run.py:_run_bootstrap_training_time_rollup`
  catches and drops a sentinel file independently from the B5
  and B6 sentinels.
- Reuses `RunManifest.fingerprint()` for freshness.
- Reuses the B13 OOM ceiling and profile-default n_resamples.
- The CI wrapper attaches at the SAME hook point as
  `_run_bootstrap_rollup` (after `run_raw_loss` succeeds), not
  after `run_training_time`. The training-time renderer is a
  separate downstream step.
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from benchmarks.bootstrap_manifest import (
    TrainingTimeRollupRow,
    write_training_time_rollup,
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
    "aggregate_bootstrap_training_time_rollup",
    "is_training_time_rollup_enabled",
    "training_time_rollup_output_path",
]

_PRIMARY_METRIC = "wall_seconds"


def is_training_time_rollup_enabled(config: BenchmarkConfig) -> bool:
    """True iff any ExperimentSpec(kind="training_time") has
    bootstrap_training_time_enabled=True. Public so the CLI
    wrapper can gate the dispatch.

    The wrapper attaches AFTER `run_raw_loss` (not after
    `run_training_time`) because the data lives in the B5
    manifest; the gate is a `training_time` spec with
    `bootstrap_training_time_enabled=True` (the field default)."""
    return any(
        spec.kind == "training_time" and spec.bootstrap_training_time_enabled
        for spec in config.experiments
    )


def training_time_rollup_output_path() -> str:
    """Stable filename token for log messages."""
    return "bootstrap_training_time_rollup.parquet"


def _emit_sentinel_row(
    *,
    dataset_name: str,
    model_name: str,
    hardware_tier: str,
    task_type: str,
    n_seeds: int,
    n_cells_evaluated: int,
    n_skipped_cells: int,
    bootstrap_skipped_reason: str,
    n_resamples: int,
    manifest_fingerprint: str,
) -> TrainingTimeRollupRow:
    return TrainingTimeRollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        hardware_tier=hardware_tier,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells_evaluated,
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
    hardware_tier: str,
    task_type: str,
    n_resamples: int,
    manifest_fingerprint: str,
    per_fold_enabled: bool,
) -> TrainingTimeRollupRow:
    """One training-time rollup row for a (dataset, model,
    hardware_tier, task) group."""
    n_seeds = int(block["seed"].nunique())
    ok = block.loc[block["skipped_reason"].isna()]
    n_skipped = int(block.shape[0] - ok.shape[0])

    if ok.empty:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_name=model_name,
            hardware_tier=hardware_tier,
            task_type=task_type,
            n_seeds=n_seeds,
            n_cells_evaluated=0,
            n_skipped_cells=n_skipped,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )

    if "wall_seconds" not in ok.columns:
        raise RawRollupError(
            f"aggregate_bootstrap_training_time_rollup: dataset={dataset_name!r} "
            f"model={model_name!r} OK block has no `wall_seconds` column"
        )
    wall_seconds_series = ok["wall_seconds"].astype(float)
    if wall_seconds_series.isna().any():
        # OK cells must have a measured wall_seconds; a NaN
        # indicates corrupt B5 data and raises.
        raise RawRollupError(
            f"aggregate_bootstrap_training_time_rollup: dataset={dataset_name!r} "
            f"model={model_name!r} hardware_tier={hardware_tier!r} has an OK cell "
            "with NaN wall_seconds"
        )

    values = wall_seconds_series.to_numpy()
    n_cells = values.shape[0]

    # OOM gate (reuses B13's ceiling).
    if n_cells * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING:
        raise RawRollupError(
            f"aggregate_bootstrap_training_time_rollup: dataset={dataset_name!r} "
            f"with n_cells={n_cells} * n_resamples={n_resamples} exceeds the "
            f"bootstrap-row-count ceiling ({BOOTSTRAP_ROW_COUNT_CEILING})"
        )

    # B14.0 cell-as-entity contract: each cell is its own entity.
    entity_ids = np.arange(n_cells, dtype=np.int64)
    mean, ci_lo, ci_hi, fallback_reason = entity_block_bootstrap_ci(
        values,
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
            losses=values,
            entities=entity_ids,
            fold_indices=ok["fold_index"].to_numpy().astype(np.int64),
            seeds=ok["seed"].to_numpy().astype(np.int64),
            n_resamples=n_resamples,
            confidence=BOOTSTRAP_CONFIDENCE,
            base_seed=BOOTSTRAP_DEFAULT_SEED,
            fold_seed_offset=_bootstrap_aggregate.BOOTSTRAP_PER_FOLD_SEED_OFFSET,
            ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
            metric_fn=lambda x: float(np.nanmean(x)),
            dataset_name=dataset_name,
            kind="training_time",
        )

    return TrainingTimeRollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        hardware_tier=hardware_tier,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells,
        n_skipped_cells=n_skipped,
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


def aggregate_bootstrap_training_time_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[TrainingTimeRollupRow]:
    """Aggregate the B5 manifest into a list of training-time CI rollup rows.

    Returns `[]` when the B5 manifest is absent or empty.

    Args:
        config: the live BenchmarkConfig.
        output_root: the run output root.
        env: the RunEnvironment; profile sets the default n_resamples.
        manifest: the loaded RunManifest; fingerprint goes on every row.

    Raises:
        RawRollupError: a malformed OK cell or an OOM-gated row count.
    """
    b5_df = load_run(output_root)
    if b5_df.empty:
        return []

    profile = env.profile if hasattr(env, "profile") else "standard"
    n_resamples = resolve_n_resamples(config.experiments, str(profile), kind="training_time")
    per_fold_enabled = _bootstrap_aggregate.spec_has_per_fold_cis_enabled(
        config.experiments, kind="training_time"
    )
    manifest_fingerprint = manifest.fingerprint()

    rows: list[TrainingTimeRollupRow] = []
    grouped = b5_df.groupby(["dataset_name", "model_name", "hardware_tier", "task_type"], sort=True)
    for group_key, block in grouped:
        dataset_name, model_name, hardware_tier, task_type = cast(
            tuple[str, str, str, str], group_key
        )
        rows.append(
            _build_group_rollup(
                block,
                dataset_name=dataset_name,
                model_name=model_name,
                hardware_tier=hardware_tier,
                task_type=task_type,
                n_resamples=n_resamples,
                manifest_fingerprint=manifest_fingerprint,
                per_fold_enabled=per_fold_enabled,
            )
        )

    write_training_time_rollup(output_root, rows)
    logger.info(
        "aggregate_bootstrap_training_time_rollup: wrote %d rollup rows to %s",
        len(rows),
        training_time_rollup_output_path(),
    )
    return rows
