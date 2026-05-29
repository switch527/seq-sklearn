"""Phase B14 pairwise bootstrap-CI aggregator (D-B13.1).

Reads the B6 pairwise manifest (via
`benchmarks.experiments.ensemble.load_pairwise`), groups by
`(dataset_name, model_a, model_b, task_type)`, filters to OK
cells (`skipped_reason is None`), extracts the per-cell
`complementarity_score` for classification, calls the entity-
block bootstrap with the cell index as the entity id (each
(seed, fold) cell is its own entity; degenerates to row
bootstrap over cells), and emits one `PairwiseRollupRow` per
group.

Regression cells emit a sentinel row with
`bootstrap_skipped_reason="regression_complementarity_undefined"`
because `complementarity_score` is defined only for
classification per B6.

Design contracts (`docs/benchmark_suite_design_b14_delta.md`
sections B14.0, B14.2):

- Reuses the B13 `RawRollupError(RuntimeError)` typed failure
  surface; the CLI wrapper at
  `benchmarks/run.py:_run_bootstrap_pairwise_rollup` catches
  and drops a sentinel file independently from the B5 and B7
  sentinels.
- Reuses `RunManifest.fingerprint()` for freshness; every
  emitted row carries the fingerprint at aggregation time.
- Reuses the B13 OOM ceiling
  (`BOOTSTRAP_ROW_COUNT_CEILING`) and profile-default
  `n_resamples` dispatch via `_bootstrap_aggregate`.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

from benchmarks.bootstrap_manifest import PairwiseRollupRow, write_pairwise_rollup
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments.ensemble import load_pairwise
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.metrics.bootstrap import (
    BOOTSTRAP_RNG_ALGORITHM,
    entity_block_bootstrap_ci,
)
from benchmarks.report._bootstrap_aggregate import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_DEFAULT_SEED,
    BOOTSTRAP_N_RESAMPLES_BY_PROFILE,
    BOOTSTRAP_ROW_COUNT_CEILING,
    numpy_version,
)
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import RunManifest

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_bootstrap_pairwise_rollup",
    "is_pairwise_rollup_enabled",
    "pairwise_rollup_output_path",
]

_PRIMARY_METRIC = "complementarity_score"
_REGRESSION_SENTINEL_REASON = "regression_complementarity_undefined"
_CLASSIFICATION_TASK_TYPES: frozenset[str] = frozenset({"binary", "multiclass"})


def is_pairwise_rollup_enabled(config: BenchmarkConfig) -> bool:
    """True iff any ExperimentSpec(kind="ensemble") has
    bootstrap_pairwise_enabled=True. Public so the CLI wrapper
    can gate the dispatch."""
    return any(
        spec.kind == "ensemble" and spec.bootstrap_pairwise_enabled
        for spec in config.experiments
    )


def pairwise_rollup_output_path() -> str:
    """Stable filename token for log messages."""
    return "bootstrap_pairwise_rollup.parquet"


def _resolve_n_resamples(experiments: list[ExperimentSpec], profile: str) -> int:
    """Priority: per-ensemble-spec override > profile default."""
    for spec in experiments:
        if spec.kind == "ensemble" and spec.bootstrap_n_resamples is not None:
            return spec.bootstrap_n_resamples
    return BOOTSTRAP_N_RESAMPLES_BY_PROFILE.get(profile, 10_000)


def _emit_sentinel_row(
    *,
    dataset_name: str,
    model_a: str,
    model_b: str,
    task_type: str,
    n_seeds: int,
    n_cells_evaluated: int,
    n_skipped_cells: int,
    bootstrap_skipped_reason: str,
    n_resamples: int,
    manifest_fingerprint: str,
) -> PairwiseRollupRow:
    return PairwiseRollupRow(
        dataset_name=dataset_name,
        model_a=model_a,
        model_b=model_b,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells_evaluated,
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


def _build_group_rollup(
    block: pd.DataFrame,
    *,
    dataset_name: str,
    model_a: str,
    model_b: str,
    task_type: str,
    n_resamples: int,
    manifest_fingerprint: str,
) -> PairwiseRollupRow:
    """One pairwise rollup row for a (dataset, A, B, task) group."""
    n_seeds = int(block["seed"].nunique())
    ok = block.loc[block["skipped_reason"].isna()]
    n_skipped = int(block.shape[0] - ok.shape[0])

    # B14.0 regression-cell sentinel: complementarity_score is
    # defined only for classification; regression groups emit a
    # sentinel row that the renderer's "Bootstrap skipped"
    # footnote surfaces verbatim.
    if task_type not in _CLASSIFICATION_TASK_TYPES:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_a=model_a,
            model_b=model_b,
            task_type=task_type,
            n_seeds=n_seeds,
            n_cells_evaluated=0,
            n_skipped_cells=n_skipped,
            bootstrap_skipped_reason=_REGRESSION_SENTINEL_REASON,
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )

    if ok.empty:
        return _emit_sentinel_row(
            dataset_name=dataset_name,
            model_a=model_a,
            model_b=model_b,
            task_type=task_type,
            n_seeds=n_seeds,
            n_cells_evaluated=0,
            n_skipped_cells=n_skipped,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )

    # Per-cell complementarity_score is `(1 - pearson_error_corr)
    # + disagreement_rate` per B6 (proxy ranking key). The
    # pairwise manifest already stores both inputs; compute the
    # score per cell here so the rollup's mean matches the B6
    # renderer's `complementarity_score`.
    pearson_err = ok["pearson_error_corr"].astype(float)
    disagreement = ok["disagreement_rate"].astype(float)
    score = (1.0 - pearson_err) + disagreement
    score_values = score.to_numpy()
    if np.any(pd.isna(score_values)):
        # An OK row with a NaN complementarity input is
        # malformed; raise so a future schema drift surfaces
        # instead of silently truncating the bootstrap.
        raise RawRollupError(
            f"aggregate_bootstrap_pairwise_rollup: dataset={dataset_name!r} "
            f"model_a={model_a!r} model_b={model_b!r} has an OK cell with "
            "NaN complementarity_score inputs (pearson_error_corr or "
            "disagreement_rate)"
        )

    n_cells = score_values.shape[0]

    # OOM gate (reuses B13's ceiling). For pairwise the row count
    # equals the cell count; this gate is conservative.
    if n_cells * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING:
        raise RawRollupError(
            f"aggregate_bootstrap_pairwise_rollup: dataset={dataset_name!r} "
            f"with n_cells={n_cells} * n_resamples={n_resamples} exceeds the "
            f"bootstrap-row-count ceiling ({BOOTSTRAP_ROW_COUNT_CEILING})"
        )

    # B14.0 cell-as-entity contract: each (seed, fold) cell is
    # its own entity. The primitive degenerates to row bootstrap
    # over cells but the call site stays uniform with B5.
    entity_ids = np.arange(n_cells, dtype=np.int64)
    mean, ci_lo, ci_hi = entity_block_bootstrap_ci(
        score_values,
        entity_ids,
        n_resamples=n_resamples,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_DEFAULT_SEED,
    )

    return PairwiseRollupRow(
        dataset_name=dataset_name,
        model_a=model_a,
        model_b=model_b,
        task_type=task_type,
        primary_metric=_PRIMARY_METRIC,
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells,
        n_skipped_cells=n_skipped,
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


def aggregate_bootstrap_pairwise_rollup(
    config: BenchmarkConfig,
    *,
    output_root: "object",
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[PairwiseRollupRow]:
    """Aggregate the B6 pairwise manifest into a list of CI rollup rows.

    Returns `[]` when the pairwise manifest is absent or empty
    (the CLI wrapper treats this as "nothing to roll up" rather
    than a failure).

    Args:
        config: the live BenchmarkConfig; used to resolve the
            per-ensemble-spec n_resamples override.
        output_root: the run output root (Path-like); passed
            verbatim to `load_pairwise` and `write_pairwise_rollup`.
        env: the RunEnvironment for the live run; the profile
            determines the default n_resamples.
        manifest: the loaded RunManifest; the fingerprint is
            captured on every emitted row.

    Raises:
        RawRollupError: a malformed OK cell or an OOM-gated row
            count. The CLI wrapper catches this and drops the
            failure sentinel.
    """
    pairwise_df = load_pairwise(output_root)  # type: ignore[arg-type]
    if pairwise_df.empty:
        return []

    profile = env.profile if hasattr(env, "profile") else "standard"
    n_resamples = _resolve_n_resamples(list(config.experiments), str(profile))
    manifest_fingerprint = manifest.fingerprint()

    rows: list[PairwiseRollupRow] = []
    grouped = pairwise_df.groupby(
        ["dataset_name", "model_a", "model_b", "task_type"], sort=True
    )
    for group_key, block in grouped:
        dataset_name, model_a, model_b, task_type = cast(
            tuple[str, str, str, str], group_key
        )
        rows.append(
            _build_group_rollup(
                block,
                dataset_name=dataset_name,
                model_a=model_a,
                model_b=model_b,
                task_type=task_type,
                n_resamples=n_resamples,
                manifest_fingerprint=manifest_fingerprint,
            )
        )

    write_pairwise_rollup(output_root, rows)  # type: ignore[arg-type]
    logger.info(
        "aggregate_bootstrap_pairwise_rollup: wrote %d rollup rows to %s",
        len(rows),
        pairwise_rollup_output_path(),
    )
    return rows
