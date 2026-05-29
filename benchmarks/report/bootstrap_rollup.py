"""Phase B13 bootstrap-CI aggregator (B5.4).

Reads the B5 manifest + per-cell predictions shards, re-resolves
`entity_id` for each row via the registered dataset loader,
computes per-row losses, calls the entity-block bootstrap, and
emits a per-(dataset, model, task_type) `RollupRow`. The output
shard is written via `benchmarks.bootstrap_manifest.write_rollup`.

Design contracts (`docs/benchmark_suite_design_b13_delta.md`):

- B13.0: `RawRollupError(RuntimeError)` is the typed failure
  surface. Subclasses `RuntimeError` so a future driver call
  routes via the existing narrow catch tuple; the v1 CLI call
  site wraps `_run_bootstrap_rollup` with an explicit
  `try/except RawRollupError` so the leaderboard falls back to
  the std variant rather than the run crashing.

- R7 (Gemini-C1): the aggregator validates loader determinism
  via a defensive post-load sort by `(entity_col, time_col)`
  AND a row-count drift check against the predictions shard's
  `max(panel_row_index) + 1`. Either mismatch raises
  `RawRollupError`.

- R-B13-3 (Gemini-C2): the aggregator enforces a row-count
  ceiling `N * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING (5e10)`
  via `RawRollupError` so a full-tier dataset doesn't OOM the
  naive bootstrap path. The sufficient-statistics optimization
  is D-B13.7.

- R8 + B13.4 (Gemini-C3): every emitted `RollupRow` carries the
  `manifest_fingerprint` captured at aggregation time. The
  renderer freshness check rejects a stale rollup.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.bootstrap_manifest import RollupRow, write_rollup
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.manifest import load_run
from benchmarks.metrics.bootstrap import (
    BOOTSTRAP_RNG_ALGORITHM,
    entity_block_bootstrap_ci,
)
from benchmarks.metrics.pairwise import classification_nll
from benchmarks.predictions import (
    has_predictions,
    load_predictions,
    proba_matrix,
)
from benchmarks.registry import get_dataset, get_loader
from benchmarks.report._bootstrap_aggregate import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_DEFAULT_SEED,
    BOOTSTRAP_ROW_COUNT_CEILING,
    numpy_version,
    resolve_n_resamples,
)
from benchmarks.run_manifest import RunManifest

logger = logging.getLogger(__name__)

__all__ = [
    "RawRollupError",
    "aggregate_bootstrap_rollup",
]


class RawRollupError(RuntimeError):
    """Typed bootstrap-rollup aggregator failure.

    Subclasses `RuntimeError` so the B5/B8 drivers' narrow catch
    tuple (`(RuntimeError, MemoryError, NotFittedError,
    ProbaUnsupportedError, QuantilesUnsupportedError)`) would
    route a driver-called aggregator failure as an
    `adapter_error` skip. The v1 entrypoint runs at report
    time; `benchmarks/run.py:_run_bootstrap_rollup` wraps the
    aggregator call with `try/except RawRollupError` so the
    leaderboard falls back to the std variant + emits a
    "Bootstrap aggregator failed" footnote.
    """


# B13 task-type primary-loss mapping. (The bootstrap constants
# moved to `benchmarks/report/_bootstrap_aggregate.py` in B14 so
# the new B6 and B7 rollup aggregators share them; the symbols
# imported above keep this module's call sites unchanged.)
_PRIMARY_LOSS_BY_TASK: dict[str, str] = {
    "binary": "log_loss",
    "multiclass": "log_loss",
    "regression_point": "rmse",
}


def _is_rollup_enabled(experiments: list[ExperimentSpec]) -> bool:
    """The rollup runs only when a raw_loss spec has
    `bootstrap_rollup_enabled=True`. Disabled if every raw_loss
    spec sets it to False; if there is no raw_loss spec at all,
    there's nothing to roll up."""
    return any(spec.kind == "raw_loss" and spec.bootstrap_rollup_enabled for spec in experiments)


def _load_panel_or_raise(dataset_name: str, cache_dir: Path | None) -> pd.DataFrame:
    """Re-load the dataset panel via the registered loader.

    Loader contract (R7): every loader produces row-identical
    output across calls. The aggregator's re-load uses the same
    cache_dir as the B5 driver did, so a cache-hit produces the
    same panel.

    Raises:
        RawRollupError: the loader raised.
    """
    try:
        loader = get_loader(dataset_name)
        # cache_dir is the registered loader's cache root; falls
        # back to a `Path(".")` sentinel when None so the loader
        # contract is satisfied. Real loaders honor None.
        panel_dataset = loader(cache_dir if cache_dir is not None else Path())
    except Exception as exc:
        raise RawRollupError(
            f"aggregate_bootstrap_rollup: loader for dataset {dataset_name!r} "
            f"failed: {type(exc).__name__}: {exc}"
        ) from exc
    return panel_dataset.panel


def _resolve_entity_ids(
    panel: pd.DataFrame,
    spec_entity_col: str,
    spec_time_col: str,
    panel_row_index: np.ndarray,
) -> np.ndarray:
    """Re-resolve entity_ids for the predictions shard's rows.

    Per R7 defensive sort: re-sort the panel by
    `(entity_col, time_col)` BEFORE indexing by `panel_row_index`
    so a non-deterministic loader (loader-determinism contract
    violation) still produces deterministic results.

    Raises:
        RawRollupError: row-count drift (panel size doesn't
            accommodate the predictions shard's max
            panel_row_index).
    """
    # NB: the existing B5 driver writes panel_row_index BEFORE any
    # post-load sort, so the indices are valid against the panel
    # as the loader produced it. R7's defensive sort produces an
    # equivalent panel because every loader is required to ship
    # rows in (entity, time) order by contract. The sort is a
    # safety net, not a transform: if it changes the row layout
    # of a real loader, that loader is non-conforming.
    sorted_panel = panel.sort_values(
        by=[spec_entity_col, spec_time_col], kind="stable"
    ).reset_index(drop=True)
    max_idx = int(panel_row_index.max()) if panel_row_index.size > 0 else -1
    if max_idx >= len(sorted_panel):
        raise RawRollupError(
            f"aggregate_bootstrap_rollup: loader row-count drift: predictions "
            f"shard references panel_row_index up to {max_idx} but loader "
            f"returned only {len(sorted_panel)} rows"
        )
    return sorted_panel.iloc[panel_row_index][spec_entity_col].to_numpy()


def _per_row_losses(
    task_type: str,
    predictions: pd.DataFrame,
) -> tuple[np.ndarray, Callable[[np.ndarray], float]] | None:
    """Compute per-row losses + the matching bootstrap `metric_fn`.

    Returns None when the predictions shard cannot produce a
    per-row loss vector (e.g., classification cell with no proba
    column).

    Classification: per-row NLL via
    `benchmarks.metrics.pairwise.classification_nll` (clip eps =
    1e-15). `metric_fn = np.nanmean`.

    Regression: per-row squared error. `metric_fn =
    sqrt(np.nanmean(.))` so sqrt applies INSIDE each resample
    (closes the Jensen-bias gap).
    """
    y_true = predictions["y_true"].to_numpy()
    if task_type in {"binary", "multiclass"}:
        proba = proba_matrix(predictions)
        if proba is None:
            return None
        labels = np.arange(proba.shape[1])
        per_row = classification_nll(y_true, proba, labels)

        def _metric_classification(x: np.ndarray) -> float:
            return float(np.nanmean(x))

        return per_row, _metric_classification

    if task_type == "regression_point":
        y_pred = predictions["y_pred"].to_numpy()
        per_row = (y_true - y_pred) ** 2

        def _metric_regression(x: np.ndarray) -> float:
            return float(np.sqrt(np.nanmean(x)))

        return per_row, _metric_regression

    return None


def _cell_key_for_row(row: pd.Series) -> str:
    from benchmarks.manifest import cell_key

    return cell_key(
        dataset_name=str(row["dataset_name"]),
        model_name=str(row["model_name"]),
        seed=int(row["seed"]),
        variant=str(row["variant"]),
        fold_index=int(row["fold_index"]),
    )


def _build_group_rollup(
    *,
    dataset_name: str,
    model_name: str,
    task_type: str,
    group_rows: pd.DataFrame,
    output_root: Path,
    cache_dir: Path | None,
    n_resamples: int,
    manifest_fingerprint: str,
) -> RollupRow:
    """Aggregate one (dataset, model, task_type) group to one RollupRow."""
    ok_mask = group_rows["skipped_reason"].isna()
    ok_cells = group_rows.loc[ok_mask]
    n_skipped = int((~ok_mask).sum())
    n_seeds = int(group_rows["seed"].nunique())
    primary_metric = _PRIMARY_LOSS_BY_TASK.get(task_type, "unknown")

    if ok_cells.empty:
        # All-skipped sentinel.
        return RollupRow(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_metric=primary_metric,
            n_seeds=n_seeds,
            n_cells_evaluated=0,
            n_skipped_cells=n_skipped,
            n_rows=0,
            n_entities=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
            bootstrap_n_resamples=n_resamples,
            bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_numpy_version=numpy_version(),
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
            manifest_fingerprint=manifest_fingerprint,
        )

    panel = _load_panel_or_raise(dataset_name, cache_dir)
    spec = get_dataset(dataset_name)
    losses_blocks: list[np.ndarray] = []
    entity_blocks: list[np.ndarray] = []
    metric_fn: Callable[[np.ndarray], float] | None = None

    cells_kept = 0
    for _, row in ok_cells.iterrows():
        key = _cell_key_for_row(row)
        if not has_predictions(output_root, key):
            logger.warning(
                "aggregate_bootstrap_rollup: predictions shard missing for %s; "
                "dropping from rollup",
                key,
            )
            continue
        predictions = load_predictions(output_root, key)
        result = _per_row_losses(task_type, predictions)
        if result is None:
            logger.warning(
                "aggregate_bootstrap_rollup: cannot compute per-row losses for %s "
                "(task=%s, missing proba?); dropping",
                key,
                task_type,
            )
            continue
        per_row_losses, cell_metric_fn = result
        if metric_fn is None:
            metric_fn = cell_metric_fn
        panel_row_index = predictions["panel_row_index"].to_numpy()
        entity_ids = _resolve_entity_ids(panel, spec.entity_col, spec.time_col, panel_row_index)
        losses_blocks.append(per_row_losses)
        entity_blocks.append(entity_ids)
        cells_kept += 1

    if cells_kept == 0 or metric_fn is None:
        # All OK cells failed the predictions-shard / loss path;
        # surface as a sentinel.
        return RollupRow(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            primary_metric=primary_metric,
            n_seeds=n_seeds,
            n_cells_evaluated=0,
            n_skipped_cells=n_skipped,
            n_rows=0,
            n_entities=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
            bootstrap_n_resamples=n_resamples,
            bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_numpy_version=numpy_version(),
            bootstrap_skipped_reason="all_predictions_shards_missing_or_unloadable",
            manifest_fingerprint=manifest_fingerprint,
        )

    losses = np.concatenate(losses_blocks)
    entities = np.concatenate(entity_blocks)

    # OOM gate per R-B13-3 / Gemini-C2.
    if losses.shape[0] * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING:
        raise RawRollupError(
            f"aggregate_bootstrap_rollup: dataset {dataset_name!r} with "
            f"N={losses.shape[0]} rows * n_resamples={n_resamples} exceeds "
            f"the bootstrap-row-count ceiling ({BOOTSTRAP_ROW_COUNT_CEILING}); "
            "use the per-entity-sufficient-statistics path documented in "
            "D-B13.7 (B13-followup)"
        )

    mean, ci_lo, ci_hi = entity_block_bootstrap_ci(
        losses,
        entities,
        n_resamples=n_resamples,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_DEFAULT_SEED,
        metric_fn=metric_fn,
    )

    n_unique_entities = int(np.unique(entities).shape[0])
    return RollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        primary_metric=primary_metric,
        n_seeds=n_seeds,
        n_cells_evaluated=cells_kept,
        n_skipped_cells=n_skipped,
        n_rows=int(losses.shape[0]),
        n_entities=n_unique_entities,
        primary_metric_mean=mean,
        primary_metric_ci_lo=ci_lo,
        primary_metric_ci_hi=ci_hi,
        bootstrap_seed=BOOTSTRAP_DEFAULT_SEED,
        bootstrap_n_resamples=n_resamples,
        bootstrap_rng_algorithm=BOOTSTRAP_RNG_ALGORITHM,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_numpy_version=numpy_version(),
        bootstrap_skipped_reason=None,
        manifest_fingerprint=manifest_fingerprint,
    )


def aggregate_bootstrap_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[RollupRow]:
    """Aggregate per-(dataset, model, task_type) bootstrap CI rollup
    rows from the B5 manifest at `output_root`.

    Raises:
        ValueError: the B5 manifest at `output_root` is empty.
        RawRollupError: a typed aggregator failure (loader error,
            row-count drift, row-count ceiling).
    """
    manifest_df = load_run(output_root)
    if manifest_df.empty:
        raise ValueError(
            f"aggregate_bootstrap_rollup: B5 manifest at {output_root} is "
            f"empty; run --experiment=raw_loss first"
        )

    n_resamples = resolve_n_resamples(
        config.experiments, env.profile, kind="raw_loss"
    )
    cache_dir = config.cache_dir
    manifest_fingerprint = manifest.fingerprint()
    out: list[RollupRow] = []
    grouped = manifest_df.groupby(["dataset_name", "model_name", "task_type"], sort=True)
    for group_key, group_rows in grouped:
        # pyright's pandas stub types groupby keys as Hashable; the
        # 3-col groupby actually emits a 3-tuple of strings.
        from typing import cast as _cast

        key_tuple = _cast(tuple[str, str, str], group_key)
        dataset_name, model_name, task_type = (
            str(key_tuple[0]),
            str(key_tuple[1]),
            str(key_tuple[2]),
        )
        if task_type not in _PRIMARY_LOSS_BY_TASK:
            # Regression_quantile cells inherit the existing B5
            # skip; not part of the v1 rollup surface.
            continue
        rollup = _build_group_rollup(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            group_rows=group_rows,
            output_root=output_root,
            cache_dir=cache_dir,
            n_resamples=n_resamples,
            manifest_fingerprint=manifest_fingerprint,
        )
        out.append(rollup)
    write_rollup(output_root, out)
    return out


def is_rollup_enabled(config: BenchmarkConfig) -> bool:
    """Whether `config` opts into the B13 rollup step.

    True iff at least one `ExperimentSpec(kind="raw_loss")` has
    `bootstrap_rollup_enabled=True` (the default).
    """
    return _is_rollup_enabled(list(config.experiments))
