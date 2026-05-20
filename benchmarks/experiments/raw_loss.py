"""Phase B5 raw-loss experiment driver (the B6.1 deliverable).

Iterates the (dataset, model, seed, fold) Cartesian product implied
by a `BenchmarkConfig`, applies the B3 fair-comparison protocol,
runs B4 metric computation, and persists each cell via the
B7.2 atomic shard-then-sentinel manifest. Skipped cells (task-type
mismatch, regression-quantile B5-followup, adapter crash, missing
optional dep) write a typed `skipped_reason` and the metric
columns remain `None`.

The driver is the SINGLE consumer of the (B1, B2, B3, B4) surfaces
the prior phases shipped:

- `benchmarks.registry`: dataset/model lookup + adapter factories.
- `benchmarks.protocol`: lookback resolution, splitter, fingerprint.
- `benchmarks.metrics.compute_all`: typed `MetricsRecord` per cell.
- `benchmarks.metrics.resource.measure_fit` + `measure_inference_latency`.

Resumability (B7.2): the driver checks `is_cell_complete` BEFORE
calling the adapter; the sentinel marker is the single source of
truth for "this cell is done", so a crashed run resumed against
the same results dir picks up where it left off.

B5-followup deferrals (see `docs/benchmark_suite_implementation_plan.md`):

- `regression_quantile` task type: the seq-sklearn regressor adapter
  exposes `predict_quantiles` but does NOT expose the configured
  `quantile_levels` tuple, so the pinball metric labels cannot be
  built from the protocol surface. B5-followup extends the protocol
  with `quantile_levels: tuple[float, ...] | None` and lands the
  pinball reporting; current cells emit
  `skipped_reason="regression_quantile_b5_followup"`.
- Torch CUDA hooks in `measure_fit`: B6 wires
  `torch.cuda.reset_peak_memory_stats` and
  `torch.cuda.max_memory_allocated` once the harness has a CUDA
  detection layer; B5 always passes `cuda_available=False`.
- `library_git_sha` / `benchmarks_git_sha`: best-effort via
  `git rev-parse HEAD` against the current repo; falls back to
  `"unknown"` when the run is outside a git checkout.
- Out-of-fold prediction persistence (B6.2.1): B5 records metrics
  only. B6 (ensemble) adds the predictions-side parquet sibling.
"""

import logging
import subprocess
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import BenchmarkConfig, DatasetSpec, ExperimentSpec, TaskType
from benchmarks.datasets._base import PanelDataset
from benchmarks.manifest import (
    ResultRow,
    cell_key,
    is_cell_complete,
    metrics_to_row_fields,
    write_cell,
)
from benchmarks.metrics import compute_all
from benchmarks.metrics.resource import (
    FitResource,
    InferenceLatency,
    measure_fit,
    measure_inference_latency,
)
from benchmarks.protocol import (
    fingerprint_folds,
    make_splitter,
    resolve_lookback,
)
from benchmarks.registry import (
    get_dataset,
    get_loader,
    get_model,
    instantiate_adapter,
    list_datasets,
    list_models,
)

logger = logging.getLogger(__name__)


_ALL_SENTINEL = "all"
_DEFAULT_N_SPLITS = 5
_DEFAULT_LATENCY_REPS = 20
_DEFAULT_LATENCY_WARMUP = 3
_RAW_LOSS_VARIANT = "default"
_HPO_TIER_RAW_LOSS = "none"


class RunEnvironment(BaseModel):
    """Per-run environment metadata recorded on every result row.

    Built once at the top of `run_raw_loss` and threaded into every
    `ResultRow` so the B8.1 reproducibility contract (library +
    benchmarks SHA, run_id, profile, hardware tier) is captured
    uniformly across cells.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    library_git_sha: str
    benchmarks_git_sha: str
    hardware_tier: str  # B5 reports "cpu" or "gpu_single"; B7 tightens
    profile: str  # "smoke" | "standard" | "full"


class RawLossExperimentResult(BaseModel):
    """Summary returned by `run_raw_loss`.

    Counts the cells the driver touched in this invocation
    (excluding cells already complete from a prior run). The full
    leaderboard lives in the manifest under `results_dir`; this
    summary is the CLI's exit-code source and the test-suite
    assertion target.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    cells_attempted: int  # adapter actually fit
    cells_skipped_task_mismatch: int  # dataset.task_type not in model.task_types
    cells_skipped_quantile_followup: int  # regression_quantile pending B5-followup
    cells_skipped_adapter_error: int  # adapter raised at fit / predict / proba
    cells_already_complete: int  # sentinel present at scan time


def _utc_now() -> str:
    """Current UTC time as a stable ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_git_sha(start: Path) -> str:
    """Best-effort `git rev-parse HEAD` for the repo containing `start`.

    Returns the literal string `"unknown"` when the call fails (no
    git binary, detached HEAD with no commits, not a checkout, etc.).
    The harness records the result verbatim so a B8.1 reader can see
    "unknown" and treat the row as non-reproducible.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_run_environment(
    *,
    profile: str,
    hardware_tier: str = "cpu",
    library_repo_root: Path | None = None,
    benchmarks_repo_root: Path | None = None,
) -> RunEnvironment:
    """Capture the per-run environment metadata.

    `library_repo_root` and `benchmarks_repo_root` default to the
    library + benchmarks repos discovered relative to this module;
    the caller can override (e.g. tests) to pin a fixed SHA.
    """
    if library_repo_root is None:
        library_repo_root = Path(__file__).resolve().parents[2]
    if benchmarks_repo_root is None:
        benchmarks_repo_root = Path(__file__).resolve().parents[2]
    return RunEnvironment(
        run_id=str(uuid.uuid4()),
        library_git_sha=_resolve_git_sha(library_repo_root),
        benchmarks_git_sha=_resolve_git_sha(benchmarks_repo_root),
        hardware_tier=hardware_tier,
        profile=profile,
    )


def _resolve_names(names: Sequence[str], all_names: Sequence[str]) -> tuple[str, ...]:
    """Expand the `"all"` sentinel to every registered name.

    A config that pins `datasets = ["all"]` resolves at run-time to
    the current registry contents; this lets a registry-extension
    branch run without a config rewrite.
    """
    if len(names) == 1 and names[0] == _ALL_SENTINEL:
        return tuple(all_names)
    return tuple(names)


def _resolve_seeds(experiments: Sequence[ExperimentSpec]) -> tuple[int, ...]:
    """Pull the deduplicated, sorted seed tuple from the `raw_loss`
    experiment spec(s).

    A `BenchmarkConfig` MAY declare multiple `ExperimentSpec`
    entries for the same kind (e.g. `raw_loss` + `ensemble` reusing
    the same seeds). The driver concatenates every `raw_loss` spec's
    seeds and dedupes; sorted order makes the iteration deterministic.

    Raises:
        ValueError: no `raw_loss` experiment is configured.
    """
    seeds: set[int] = set()
    for spec in experiments:
        if spec.kind == "raw_loss":
            seeds.update(spec.seeds)
    if not seeds:
        raise ValueError(
            "run_raw_loss: BenchmarkConfig declares no raw_loss "
            "experiment; pass --experiment=raw_loss only with a "
            "config carrying ExperimentSpec(kind='raw_loss')"
        )
    return tuple(sorted(seeds))


def _take_panel_rows(
    panel: pd.DataFrame, y: np.ndarray, indices: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Slice a panel + target by integer indices, preserving the
    panel's column order."""
    return panel.iloc[indices].reset_index(drop=True), y[indices]


def _maybe_predict_proba(
    adapter: SeqSklearnAdapter, panel: pd.DataFrame
) -> np.ndarray | None:
    """Call `predict_proba` when supported; return None otherwise.

    Two non-supported paths: `supports_proba=False` (regressor
    adapters, KNN-DTW, etc.) and the typed `ProbaUnsupportedError`
    raised by classifier adapters whose underlying estimator does
    not expose calibrated probabilities. Both collapse to None so
    the metric layer sees an explicit absence.
    """
    if not adapter.supports_proba:
        return None
    try:
        return adapter.predict_proba(panel)
    except ProbaUnsupportedError:
        return None


def _maybe_predict_quantiles(
    adapter: SeqSklearnAdapter,
    task_type: TaskType,
    panel: pd.DataFrame,
) -> np.ndarray | None:
    """Call `predict_quantiles` only when the cell's task type asks
    for it AND the adapter supports it.

    A classifier adapter raises `QuantilesUnsupportedError` here;
    the regressor adapters return an `(N, n_quantiles)` array. The
    pinball labels are NOT computable from the adapter surface in
    B5 (see the module-docstring B5-followup); the caller checks
    `task_type == "regression_quantile"` separately and surfaces
    the skipped-reason there.
    """
    if task_type != "regression_quantile":
        return None
    try:
        return adapter.predict_quantiles(panel)
    except QuantilesUnsupportedError:
        return None


def _build_row(
    *,
    env: RunEnvironment,
    spec: DatasetSpec,
    model_name: str,
    seed: int,
    fold_index: int,
    n_folds: int,
    split_fingerprint: str,
    l_resolved: int,
    started_at_utc: str,
    skipped_reason: str | None,
    metrics_dump: dict[str, Any] | None,
    fit_resource: FitResource | None,
    latency: InferenceLatency | None,
) -> ResultRow:
    """Assemble a `ResultRow` from the per-cell pieces.

    `metrics_dump` is `None` on skipped cells; the metric columns
    default to None on `ResultRow`. `fit_resource` and `latency` are
    also `None` on skipped cells.
    """
    fields: dict[str, Any] = {
        "library_git_sha": env.library_git_sha,
        "benchmarks_git_sha": env.benchmarks_git_sha,
        "run_id": env.run_id,
        "started_at_utc": started_at_utc,
        "dataset_name": spec.name,
        "model_name": model_name,
        "seed": seed,
        "variant": _RAW_LOSS_VARIANT,
        "task_type": spec.task_type,
        "fold_index": fold_index,
        "n_folds": n_folds,
        "split_fingerprint": split_fingerprint,
        "l_resolved": l_resolved,
        "hardware_tier": env.hardware_tier,
        "profile": env.profile,
        "hpo_tier": _HPO_TIER_RAW_LOSS,
        "skipped_reason": skipped_reason,
    }
    if metrics_dump is not None:
        fields.update(metrics_to_row_fields(metrics_dump))
    if fit_resource is not None:
        fields["wall_seconds"] = fit_resource.wall_seconds
        fields["peak_rss_bytes"] = fit_resource.peak_rss_bytes
        fields["peak_cuda_bytes"] = fit_resource.peak_cuda_bytes
    if latency is not None:
        fields["median_predict_ms"] = latency.median_ms
        fields["p95_predict_ms"] = latency.p95_ms
        fields["n_predict_reps"] = latency.n_reps
    return ResultRow(**fields)


def _iter_folds(
    splitter: Any, panel: pd.DataFrame
) -> Iterator[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    """Yield `(fold_index, (train_idx, test_idx))` per fold.

    Wrapped so the driver and the fingerprint call share a single
    materialization of the fold list (the splitter is an iterator;
    consuming it twice would skip the second pass).
    """
    for i, (train_idx, test_idx) in enumerate(splitter.split(panel)):
        yield i, (np.asarray(train_idx), np.asarray(test_idx))


def _run_one_cell(
    *,
    env: RunEnvironment,
    panel_dataset: PanelDataset,
    spec: DatasetSpec,
    model_name: str,
    seed: int,
    fold_index: int,
    n_folds: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_fingerprint: str,
    l_resolved: int,
    labels: np.ndarray | None,
) -> ResultRow:
    """Evaluate ONE (dataset, model, seed, fold) cell.

    Returns the constructed `ResultRow`; the caller persists it via
    `write_cell`. Adapter crashes are caught and emitted as
    `skipped_reason` rows so a single broken model never aborts the
    run.
    """
    started_at_utc = _utc_now()

    # B5-followup: regression_quantile cells need the protocol to
    # expose `quantile_levels` before pinball labels can be built;
    # skip cleanly for now.
    if spec.task_type == "regression_quantile":
        return _build_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            skipped_reason="regression_quantile_b5_followup",
            metrics_dump=None,
            fit_resource=None,
            latency=None,
        )

    try:
        adapter = instantiate_adapter(model_name, spec=spec)
        train_panel, train_y = _take_panel_rows(
            panel_dataset.panel, panel_dataset.y, train_idx
        )
        test_panel, test_y = _take_panel_rows(
            panel_dataset.panel, panel_dataset.y, test_idx
        )

        # B5.3 wall-clock + RSS; CUDA hooks are a B5-followup wired by B6.
        fit_resource = measure_fit(
            lambda: _fit_with_seed(adapter, train_panel, train_y, seed),
            cuda_available=False,
        )
        y_pred = adapter.predict(test_panel)
        y_proba = _maybe_predict_proba(adapter, test_panel)
        # The latency closure runs `predict` once per rep; ensures
        # warm cache parity across models.
        latency = measure_inference_latency(
            lambda: adapter.predict(test_panel),
            n_reps=_DEFAULT_LATENCY_REPS,
            n_warmup=_DEFAULT_LATENCY_WARMUP,
        )

        metrics = compute_all(
            task_type=spec.task_type,
            y_true=test_y,
            y_pred=y_pred,
            y_proba=y_proba,
            pos_label=spec.positive_label,
            labels=labels,
        )
        return _build_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            skipped_reason=None,
            metrics_dump=metrics.model_dump(),
            fit_resource=fit_resource,
            latency=latency,
        )
    except Exception as exc:  # noqa: BLE001
        # Any uncaught adapter error becomes a skipped-reason row.
        # The bare-Exception is intentional: a single broken adapter
        # must not abort the harness, and the typed error categories
        # (Proba/QuantilesUnsupported) were already handled above.
        # The reason string preserves the exception class name + msg
        # so a leaderboard reader can triage without re-running.
        logger.warning(
            "run_raw_loss: cell %s/%s seed=%d fold=%d skipped: %s: %s",
            spec.name,
            model_name,
            seed,
            fold_index,
            type(exc).__name__,
            exc,
        )
        return _build_row(
            env=env,
            spec=spec,
            model_name=model_name,
            seed=seed,
            fold_index=fold_index,
            n_folds=n_folds,
            split_fingerprint=split_fingerprint,
            l_resolved=l_resolved,
            started_at_utc=started_at_utc,
            skipped_reason=f"adapter_error: {type(exc).__name__}: {exc}",
            metrics_dump=None,
            fit_resource=None,
            latency=None,
        )


def _fit_with_seed(
    adapter: SeqSklearnAdapter,
    train_panel: pd.DataFrame,
    train_y: np.ndarray,
    seed: int,
) -> None:
    """Fit `adapter` on `(train_panel, train_y)` with the cell's seed
    threaded into numpy + python random state.

    The library's own seeding lives inside the adapter's estimator
    (B8.2); for the GBM and sklearn-passthrough adapters that will
    land in B5-followup, this helper seeds the process-level state
    so a fit that internally calls `np.random.*` is reproducible.
    """
    # The library's deterministic-mode wiring lives inside the
    # estimator's fit; harness-side numpy/python seeding covers the
    # comparator adapters (GBM / sklearn) where the bench is the
    # only seeding boundary.
    np.random.seed(seed)
    adapter.fit(train_panel, train_y)


def _resolve_labels(
    panel_dataset: PanelDataset, task_type: TaskType
) -> np.ndarray | None:
    """Return the canonical class-label array for classification
    cells; `None` for regression.

    `compute_all` for classification needs `labels` so the per-fold
    metric calls don't drop a class absent from a particular test
    fold. We compute it once from the full panel target so every
    seed/fold sees the same label set (B8.1 reproducibility).
    """
    if task_type in {"binary", "multiclass"}:
        return np.unique(panel_dataset.y)
    return None


def run_raw_loss(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
) -> RawLossExperimentResult:
    """Execute the raw-loss experiment across the config's (dataset,
    model, seed, fold) cells.

    Cell ordering is `dataset_name` -> `model_name` -> `seed` ->
    `fold_index`, all sorted; deterministic so a partial-run dump is
    inspectable.

    Resumability: every cell's sentinel is checked BEFORE the
    adapter is instantiated. Already-complete cells are skipped at
    near-zero cost (one filesystem stat per cell).

    Raises:
        DatasetNotRegisteredError, ModelNotRegisteredError: a name
            in the config has no registry entry.
        ValueError: no `raw_loss` experiment is configured.
    """
    dataset_names = _resolve_names(config.datasets, list_datasets())
    model_names = _resolve_names(config.models, list_models())
    seeds = _resolve_seeds(config.experiments)
    logger.info(
        "run_raw_loss: env.run_id=%s datasets=%d models=%d seeds=%s",
        env.run_id,
        len(dataset_names),
        len(model_names),
        seeds,
    )
    cache_dir = config.cache_dir
    if cache_dir is None:
        raise ValueError(
            "run_raw_loss: BenchmarkConfig.cache_dir must be set; the "
            "loader callables consume it for the on-disk cache root"
        )

    attempted = 0
    skipped_task_mismatch = 0
    skipped_quantile_followup = 0
    skipped_adapter_error = 0
    already_complete = 0

    for dataset_name in sorted(dataset_names):
        spec = get_dataset(dataset_name)
        if spec.excluded:
            # B1.5 negatives: registered with `excluded=True` for
            # the rubric audit but never benchmarked.
            logger.info(
                "run_raw_loss: dataset %s skipped (excluded=True; reason=%s)",
                dataset_name,
                spec.exclusion_reason,
            )
            continue
        loader = get_loader(dataset_name)
        load_start = time.perf_counter()
        panel_dataset = loader(cache_dir)
        logger.info(
            "run_raw_loss: dataset %s loaded in %.2fs (%d rows)",
            dataset_name,
            time.perf_counter() - load_start,
            len(panel_dataset.panel),
        )

        l_resolved = resolve_lookback(spec)
        splitter = make_splitter(spec, lookback=l_resolved, n_splits=_DEFAULT_N_SPLITS)
        folds = list(_iter_folds(splitter, panel_dataset.panel))
        if not folds:
            logger.warning(
                "run_raw_loss: dataset %s produced no folds; check "
                "n_splits and per-entity row counts",
                dataset_name,
            )
            continue
        n_folds = len(folds)
        split_fingerprint = fingerprint_folds(idx for _, idx in folds)
        labels = _resolve_labels(panel_dataset, spec.task_type)

        for model_name in sorted(model_names):
            model_spec = get_model(model_name)
            applicable = spec.task_type in model_spec.task_types
            for seed in seeds:
                for fold_index, (train_idx, test_idx) in folds:
                    key_started = time.perf_counter()
                    # Resumability: sentinel-first check.
                    key = cell_key(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        variant=_RAW_LOSS_VARIANT,
                        fold_index=fold_index,
                    )
                    if is_cell_complete(output_root, key):
                        already_complete += 1
                        continue

                    if not applicable:
                        skipped_task_mismatch += 1
                        row = _build_row(
                            env=env,
                            spec=spec,
                            model_name=model_name,
                            seed=seed,
                            fold_index=fold_index,
                            n_folds=n_folds,
                            split_fingerprint=split_fingerprint,
                            l_resolved=l_resolved,
                            started_at_utc=_utc_now(),
                            skipped_reason=(
                                f"task_type_mismatch: model.task_types="
                                f"{model_spec.task_types} dataset.task_type="
                                f"{spec.task_type}"
                            ),
                            metrics_dump=None,
                            fit_resource=None,
                            latency=None,
                        )
                    else:
                        row = _run_one_cell(
                            env=env,
                            panel_dataset=panel_dataset,
                            spec=spec,
                            model_name=model_name,
                            seed=seed,
                            fold_index=fold_index,
                            n_folds=n_folds,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            split_fingerprint=split_fingerprint,
                            l_resolved=l_resolved,
                            labels=labels,
                        )
                        if row.skipped_reason is None:
                            attempted += 1
                        elif row.skipped_reason.startswith(
                            "regression_quantile_b5_followup"
                        ):
                            skipped_quantile_followup += 1
                        else:
                            skipped_adapter_error += 1

                    write_cell(output_root, row)
                    logger.info(
                        "run_raw_loss: cell %s done in %.2fs (reason=%s)",
                        key,
                        time.perf_counter() - key_started,
                        row.skipped_reason or "ok",
                    )

    summary = RawLossExperimentResult(
        run_id=env.run_id,
        cells_attempted=attempted,
        cells_skipped_task_mismatch=skipped_task_mismatch,
        cells_skipped_quantile_followup=skipped_quantile_followup,
        cells_skipped_adapter_error=skipped_adapter_error,
        cells_already_complete=already_complete,
    )
    logger.info("run_raw_loss: complete: %s", summary.model_dump())
    return summary


__all__ = [
    "RawLossExperimentResult",
    "RunEnvironment",
    "build_run_environment",
    "run_raw_loss",
]
