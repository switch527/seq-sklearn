"""Phase B6 ensemble-complementarity experiment driver (B6.2).

For each (dataset, model_a, model_b) pair represented in a B5
raw-loss manifest, the driver loads both models' per-fold prediction
shards, inner-joins on `panel_row_index` (different per-model
below-floor strips so row counts can differ), and computes the
B6.2.2/.3/.4 pairwise statistics.

The driver is the SINGLE consumer of `benchmarks.predictions` plus
`benchmarks.metrics.pairwise`. Output is a new shard layout under
the run's `output_root`:

    output_root/
      results/                # B5 raw-loss metric shards
        {cell_key}.parquet
      predictions/            # B5 raw-loss prediction shards
        {cell_key}.parquet
      pairwise/               # B6.2 pairwise statistics
        {pair_key}.parquet    # one row per (dataset, A, B, seed, fold)
      pairwise/_done/         # B7.2-style sentinel markers
        {pair_key}.json

Pair key shape: `{dataset}__{model_a}__vs__{model_b}__seed_{seed}
__fold_{fold}`. Ordering: the pair (A, B) is always written with
A < B lexicographically so the (A, B) and (B, A) pairs collapse to
the same shard (the diversity statistics are symmetric).

B6-followup deferrals (documented in the impl plan):

- B6.2.5 (complementarity ensemble): build a GBM-only ensemble and
  a GBM+seq-sklearn ensemble on the same folds and report the delta
  loss. Requires nested-stacking + an external GBM adapter, which
  ships in a B6-followup branch alongside the registered comparator
  models.
- Regression-quantile pairs: B5 already defers quantile metrics
  pending the protocol's `quantile_levels` extension; the pairwise
  driver inherits the same deferral.
- Aggregation: the pairwise shards land per-fold; the per-dataset
  rollup (mean ± std across seeds + folds) lives in
  `benchmarks/report/ensemble.py`.
"""

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from benchmarks.config import BenchmarkConfig, ExperimentSpec, TaskType
from benchmarks.experiments.raw_loss import RunEnvironment
from benchmarks.manifest import load_run
from benchmarks.metrics.pairwise import (
    classification_nll,
    disagreement_rate,
    double_fault_rate,
    joint_counts,
    pearson_correlation,
    phi_coefficient,
    regression_signed_residual,
    spearman_correlation,
    yule_q,
)
from benchmarks.predictions import (
    has_predictions,
    load_predictions,
    proba_matrix,
)

logger = logging.getLogger(__name__)


_PAIRWISE_DIRNAME = "pairwise"
_SENTINEL_DIRNAME = "_done"
_SHARD_SUFFIX = ".parquet"
_SENTINEL_SUFFIX = ".json"

# Pair-key token rules (mirror `benchmarks.manifest._CELL_KEY_TOKEN_RE`).
_PAIR_KEY_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]*$")

# Skip reason prefixes for pairwise cells. Two B6-design-named
# reasons (task-type mismatch between models; regression_quantile
# followup) are structurally unreachable in B6: every cell within
# a dataset shares the same task_type by B5 invariant, and
# `_eligible_pair_cells` filters quantile cells before the loop.
# B6 ships only the reachable reasons; if a B6-followup that adds
# cross-dataset pairs or relaxes the quantile filter lands, the
# corresponding constant + counter return alongside.
_SKIP_REASON_PREDICTIONS_MISSING = "predictions_missing"
_SKIP_REASON_EMPTY_JOIN = "empty_panel_row_index_join"

_CLASSIFICATION_TASK_TYPES: frozenset[TaskType] = frozenset({"binary", "multiclass"})
_REGRESSION_POINT_TASK_TYPE: TaskType = "regression_point"


class PairwiseKeyError(ValueError):
    """Raised when a pair-key component contains characters that
    would conflict with the `__` / `__vs__` separator scheme."""


def pair_key(
    *,
    dataset_name: str,
    model_a: str,
    model_b: str,
    seed: int,
    fold_index: int,
) -> str:
    """Build the canonical pair-key string for a (dataset, A, B,
    seed, fold) shard.

    The (A, B) ordering is canonicalized to lexicographic so the
    (A, B) and (B, A) shards collapse to the same path. The four
    tokens (dataset_name, model_a, model_b, seed/fold) are each
    validated against the same separator-safety regex used by
    `benchmarks.manifest.cell_key`.

    Raises:
        PairwiseKeyError: a token would collide with the separator,
            or `model_a == model_b`, or `fold_index < 0`.
    """
    if model_a == model_b:
        raise PairwiseKeyError(
            f"pair_key: model_a == model_b == {model_a!r}; pairwise "
            f"statistics require two distinct models"
        )
    lo, hi = sorted([model_a, model_b])
    for label, value in (
        ("dataset_name", dataset_name),
        ("model_a", lo),
        ("model_b", hi),
    ):
        if "__" in value:
            raise PairwiseKeyError(f"pair_key: {label}={value!r} contains the separator '__'")
        if not _PAIR_KEY_TOKEN_RE.fullmatch(value):
            raise PairwiseKeyError(
                f"pair_key: {label}={value!r} must match {_PAIR_KEY_TOKEN_RE.pattern!r}"
            )
    if fold_index < 0:
        raise PairwiseKeyError(f"pair_key: fold_index must be >= 0; got {fold_index}")
    return f"{dataset_name}__{lo}__vs__{hi}__seed_{seed}__fold_{fold_index}"


def pairwise_dir(root: Path) -> Path:
    """`{root}/pairwise/`: the per-run pairwise-shard directory."""
    return root / _PAIRWISE_DIRNAME


def pairwise_sentinel_dir(root: Path) -> Path:
    """`{root}/pairwise/_done/`: the per-run sentinel marker directory."""
    return root / _PAIRWISE_DIRNAME / _SENTINEL_DIRNAME


def pairwise_shard_path(root: Path, key: str) -> Path:
    """Path of the pairwise shard for `key`."""
    return pairwise_dir(root) / f"{key}{_SHARD_SUFFIX}"


def pairwise_sentinel_path(root: Path, key: str) -> Path:
    """Path of the pairwise sentinel marker for `key`."""
    return pairwise_sentinel_dir(root) / f"{key}{_SENTINEL_SUFFIX}"


def is_pair_complete(root: Path, key: str) -> bool:
    """Return True iff the pairwise sentinel exists for `key`."""
    return pairwise_sentinel_path(root, key).exists()


def _atomic_write_bytes(payload: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(payload)
    tmp.replace(dest)


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


class PairwiseRow(BaseModel):
    """One pairwise statistic record per (dataset, A, B, seed, fold).

    The classification-only fields (`n11`, `n10`, `n01`, `n00`,
    `yule_q`, `phi`, `disagreement_rate`, `double_fault_rate`) are
    `None` on regression cells; the cross-task fields
    (`pearson_pred_corr`, `spearman_pred_corr`, `pearson_error_corr`)
    are populated on every successful pair regardless of task type.
    Skipped cells (`skipped_reason is not None`) carry None for
    every statistic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    library_git_sha: str
    run_id: str
    started_at_utc: str

    dataset_name: str
    model_a: str
    model_b: str
    seed: int
    fold_index: int = Field(ge=0)
    task_type: TaskType

    skipped_reason: str | None = None

    # Number of rows surviving the panel_row_index inner-join.
    n_samples: int | None = None

    # Classification 2x2 joint-correctness table + diversity stats.
    n11: int | None = None
    n10: int | None = None
    n01: int | None = None
    n00: int | None = None
    yule_q: float | None = None
    phi: float | None = None
    disagreement_rate: float | None = None
    double_fault_rate: float | None = None

    # Cross-task correlations.
    pearson_pred_corr: float | None = None
    spearman_pred_corr: float | None = None
    pearson_error_corr: float | None = None

    def pair_key(self) -> str:
        return pair_key(
            dataset_name=self.dataset_name,
            model_a=self.model_a,
            model_b=self.model_b,
            seed=self.seed,
            fold_index=self.fold_index,
        )


class EnsembleExperimentResult(BaseModel):
    """Summary returned by `run_ensemble`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    pairs_attempted: int  # pairwise stats actually computed
    pairs_skipped_predictions_missing: int  # one model's shard absent
    pairs_skipped_empty_join: int  # panel_row_index inner-join was empty
    pairs_already_complete: int  # sentinel present at scan time


def _optional_scalar_columns_for_pairwise() -> tuple[tuple[str, str], ...]:
    """Map of `PairwiseRow` optional scalar fields to their nullable
    pandas dtype. Same pattern as `benchmarks.manifest._optional_
    scalar_columns`: cast `float | None` to `float64` (NaN for None)
    and `int | None` to `Int64` (pandas nullable integer)."""
    out: list[tuple[str, str]] = []
    for name, info in PairwiseRow.model_fields.items():
        annotation = str(info.annotation)
        if annotation == "float | None":
            out.append((name, "float64"))
        elif annotation == "int | None":
            out.append((name, "Int64"))
    return tuple(out)


def _coerce_pairwise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    cols = _optional_scalar_columns_for_pairwise()
    if not cols:
        return df
    df = df.copy()
    for name, dtype in cols:
        if name in df.columns:
            df[name] = df[name].astype(dtype)
    return df


def write_pairwise_cell(root: Path, row: PairwiseRow) -> None:
    """Persist one pairwise cell atomically (shard then sentinel)."""
    key = row.pair_key()
    df = pd.DataFrame([row.model_dump()])
    df = _coerce_pairwise_dtypes(df)
    _atomic_write_parquet(df, pairwise_shard_path(root, key))
    # Fresh completion timestamp at sentinel-write time; the
    # row's `started_at_utc` is preserved separately so an
    # auditor can compute the (start, complete) duration.
    payload = json.dumps(
        {
            "pair_key": key,
            "completed_at_utc": _utc_now(),
            "started_at_utc": row.started_at_utc,
            "skipped_reason": row.skipped_reason,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write_bytes(payload, pairwise_sentinel_path(root, key))


def load_pairwise(root: Path) -> pd.DataFrame:
    """Read every pairwise shard under `root/pairwise/` and concatenate."""
    dir_ = pairwise_dir(root)
    if not dir_.exists():
        return _empty_pairwise_dataframe()
    shards: list[Path] = sorted(dir_.glob(f"*{_SHARD_SUFFIX}"))
    if not shards:
        return _empty_pairwise_dataframe()
    frames = [pd.read_parquet(p) for p in shards]
    return pd.concat(frames, ignore_index=True)


def _empty_pairwise_dataframe() -> pd.DataFrame:
    columns = pd.Index(list(PairwiseRow.model_fields.keys()))
    return pd.DataFrame(columns=columns)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_seeds(experiments: Iterable[ExperimentSpec]) -> tuple[int, ...]:
    """Pull seeds from the `ensemble` experiment spec(s).

    Raises:
        ValueError: no `ensemble` experiment is configured.
    """
    seeds: set[int] = set()
    for spec in experiments:
        if spec.kind == "ensemble":
            seeds.update(spec.seeds)
    if not seeds:
        raise ValueError(
            "run_ensemble: BenchmarkConfig declares no ensemble "
            "experiment; add ExperimentSpec(kind='ensemble') to the "
            "config or pass --experiment=ensemble only when the "
            "config carries one"
        )
    return tuple(sorted(seeds))


def _pairwise_for_cell(
    *,
    env: RunEnvironment,
    output_root: Path,
    dataset_name: str,
    model_a: str,
    model_b: str,
    task_type: TaskType,
    seed: int,
    fold_index: int,
    classes: np.ndarray | None,
) -> PairwiseRow:
    """Compute the pairwise statistics for one (A, B, seed, fold) pair.

    Loads both prediction shards, inner-joins on `panel_row_index`,
    and computes the B6.2.2/.3/.4 statistics. The classification-
    only diversity stats are computed when `task_type` is binary or
    multiclass; regression cells skip those four fields.

    `classes` is the canonical class label set for classification
    cells (mirrors B5's `_resolve_labels` derived from the FULL
    panel target). `None` for regression cells.
    """
    started_at_utc = _utc_now()
    base_row_fields: dict[str, Any] = {
        "library_git_sha": env.library_git_sha,
        "run_id": env.run_id,
        "started_at_utc": started_at_utc,
        "dataset_name": dataset_name,
        "model_a": min(model_a, model_b),
        "model_b": max(model_a, model_b),
        "seed": seed,
        "fold_index": fold_index,
        "task_type": task_type,
    }

    # Reconstruct the cell key for each model so we can load
    # predictions. Local import to avoid a top-of-file circular.
    from benchmarks.manifest import cell_key

    key_a = cell_key(
        dataset_name=dataset_name,
        model_name=model_a,
        seed=seed,
        variant="default",
        fold_index=fold_index,
    )
    key_b = cell_key(
        dataset_name=dataset_name,
        model_name=model_b,
        seed=seed,
        variant="default",
        fold_index=fold_index,
    )
    if not has_predictions(output_root, key_a) or not has_predictions(output_root, key_b):
        return PairwiseRow(
            **base_row_fields,
            skipped_reason=(
                f"{_SKIP_REASON_PREDICTIONS_MISSING}: "
                f"one or both of ({model_a!r}, {model_b!r}) has no "
                f"prediction shard for seed={seed} fold={fold_index}"
            ),
        )

    df_a = load_predictions(output_root, key_a)
    df_b = load_predictions(output_root, key_b)

    # Inner-join on panel_row_index. Different per-model below-floor
    # strips can drop different rows; the join only counts rows where
    # BOTH models produced a prediction.
    joined = df_a.merge(
        df_b,
        on="panel_row_index",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if joined.empty:
        return PairwiseRow(
            **base_row_fields,
            skipped_reason=_SKIP_REASON_EMPTY_JOIN,
        )

    y_true = joined["y_true_a"].to_numpy()  # equal to y_true_b
    y_pred_a = joined["y_pred_a"].to_numpy()
    y_pred_b = joined["y_pred_b"].to_numpy()

    pred_pearson = pearson_correlation(y_pred_a, y_pred_b)
    pred_spearman = spearman_correlation(y_pred_a, y_pred_b)

    if task_type in _CLASSIFICATION_TASK_TYPES:
        # Hard-prediction joint counts on the canonical class set.
        n11, n10, n01, n00 = joint_counts(y_true, y_pred_a, y_pred_b)
        n_samples = n11 + n10 + n01 + n00
        # Per-sample error = -log p(y_true). Pull each model's proba
        # matrix from the joined frame; the `_a` / `_b` suffixes
        # apply uniformly to every overlapping column including the
        # `y_proba_*` columns.
        proba_a = _proba_matrix_with_suffix(joined, "_a")
        proba_b = _proba_matrix_with_suffix(joined, "_b")
        if proba_a is not None and proba_b is not None and classes is not None:
            errors_a = classification_nll(y_true, proba_a, classes)
            errors_b = classification_nll(y_true, proba_b, classes)
            error_pearson = pearson_correlation(errors_a, errors_b)
        else:
            error_pearson = float("nan")
        return PairwiseRow(
            **base_row_fields,
            n_samples=n_samples,
            n11=n11,
            n10=n10,
            n01=n01,
            n00=n00,
            yule_q=yule_q(n11=n11, n10=n10, n01=n01, n00=n00),
            phi=phi_coefficient(n11=n11, n10=n10, n01=n01, n00=n00),
            disagreement_rate=disagreement_rate(n11=n11, n10=n10, n01=n01, n00=n00),
            double_fault_rate=double_fault_rate(n11=n11, n10=n10, n01=n01, n00=n00),
            pearson_pred_corr=pred_pearson,
            spearman_pred_corr=pred_spearman,
            pearson_error_corr=error_pearson,
        )

    # regression_point: only the cross-task correlations apply.
    errors_a = regression_signed_residual(y_true, y_pred_a)
    errors_b = regression_signed_residual(y_true, y_pred_b)
    return PairwiseRow(
        **base_row_fields,
        n_samples=len(y_true),
        pearson_pred_corr=pred_pearson,
        spearman_pred_corr=pred_spearman,
        pearson_error_corr=pearson_correlation(errors_a, errors_b),
    )


def _proba_matrix_with_suffix(df: pd.DataFrame, suffix: str) -> np.ndarray | None:
    """Extract a `(N, n_classes)` proba matrix from a suffixed merge.

    The B5 prediction shard's proba columns follow the pattern
    `y_proba_{k}`; after a pandas merge with `suffixes=("_a", "_b")`,
    one side's columns become `y_proba_{k}_a`. This helper finds
    them by suffix and returns the stacked matrix in NUMERIC `k`
    order (not lexicographic; otherwise `y_proba_10_a` sorts before
    `y_proba_2_a` and the class column ordering is corrupted for
    datasets with 10 or more classes).
    """
    prefix = "y_proba_"
    matches: list[str] = [
        str(c) for c in df.columns if str(c).startswith(prefix) and str(c).endswith(suffix)
    ]
    if not matches:
        return None
    cols = sorted(matches, key=lambda c: int(c[len(prefix) : -len(suffix)]))
    return df[cols].to_numpy(dtype=np.float64)


def _eligible_pair_cells(manifest: pd.DataFrame) -> pd.DataFrame:
    """Filter the B5 manifest to OK classification + regression_point
    rows (the cell types the pairwise driver can score)."""
    mask_ok = manifest["skipped_reason"].isna()
    mask_task = manifest["task_type"].isin(
        [*_CLASSIFICATION_TASK_TYPES, _REGRESSION_POINT_TASK_TYPE]
    )
    return manifest.loc[mask_ok & mask_task]


def _resolve_classification_classes(
    manifest: pd.DataFrame, output_root: Path, dataset_name: str
) -> np.ndarray | None:
    """Load any one OK prediction shard for `dataset_name` and read
    its `y_proba_*` columns to size the canonical class label
    array.
    """
    block = manifest.loc[
        (manifest["dataset_name"] == dataset_name) & manifest["skipped_reason"].isna()
    ]
    if block.empty:
        return None
    task_type = block.iloc[0]["task_type"]
    if task_type not in _CLASSIFICATION_TASK_TYPES:
        return None
    # Find any cell with a prediction shard.
    from benchmarks.manifest import cell_key

    for _, row in block.iterrows():
        key = cell_key(
            dataset_name=str(row["dataset_name"]),
            model_name=str(row["model_name"]),
            seed=int(row["seed"]),
            variant="default",
            fold_index=int(row["fold_index"]),
        )
        if not has_predictions(output_root, key):
            continue
        df = load_predictions(output_root, key)
        proba = proba_matrix(df)
        if proba is None:
            continue
        return np.arange(proba.shape[1])
    return None


def run_ensemble(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
) -> EnsembleExperimentResult:
    """Execute the pairwise ensemble-complementarity experiment.

    Reads the B5 raw-loss manifest under `output_root`, iterates
    over (dataset, model_a < model_b, seed, fold) pairs, computes
    the B6.2 pairwise statistics, and writes per-pair shards under
    `pairwise/`.

    Resumability: each pairwise cell has its own sentinel; the
    driver re-uses the B7.2 atomic-write pattern.

    Raises:
        ValueError: no `ensemble` experiment in the config, or the
            B5 manifest under `output_root` is empty / missing.
    """
    seeds = _resolve_seeds(config.experiments)
    logger.info(
        "run_ensemble: env.run_id=%s seeds=%s output_root=%s",
        env.run_id,
        seeds,
        output_root,
    )
    manifest = load_run(output_root)
    if manifest.empty:
        raise ValueError(
            f"run_ensemble: B5 manifest at {output_root} is empty; "
            f"run `--experiment=raw_loss` first"
        )

    eligible = _eligible_pair_cells(manifest)
    if eligible.empty:
        logger.warning(
            "run_ensemble: no eligible (binary | multiclass | "
            "regression_point) OK cells in the B5 manifest; nothing "
            "to pair"
        )
        return EnsembleExperimentResult(
            run_id=env.run_id,
            pairs_attempted=0,
            pairs_skipped_predictions_missing=0,
            pairs_skipped_empty_join=0,
            pairs_already_complete=0,
        )

    attempted = 0
    skipped_predictions_missing = 0
    skipped_empty_join = 0
    already_complete = 0

    for dataset_name, ds_block in eligible.groupby("dataset_name", sort=True):
        # All cells in a dataset share the same task_type by
        # registry invariant.
        task_type: TaskType = ds_block.iloc[0]["task_type"]
        if task_type not in _CLASSIFICATION_TASK_TYPES and task_type != _REGRESSION_POINT_TASK_TYPE:
            # Defensive; _eligible_pair_cells already filters.
            continue
        classes = _resolve_classification_classes(manifest, output_root, str(dataset_name))
        models_present = sorted(set(ds_block["model_name"]))
        # Enumerate unordered pairs (A < B).
        for i, model_a in enumerate(models_present):
            for model_b in models_present[i + 1 :]:
                for seed in seeds:
                    seed_cells = ds_block.loc[ds_block["seed"] == seed]
                    fold_indices = sorted(set(seed_cells["fold_index"]))
                    for fold_index in fold_indices:
                        key = pair_key(
                            dataset_name=str(dataset_name),
                            model_a=model_a,
                            model_b=model_b,
                            seed=int(seed),
                            fold_index=int(fold_index),
                        )
                        if is_pair_complete(output_root, key):
                            already_complete += 1
                            continue
                        key_started = time.perf_counter()
                        row = _pairwise_for_cell(
                            env=env,
                            output_root=output_root,
                            dataset_name=str(dataset_name),
                            model_a=model_a,
                            model_b=model_b,
                            task_type=task_type,
                            seed=int(seed),
                            fold_index=int(fold_index),
                            classes=classes,
                        )
                        if row.skipped_reason is None:
                            attempted += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_PREDICTIONS_MISSING):
                            skipped_predictions_missing += 1
                        elif row.skipped_reason.startswith(_SKIP_REASON_EMPTY_JOIN):
                            skipped_empty_join += 1
                        else:
                            # An unclassified `skipped_reason` indicates
                            # that `_pairwise_for_cell` grew a new skip
                            # path the outer loop does not know about.
                            # Loud-warn so a future B6-followup that
                            # adds a category does not silently
                            # un-count its skips.
                            logger.warning(
                                "run_ensemble: unclassified skip reason "
                                "%r for pair %s; counter not incremented",
                                row.skipped_reason,
                                key,
                            )
                        write_pairwise_cell(output_root, row)
                        logger.info(
                            "run_ensemble: pair %s done in %.2fs (reason=%s)",
                            key,
                            time.perf_counter() - key_started,
                            row.skipped_reason or "ok",
                        )

    summary = EnsembleExperimentResult(
        run_id=env.run_id,
        pairs_attempted=attempted,
        pairs_skipped_predictions_missing=skipped_predictions_missing,
        pairs_skipped_empty_join=skipped_empty_join,
        pairs_already_complete=already_complete,
    )
    logger.info("run_ensemble: complete: %s", summary.model_dump())
    return summary


__all__ = [
    "EnsembleExperimentResult",
    "PairwiseKeyError",
    "PairwiseRow",
    "is_pair_complete",
    "load_pairwise",
    "pair_key",
    "pairwise_dir",
    "pairwise_shard_path",
    "run_ensemble",
    "write_pairwise_cell",
]
