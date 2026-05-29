"""Bootstrap-rollup manifest (Phase B13 / B5.4).

Per-(dataset, model, task_type) bootstrap-CI rollup record + the
parquet shard write/load helpers. The shard is written ONCE per
run by the B13 aggregator; the B5 leaderboard renderer reads it
and dispatches to the CI variant when the rollup's
`manifest_fingerprint` matches the live run manifest's.

The rollup shard is intentionally separate from the per-cell
predictions shards (no atomic-shard-plus-sentinel resumability
machinery here; the rollup is whole-run, not per-cell). A
re-run overwrites the file in one atomic rename.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EnsembleLiftRollupRow",
    "HPOUpliftRollupRow",
    "PairwiseRollupRow",
    "RollupRow",
    "TrainingTimeRollupRow",
    "aggregator_failed_sentinel_path",
    "ensemble_lift_aggregator_failed_sentinel_path",
    "ensemble_lift_rollup_path",
    "hpo_uplift_aggregator_failed_sentinel_path",
    "hpo_uplift_rollup_path",
    "load_ensemble_lift_rollup",
    "load_hpo_uplift_rollup",
    "load_pairwise_rollup",
    "load_rollup",
    "load_training_time_rollup",
    "pairwise_aggregator_failed_sentinel_path",
    "pairwise_rollup_path",
    "rollup_path",
    "training_time_aggregator_failed_sentinel_path",
    "training_time_rollup_path",
    "write_ensemble_lift_rollup",
    "write_hpo_uplift_rollup",
    "write_pairwise_rollup",
    "write_rollup",
    "write_training_time_rollup",
]

_ROLLUP_FILENAME = "bootstrap_rollup.parquet"
_AGGREGATOR_FAILED_SENTINEL_FILENAME = "bootstrap_aggregator_failed.txt"
_PAIRWISE_ROLLUP_FILENAME = "bootstrap_pairwise_rollup.parquet"
_PAIRWISE_AGGREGATOR_FAILED_SENTINEL_FILENAME = "bootstrap_pairwise_aggregator_failed.txt"
_TRAINING_TIME_ROLLUP_FILENAME = "bootstrap_training_time_rollup.parquet"
_TRAINING_TIME_AGGREGATOR_FAILED_SENTINEL_FILENAME = "bootstrap_training_time_aggregator_failed.txt"
_HPO_UPLIFT_ROLLUP_FILENAME = "bootstrap_hpo_uplift_rollup.parquet"
_HPO_UPLIFT_AGGREGATOR_FAILED_SENTINEL_FILENAME = "bootstrap_hpo_uplift_aggregator_failed.txt"
_ENSEMBLE_LIFT_ROLLUP_FILENAME = "bootstrap_ensemble_lift_rollup.parquet"
_ENSEMBLE_LIFT_AGGREGATOR_FAILED_SENTINEL_FILENAME = "bootstrap_ensemble_lift_aggregator_failed.txt"


class RollupRow(BaseModel):
    """One per-(dataset, model, task_type) bootstrap CI entry.

    Skipped (dataset, model) groups emit a sentinel row with
    `primary_metric_mean / ci_lo / ci_hi == None` and the
    `bootstrap_skipped_reason` populated. Non-skipped rows carry
    finite CI values + a populated `manifest_fingerprint` so the
    renderer's freshness check (B13.4) can compare.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    task_type: str
    primary_metric: str  # "log_loss" | "rmse"
    n_seeds: int = Field(ge=0)
    n_cells_evaluated: int = Field(ge=0)  # ok cells in the bootstrap
    n_skipped_cells: int = Field(ge=0)
    n_rows: int = Field(ge=0)  # total rows across included cells
    n_entities: int = Field(ge=0)  # unique entities across included cells
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    # B13.4 Gemini-C3 freshness: the manifest fingerprint at
    # aggregation time. The renderer asserts
    # rollup.manifest_fingerprint == manifest.fingerprint() before
    # joining; stale rollups fall back to the std variant + a
    # warning footnote.
    manifest_fingerprint: str


def rollup_path(root: Path) -> Path:
    """`{root}/bootstrap_rollup.parquet`."""
    return root / _ROLLUP_FILENAME


def aggregator_failed_sentinel_path(root: Path) -> Path:
    """`{root}/bootstrap_aggregator_failed.txt`.

    Cross-module FS contract (B13.0): the CLI wrapper writes
    this file on `RawRollupError`; the renderer reads it before
    rollup dispatch. Single source of truth so a future
    maintainer changing either side cannot drift the filename.
    """
    return root / _AGGREGATOR_FAILED_SENTINEL_FILENAME


def pairwise_rollup_path(root: Path) -> Path:
    """`{root}/bootstrap_pairwise_rollup.parquet` (B14)."""
    return root / _PAIRWISE_ROLLUP_FILENAME


def pairwise_aggregator_failed_sentinel_path(root: Path) -> Path:
    """`{root}/bootstrap_pairwise_aggregator_failed.txt` (B14).

    Cross-module FS contract (B14.0): the pairwise CLI wrapper
    writes this file on `RawRollupError`; the pairwise renderer
    reads it before rollup dispatch. Independent from the B5
    sentinel: a B6 failure must not mask a B5 or B7 success.
    """
    return root / _PAIRWISE_AGGREGATOR_FAILED_SENTINEL_FILENAME


def training_time_rollup_path(root: Path) -> Path:
    """`{root}/bootstrap_training_time_rollup.parquet` (B14)."""
    return root / _TRAINING_TIME_ROLLUP_FILENAME


def training_time_aggregator_failed_sentinel_path(root: Path) -> Path:
    """`{root}/bootstrap_training_time_aggregator_failed.txt` (B14).

    Cross-module FS contract (B14.0): the training-time CLI
    wrapper writes this file on `RawRollupError`; the training-
    time renderer reads it before rollup dispatch. Independent
    from the B5 + B6 sentinels.
    """
    return root / _TRAINING_TIME_AGGREGATOR_FAILED_SENTINEL_FILENAME


def hpo_uplift_rollup_path(root: Path) -> Path:
    """`{root}/bootstrap_hpo_uplift_rollup.parquet` (B15)."""
    return root / _HPO_UPLIFT_ROLLUP_FILENAME


def hpo_uplift_aggregator_failed_sentinel_path(root: Path) -> Path:
    """`{root}/bootstrap_hpo_uplift_aggregator_failed.txt` (B15).

    Cross-module FS contract (B15.0): the HPO-uplift CLI
    wrapper writes this file on `RawRollupError`; the HPO-
    uplift renderer reads it before rollup dispatch.
    Independent from the B5 + B6 + B7 sentinels.
    """
    return root / _HPO_UPLIFT_AGGREGATOR_FAILED_SENTINEL_FILENAME


def write_rollup(root: Path, rows: Sequence[RollupRow]) -> None:
    """Persist the rollup rows as a single parquet shard.

    Atomic-rename semantics: writes to `{path}.tmp.{pid}` then
    `os.replace` into place so a partial write never leaves a
    half-rendered file behind. Idempotent: a second call with the
    same rows over an existing file overwrites cleanly.
    """
    dest = rollup_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Empty rollup: write an empty DataFrame with the canonical
        # column set so the loader's schema check still passes.
        column_names: list[str] = list(RollupRow.model_fields.keys())
        df = pd.DataFrame({col: [] for col in column_names})
    else:
        df = pd.DataFrame([row.model_dump() for row in rows])
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


def load_rollup(root: Path) -> list[RollupRow]:
    """Load the rollup shard from `{root}/bootstrap_rollup.parquet`.

    Returns an empty list when the shard is absent (the renderer
    treats this as "no rollup ran, fall back to scalar leaderboard").

    Raises:
        FileNotFoundError: only when explicitly asked via
            `rollup_path(root).exists()` callers; this loader's
            convention is empty-on-absent to match the renderer
            dispatch.
    """
    dest = rollup_path(root)
    if not dest.exists():
        return []
    df = pd.read_parquet(dest)
    if df.empty:
        return []
    rows: list[RollupRow] = []
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    for record in records:
        # pandas nullable types serialize None as `pd.NA`; pydantic
        # rejects pd.NA on Optional[float] fields, so we coerce.
        for key, value in list(record.items()):
            if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                record[key] = None
        rows.append(RollupRow.model_validate(record))
    return rows


class PairwiseRollupRow(BaseModel):
    """One per-(dataset, model_a, model_b, task_type) pairwise CI entry.

    The `primary_metric_*` field names mirror `RollupRow` so the
    shared `format_ci_cell` helper works without a discriminator
    (B14.0). Here they carry the `complementarity_score` (B6's
    rank key). Skipped groups emit a sentinel row with
    `primary_metric_mean / ci_lo / ci_hi == None` and the
    `bootstrap_skipped_reason` populated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_a: str
    model_b: str
    task_type: str
    primary_metric: str  # always "complementarity_score" at v1
    n_seeds: int = Field(ge=0)
    n_cells_evaluated: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str


class TrainingTimeRollupRow(BaseModel):
    """One per-(dataset, model, hardware_tier, task_type) training-time CI entry.

    The `primary_metric_*` field names mirror `RollupRow` so the
    shared render helpers reuse without a discriminator
    (B14.0). Here they carry the `wall_seconds` aggregate. The
    grouping is by hardware_tier as well so CPU and GPU
    measurements for the same model are not collapsed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    hardware_tier: str
    task_type: str
    primary_metric: str  # always "wall_seconds" at v1
    n_seeds: int = Field(ge=0)
    n_cells_evaluated: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str


def _write_rows_atomic(
    dest: Path,
    rows: Sequence[Any],
    *,
    column_names: Sequence[str],
) -> None:
    """Shared atomic-rename parquet write for any rollup-row sequence."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pd.DataFrame({col: [] for col in column_names})
    else:
        df = pd.DataFrame([row.model_dump() for row in rows])
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


def write_pairwise_rollup(root: Path, rows: Sequence[PairwiseRollupRow]) -> None:
    """Persist the pairwise rollup rows as a single parquet shard.

    Atomic-rename semantics matching `write_rollup` (B14).
    """
    _write_rows_atomic(
        pairwise_rollup_path(root),
        rows,
        column_names=list(PairwiseRollupRow.model_fields.keys()),
    )


def load_pairwise_rollup(root: Path) -> list[PairwiseRollupRow]:
    """Load the pairwise rollup shard from
    `{root}/bootstrap_pairwise_rollup.parquet`.

    Returns `[]` when the shard is absent (matches the
    renderer's "no rollup" dispatch convention).
    """
    dest = pairwise_rollup_path(root)
    if not dest.exists():
        return []
    df = pd.read_parquet(dest)
    if df.empty:
        return []
    rows: list[PairwiseRollupRow] = []
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                record[key] = None
        rows.append(PairwiseRollupRow.model_validate(record))
    return rows


def write_training_time_rollup(root: Path, rows: Sequence[TrainingTimeRollupRow]) -> None:
    """Persist the training-time rollup rows as a single parquet shard.

    Atomic-rename semantics matching `write_rollup` (B14).
    """
    _write_rows_atomic(
        training_time_rollup_path(root),
        rows,
        column_names=list(TrainingTimeRollupRow.model_fields.keys()),
    )


def load_training_time_rollup(root: Path) -> list[TrainingTimeRollupRow]:
    """Load the training-time rollup shard from
    `{root}/bootstrap_training_time_rollup.parquet`.

    Returns `[]` when the shard is absent.
    """
    dest = training_time_rollup_path(root)
    if not dest.exists():
        return []
    df = pd.read_parquet(dest)
    if df.empty:
        return []
    rows: list[TrainingTimeRollupRow] = []
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                record[key] = None
        rows.append(TrainingTimeRollupRow.model_validate(record))
    return rows


class HPOUpliftRollupRow(BaseModel):
    """One per-(dataset, model, task_type) HPO-uplift Δ CI entry (B15).

    Δ = `default_mean - tuned_mean` on the primary loss
    (positive Δ means tuning helped). The bootstrap resamples
    paired (seed, fold) cells where BOTH default and tuned
    arms ran; each resample's mean is one Δ statistic.

    Sentinel rows (`primary_metric_*=None`,
    `bootstrap_skipped_reason` populated) cover four cases:
    `default_only`, `tuned_only`, `paired_but_no_valid_loss`
    (inherited from the existing B8 UpliftRow flags) plus
    `no_paired_cells` (new B15 case when the inner-join
    yields zero matches).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    task_type: str
    primary_metric: str  # always "delta" at v1
    primary_loss_column: str  # the underlying loss column for audit (e.g. "log_loss")
    n_seeds: int = Field(ge=0)  # seeds in the OK B8 manifest for this group
    n_folds: int = Field(ge=0)  # folds in the OK B8 manifest for this group
    n_cells_paired: int = Field(ge=0)  # cells where BOTH default + tuned ran
    n_skipped_cells: int = Field(ge=0)  # paired cells dropped from the bootstrap (NaN loss)
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str


def write_hpo_uplift_rollup(root: Path, rows: Sequence[HPOUpliftRollupRow]) -> None:
    """Persist the HPO-uplift rollup rows as a single parquet shard.

    Atomic-rename semantics matching `write_rollup` (B15).
    """
    _write_rows_atomic(
        hpo_uplift_rollup_path(root),
        rows,
        column_names=list(HPOUpliftRollupRow.model_fields.keys()),
    )


def load_hpo_uplift_rollup(root: Path) -> list[HPOUpliftRollupRow]:
    """Load the HPO-uplift rollup shard from
    `{root}/bootstrap_hpo_uplift_rollup.parquet`.

    Returns `[]` when the shard is absent.
    """
    dest = hpo_uplift_rollup_path(root)
    if not dest.exists():
        return []
    df = pd.read_parquet(dest)
    if df.empty:
        return []
    rows: list[HPOUpliftRollupRow] = []
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                record[key] = None
        rows.append(HPOUpliftRollupRow.model_validate(record))
    return rows


class EnsembleLiftRollupRow(BaseModel):
    """One per-dataset ensemble-lift Δ CI entry (B16 / D-B13.4).

    Δ = `loss(GBM) - loss(GBM + seq)` on the primary loss
    (positive Δ means the seq-augmented ensemble helped). The
    bootstrap resamples paired (seed, fold) cells where BOTH
    GBM and the GBM+seq ensemble produced predictions; each
    resample's mean Δ is one statistic.

    Sentinel rows (`primary_metric_*=None`,
    `bootstrap_skipped_reason` populated) cover three cases
    matching B11's actual driver flags:

    - `no_gbm_predictions`: the baseline (GBM) family is
      absent from the manifest cells consumed by the join
      (mirrors `PerDatasetLift.no_gbm_predictions=True`).
    - `no_seq_predictions`: the seq family is absent
      (mirrors `PerDatasetLift.no_seq_predictions=True`).
    - `all_cells_skipped_in_manifest`: the manifest rows for
      this dataset exist but all map to skipped cells.

    When both B11 flags trip on the same group, the
    aggregator emits the lexically-first sentinel string
    (`no_gbm_predictions` < `no_seq_predictions`) for
    determinism. The seq-family and baseline-family
    identifiers are NOT carried on every row; the renderer
    reads them from `EnsembleLiftExperimentResult` directly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    task_type: str
    primary_metric: str  # always "delta_loss" at v1
    primary_loss_column: str  # the underlying loss column for audit ("log_loss" | "rmse")
    n_seeds: int = Field(ge=0)  # seeds in the OK B11 manifest for this dataset
    n_folds: int = Field(ge=0)  # folds in the OK B11 manifest for this dataset
    n_cells_paired: int = Field(ge=0)  # cells where BOTH GBM + GBM+seq ran
    n_skipped_cells: int = Field(ge=0)  # paired cells dropped from the bootstrap (NaN loss)
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str


def ensemble_lift_rollup_path(root: Path) -> Path:
    """`{root}/bootstrap_ensemble_lift_rollup.parquet` (B16)."""
    return root / _ENSEMBLE_LIFT_ROLLUP_FILENAME


def ensemble_lift_aggregator_failed_sentinel_path(root: Path) -> Path:
    """`{root}/bootstrap_ensemble_lift_aggregator_failed.txt` (B16).

    Cross-module FS contract (B16.0): the ensemble-lift CLI
    wrapper writes this file on `RawRollupError`; the
    ensemble-lift renderer reads it before rollup dispatch.
    Independent from the B5 + B6 + B7 + B8 sentinels so a B11
    failure must not mask any other report's success.
    """
    return root / _ENSEMBLE_LIFT_AGGREGATOR_FAILED_SENTINEL_FILENAME


def write_ensemble_lift_rollup(root: Path, rows: Sequence[EnsembleLiftRollupRow]) -> None:
    """Persist the ensemble-lift rollup rows as a single parquet shard.

    Atomic-rename semantics matching `write_rollup` (B16).
    """
    _write_rows_atomic(
        ensemble_lift_rollup_path(root),
        rows,
        column_names=list(EnsembleLiftRollupRow.model_fields.keys()),
    )


def load_ensemble_lift_rollup(root: Path) -> list[EnsembleLiftRollupRow]:
    """Load the ensemble-lift rollup shard from
    `{root}/bootstrap_ensemble_lift_rollup.parquet`.

    Returns `[]` when the shard is absent.
    """
    dest = ensemble_lift_rollup_path(root)
    if not dest.exists():
        return []
    df = pd.read_parquet(dest)
    if df.empty:
        return []
    rows: list[EnsembleLiftRollupRow] = []
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                record[key] = None
        rows.append(EnsembleLiftRollupRow.model_validate(record))
    return rows
