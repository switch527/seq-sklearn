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
    "RollupRow",
    "load_rollup",
    "rollup_path",
    "write_rollup",
]

_ROLLUP_FILENAME = "bootstrap_rollup.parquet"


class RollupRow(BaseModel):
    """One per-(dataset, model, task_type) bootstrap CI entry.

    Skipped (dataset, model) groups emit a sentinel row with
    `primary_loss_mean / ci_lo / ci_hi == None` and the
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
    primary_loss_mean: float | None = None
    primary_loss_ci_lo: float | None = None
    primary_loss_ci_hi: float | None = None
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
