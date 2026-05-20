"""Per-cell result manifest (B7.2 atomic shard-then-sentinel + B8.1).

The harness writes one parquet shard per `(dataset, model, seed,
variant, fold_index)` cell plus a sentinel JSON marker. The two-step
pattern gives the B7.2 resumability contract:

1. Each cell's row goes to a temp parquet file, then `os.replace`s
   into the canonical shard path (atomic on POSIX).
2. After the shard is durably renamed, the sentinel JSON is written
   the same way.
3. The runner checks the sentinel BEFORE computing a cell; if
   present, the cell is skipped.

Crash semantics: a crash mid-shard-write leaves no shard and no
sentinel (the temp file is orphaned but does not corrupt the
manifest). A crash between shard rename and sentinel write leaves
the shard but no sentinel; the runner re-computes that cell and
the second shard write replaces the first atomically (idempotent
from the harness's perspective; the per-cell metric values are
deterministic per (seed, fold) by B8.2).

The `ResultRow` schema flattens every B4 metric into its own column
(absent metrics are `None`) so a single parquet schema covers every
task type. The leaderboard renderer (`benchmarks/report/raw_loss.py`)
filters by task type when ranking.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from benchmarks.config import TaskType

logger = logging.getLogger(__name__)


_RESULTS_DIRNAME = "results"
_SENTINEL_DIRNAME = "_done"
_SHARD_SUFFIX = ".parquet"
_SENTINEL_SUFFIX = ".json"


# A cell key is `{dataset}__{model}__seed_{seed}__{variant}__fold_{fold}`.
# The dataset and model names are validated at registry-time to be
# non-empty; the cell-key builder enforces additional FS-safe character
# rules so a stray underscore-rich dataset name cannot collide with the
# separator scheme.
_CELL_KEY_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]*$")


class CellKeyError(ValueError):
    """Raised when a cell-key component contains characters that
    would conflict with the `__`-separator scheme (e.g. a dataset
    name containing `__` or starting with `_`)."""


def cell_key(
    *,
    dataset_name: str,
    model_name: str,
    seed: int,
    variant: str,
    fold_index: int,
) -> str:
    """Build the canonical cell-key string used as the shard /
    sentinel filename stem.

    The format is intentionally human-readable so a partial-results
    directory can be inspected by eye; the regex guard on each token
    rejects names that would alias the `__` separator (e.g. a
    dataset named `c_mapss__fd001`).

    Raises:
        CellKeyError: a token would collide with the separator.
    """
    for label, value in (
        ("dataset_name", dataset_name),
        ("model_name", model_name),
        ("variant", variant),
    ):
        if "__" in value:
            raise CellKeyError(
                f"cell_key: {label}={value!r} contains the cell-key "
                f"separator '__'; rename the registry entry"
            )
        if not _CELL_KEY_TOKEN_RE.fullmatch(value):
            raise CellKeyError(
                f"cell_key: {label}={value!r} must match {_CELL_KEY_TOKEN_RE.pattern!r}"
            )
    if fold_index < 0:
        raise CellKeyError(f"cell_key: fold_index must be >= 0; got {fold_index}")
    return f"{dataset_name}__{model_name}__seed_{seed}__{variant}__fold_{fold_index}"


def results_dir(root: Path) -> Path:
    """`{root}/results/`: the per-run shard directory."""
    return root / _RESULTS_DIRNAME


def sentinel_dir(root: Path) -> Path:
    """`{root}/results/_done/`: the per-run sentinel marker directory.

    Kept distinct from the shard directory so the runner can iterate
    shards (`results/*.parquet`) without filtering out sentinel files,
    and so a `glob` over sentinels lists completed cells directly.
    """
    return root / _RESULTS_DIRNAME / _SENTINEL_DIRNAME


def shard_path(root: Path, key: str) -> Path:
    """Path of the parquet shard for cell `key`."""
    return results_dir(root) / f"{key}{_SHARD_SUFFIX}"


def sentinel_path(root: Path, key: str) -> Path:
    """Path of the sentinel JSON marker for cell `key`."""
    return sentinel_dir(root) / f"{key}{_SENTINEL_SUFFIX}"


def is_cell_complete(root: Path, key: str) -> bool:
    """Return True iff the sentinel exists for `key`.

    The runner consults this BEFORE invoking the adapter. The shard
    is intentionally NOT checked: a half-written shard without a
    sentinel is treated as missing per B7.2 and is overwritten on
    the next attempt.
    """
    return sentinel_path(root, key).exists()


def _atomic_write_bytes(payload: bytes, dest: Path) -> None:
    """Write `payload` to `dest` atomically.

    Uses the write-to-temp-then-`os.replace` POSIX-atomic pattern;
    the temp path is `dest` with a `.tmp.{pid}` suffix so two
    concurrent writers (which the harness never produces, but
    defense in depth) cannot collide on the temp.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(payload)
    tmp.replace(dest)


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write `df` as a parquet shard atomically.

    pandas' `to_parquet` does not expose a write-to-bytes API for
    the default engine, so we write to a temp file in the same
    directory (same filesystem -> atomic rename) and then
    `os.replace` it into place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)


class ResultRow(BaseModel):
    """One row of a raw-loss experiment per cell.

    The schema is the union of B8.1 metadata and every B4 metric
    name; absent metrics (those that don't apply to the cell's
    `task_type`) are `None`. `mape_skip_reason` and `pinball_json`
    are string columns; pinball is JSON-encoded because the keys
    depend on the configured quantile levels.

    `skipped_reason` is non-None when the cell could not be
    evaluated (e.g. `ProbaUnsupportedError`, a missing optional
    dependency, an out-of-memory crash); the metric columns are all
    `None` in that case. The leaderboard renderer filters skipped
    cells out of the rank but reports them in a "skipped" footnote.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # B8.1 reproducibility metadata.
    library_git_sha: str
    run_id: str
    started_at_utc: str  # ISO-8601 UTC; the runner timestamps at row build

    # Cell identity (matches the cell-key components).
    dataset_name: str
    model_name: str
    seed: int
    variant: str = "default"
    task_type: TaskType
    fold_index: int = Field(ge=0)
    n_folds: int = Field(ge=1)

    # Protocol metadata (B4.1, B8.1).
    split_fingerprint: str  # SHA-256 hex
    l_resolved: int = Field(ge=1)

    # Environment.
    hardware_tier: str  # "cpu" | "gpu_single" (free-form for B5; B7 tightens)
    profile: str  # "smoke" | "standard" | "full"
    hpo_tier: str = "none"  # B5 is always "none"; B8 will populate

    # Skipped-cell reason; non-None means the metric columns are absent.
    skipped_reason: str | None = None

    # Number of test-fold rows whose predictions came back as `nan`
    # (rows from entities below `min_periods_predict` in the deep
    # models; A9.1 lookback-floor row from the splitter). These rows
    # are stripped before the metric layer so a single below-floor
    # row does not corrupt the whole-fold metric; the count is
    # recorded for B8.1 audit. `None` on skipped cells.
    n_below_floor_dropped: int | None = None

    # Classification metrics (Binary + Multiclass union).
    log_loss: float | None = None
    accuracy: float | None = None
    precision_zd0: float | None = None
    recall_zd0: float | None = None
    f1_zd0: float | None = None
    precision_macro_zd0: float | None = None
    precision_weighted_zd0: float | None = None
    recall_macro_zd0: float | None = None
    recall_weighted_zd0: float | None = None
    f1_macro_zd0: float | None = None
    f1_weighted_zd0: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    roc_auc_macro_ovr: float | None = None
    pr_auc_macro_ovr: float | None = None
    balanced_accuracy: float | None = None
    mcc: float | None = None
    brier: float | None = None
    brier_multiclass_mean: float | None = None
    ece_q15: float | None = None

    # Regression metrics.
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    mape: float | None = None
    mape_skip_reason: str | None = None
    pinball_json: str | None = None  # JSON dict {pinball_q0.10: ...}

    # Resource (B5.3).
    wall_seconds: float | None = None
    peak_rss_bytes: int | None = None
    peak_cuda_bytes: int | None = None
    median_predict_ms: float | None = None
    p95_predict_ms: float | None = None
    n_predict_reps: int | None = None

    def cell_key(self) -> str:
        """Return the canonical cell key for this row."""
        return cell_key(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            seed=self.seed,
            variant=self.variant,
            fold_index=self.fold_index,
        )


def write_cell(root: Path, row: ResultRow) -> None:
    """Persist one cell: atomic parquet shard, then sentinel.

    The two-phase write is B7.2's resumability contract. A crash
    between the two phases re-runs that one cell on the next
    invocation; the shard will be overwritten by the second attempt.
    """
    key = row.cell_key()
    df = pd.DataFrame([row.model_dump()])
    # A skipped cell has every metric column `None`; pandas types
    # those as `object` dtype. When that shard is later concatenated
    # with a shard whose same column holds floats, pandas emits a
    # FutureWarning that becomes a dtype error in upcoming releases.
    # Coerce the metric columns to `float64` so the shard schema is
    # consistent across skipped and successful cells.
    df = _coerce_optional_float_columns(df)
    _atomic_write_parquet(df, shard_path(root, key))
    payload = json.dumps(
        {
            "cell_key": key,
            "completed_at_utc": row.started_at_utc,
            "skipped_reason": row.skipped_reason,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write_bytes(payload, sentinel_path(root, key))


def load_run(root: Path) -> pd.DataFrame:
    """Read every shard under `root/results/` and concatenate.

    Returns an empty DataFrame with the `ResultRow` columns when no
    shards are present. Order is sorted by cell-key string so two
    runs over the same cells produce byte-identical concatenations.

    Raises:
        FileNotFoundError: `root` does not exist. An empty
            `root/results/` directory is NOT an error (the manifest
            is genuinely empty).
    """
    if not root.exists():
        raise FileNotFoundError(
            f"load_run: results root {root!r} does not exist; expected "
            f"a directory containing 'results/' from a prior run"
        )
    dir_ = results_dir(root)
    if not dir_.exists():
        return _empty_manifest_dataframe()
    shards: list[Path] = sorted(dir_.glob(f"*{_SHARD_SUFFIX}"))
    if not shards:
        return _empty_manifest_dataframe()
    frames = [pd.read_parquet(p) for p in shards]
    return pd.concat(frames, ignore_index=True)


def _empty_manifest_dataframe() -> pd.DataFrame:
    """Return a 0-row DataFrame with the `ResultRow` column schema.

    Keeps `load_run` returning a typed empty frame instead of an
    untyped one so downstream `pd.concat` with a non-empty frame
    matches columns exactly.
    """
    columns = pd.Index(list(ResultRow.model_fields.keys()))
    return pd.DataFrame(columns=columns)


def _optional_float_columns() -> tuple[str, ...]:
    """Compute the list of `ResultRow` fields whose annotation is
    `float | None`.

    Computed by inspection once per process so the column list stays
    in sync with the schema; adding a new optional-float metric
    automatically picks it up.
    """
    out: list[str] = []
    for name, info in ResultRow.model_fields.items():
        annotation = info.annotation
        # `Optional[float]` and `float | None` both stringify to
        # "float | None" under pydantic's annotation repr. Cheap
        # check that does not require typing introspection.
        if str(annotation) == "float | None":
            out.append(name)
    return tuple(out)


def _coerce_optional_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every `float | None` ResultRow column to `float64`.

    On a skipped row pydantic emits Python `None` for every metric
    field; pandas types those as `object`. Concatenating an object-
    typed null column with a float-typed shard from a successful row
    triggers a `FutureWarning` that becomes an error in pandas 3.x.
    The cast normalizes to `float64` (NaN for None), which matches
    the dtype of a successful row.
    """
    cols = [c for c in _optional_float_columns() if c in df.columns]
    if not cols:
        return df
    df = df.copy()
    df[cols] = df[cols].astype("float64")
    return df


def list_completed_keys(root: Path) -> tuple[str, ...]:
    """Return the sorted tuple of completed cell keys.

    Used by the runner's per-cell skip check; reading sentinel
    presence is `O(1)` per cell whereas reading the parquet shards
    is `O(N)` per shard.
    """
    if not sentinel_dir(root).exists():
        return ()
    keys = [
        p.name[: -len(_SENTINEL_SUFFIX)] for p in sentinel_dir(root).glob(f"*{_SENTINEL_SUFFIX}")
    ]
    return tuple(sorted(keys))


def pinball_to_json(pinball: dict[str, float] | None) -> str | None:
    """Serialize a pinball dict to JSON for the shard column.

    `None` (e.g., classification cells) round-trips to `None`. The
    JSON form uses sorted keys + compact separators so the column
    is byte-stable across runs at the same quantile configuration.
    """
    if pinball is None:
        return None
    return json.dumps(pinball, separators=(",", ":"), sort_keys=True)


def pinball_from_json(payload: str | None) -> dict[str, float] | None:
    """Inverse of `pinball_to_json`."""
    if payload is None:
        return None
    raw: dict[str, Any] = json.loads(payload)
    return {k: float(v) for k, v in raw.items()}


def metrics_to_row_fields(metrics_dump: dict[str, Any]) -> dict[str, Any]:
    """Translate a `MetricsRecord.model_dump()` dict to the flat
    `ResultRow` field shape.

    The MetricsRecord union dumps directly to a flat dict for every
    task type EXCEPT regression_quantile, whose `pinball` field is a
    nested dict. This helper hoists that dict into the
    JSON-serialized `pinball_json` column. Every other field name
    aligns 1:1 with a `ResultRow` field.
    """
    fields = dict(metrics_dump)
    pinball = fields.pop("pinball", None)
    if pinball is not None:
        fields["pinball_json"] = pinball_to_json(pinball)
    return fields
