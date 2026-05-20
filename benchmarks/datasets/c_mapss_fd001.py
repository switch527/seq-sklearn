"""C-MAPSS FD001 turbofan engine degradation dataset (NASA, OPEN).

100 engines under a single operating condition; the supervised
target is Remaining Useful Life (RUL) in cycles. Each engine
contributes one sequence of per-cycle sensor readings (3 operating-
condition columns + 21 sensor columns); the target per row is
``max_cycle_for_engine - current_cycle``, a standard piecewise-
linear RUL framing.

The archive (``CMAPSSData.zip``) is fetched from the NASA Prognostics
Data Repository per the spec's ``source_uri``. The loader does NOT
download in CI: the test path is a tmp-path fixture with a recorded
mini-archive. Production paths download once, integrity-check, and
materialize the panel to Parquet (B7.2 atomic shard-then-sentinel).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks._io.cache import require_archive
from benchmarks._io.integrity import DatasetIOError
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry.datasets import register_dataset

_NAME = "c_mapss_fd001"

# The 21 numbered sensor columns + 3 operating-condition columns per
# the official C-MAPSS dataset documentation. Order matches the
# trailing columns of `train_FD001.txt`.
_OP_COLS: tuple[str, ...] = ("op_setting_1", "op_setting_2", "op_setting_3")
_SENSOR_COLS: tuple[str, ...] = tuple(f"sensor_{i}" for i in range(1, 22))

# The FD001 training file has whitespace-separated columns in this
# order; the loader pulls them by position rather than by header (the
# raw archive has no header row).
_RAW_COLUMNS: tuple[str, ...] = ("entity_id", "cycle", *_OP_COLS, *_SENSOR_COLS)

_SPEC = DatasetSpec(
    name=_NAME,
    task_type="regression_point",
    access_tier="OPEN",
    size_tier="medium",
    balance="balanced",
    modality="numeric",
    source_uri=(
        "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
        "pcoe/pcoe-data-set-repository/"
    ),
    integrity_sha256=(
        # The archive is not bundled with this code; the hash here is
        # a placeholder that the release engineer overwrites once they
        # have downloaded the official archive bytes and recorded the
        # canonical SHA-256. The placeholder is structurally valid
        # (64 lowercase hex chars) so the pydantic pattern accepts it;
        # the integrity-check at load time will fail until the real
        # hash is pinned.
        "deadbeef" * 8
    ),
    archive_basename="CMAPSSData.zip",
    entity_col="entity_id",
    time_col="cycle",
    target_col="RUL",
    feature_real_cols=_OP_COLS + _SENSOR_COLS,
    feature_categorical_cols=(),
    lookback=30,
    observation_cutoff_rule=(
        "Each engine's full per-cycle sequence is used; the target "
        "for row (engine_id, cycle) is RUL = max_cycle - cycle (the "
        "standard piecewise-linear RUL framing). No external cutoff."
    ),
    densification_policy=None,  # cycle index is already regular
    excluded=False,
    citation=(
        "Saxena et al. 2008, Damage Propagation Modeling for "
        "Aircraft Engine Run-to-Failure Simulation, PHM Conf."
    ),
)


def _read_fd001_table(text: str) -> pd.DataFrame:
    """Parse the whitespace-separated FD001 training text into a
    typed DataFrame with the canonical column names. Pulled out so
    the unit test can exercise it on a synthetic mini-payload
    without unpacking a real archive."""
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        header=None,
        names=list(_RAW_COLUMNS),
        engine="python",
    )
    # The official files have two trailing empty columns from the
    # whitespace-padding convention; pandas drops them already, but
    # an `entity_id` that's NaN would slip through. Guard explicitly.
    if bool(df["entity_id"].isna().any()):
        raise DatasetIOError(
            f"{_NAME}: parsed file has rows with missing entity_id; "
            f"archive layout may have changed"
        )
    df["entity_id"] = df["entity_id"].astype(np.int64)
    df["cycle"] = df["cycle"].astype(np.int64)
    return df


def _materialize_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute the RUL target and emit the panel + y in F2 contract
    shape (one row per `(entity_id, cycle)`, sorted by entity then
    cycle, panel columns match the spec)."""
    sorted_df: pd.DataFrame = df.sort_values(
        ["entity_id", "cycle"]
    ).reset_index(drop=True)
    max_cycle = sorted_df.groupby("entity_id")["cycle"].transform("max")
    rul = (max_cycle - sorted_df["cycle"]).astype(np.float64)
    panel_cols = list(_SPEC.feature_real_cols)
    panel = pd.DataFrame(sorted_df[["entity_id", "cycle", *panel_cols]].copy())
    y: np.ndarray = np.asarray(rul.to_numpy(), dtype=np.float64)
    return panel, y


def load(cache_root: Path) -> PanelDataset:
    """Load C-MAPSS FD001 from the cache root.

    Reads the archive at ``cache_root/archives/c_mapss_fd001/CMAPSSData.zip``,
    verifies its SHA-256 against the spec, unpacks ``train_FD001.txt``,
    parses it, and emits a ``PanelDataset`` per the F2 contract.

    Raises:
        FileNotFoundError: archive is not in the cache (production
            paths would download here; CI uses a tmp-path fixture).
        DatasetIntegrityError: archive SHA-256 mismatch.
        DatasetIOError: the archive does not contain
            ``train_FD001.txt`` or the file is malformed.
    """
    archive = require_archive(
        cache_root, _NAME, _SPEC.archive_basename, _SPEC.integrity_sha256
    )
    with zipfile.ZipFile(archive) as zf:
        try:
            text = zf.read("train_FD001.txt").decode("utf-8")
        except KeyError as exc:
            raise DatasetIOError(
                f"{_NAME}: archive {_SPEC.archive_basename!r} does not "
                f"contain 'train_FD001.txt'"
            ) from exc
    df = _read_fd001_table(text)
    panel, y = _materialize_panel(df)
    return PanelDataset(spec=_SPEC, panel=panel, y=y)


register_dataset(_SPEC, load)
