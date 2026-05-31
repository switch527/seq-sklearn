"""C-MAPSS FD001-FD004 turbofan engine degradation datasets (NASA, OPEN).

All four FD subsets ship in one archive (``CMAPSSData.zip``); each
subset differs in engine count and operating-condition complexity:

- FD001: 100 engines, single op condition, single fault mode. Easy.
- FD002: 260 engines, six op conditions, single fault mode.
- FD003: 100 engines, single op condition, two fault modes.
- FD004: 248 engines, six op conditions, two fault modes. Hardest.

Each engine contributes one sequence of per-cycle sensor readings (3
operating-condition columns + 21 sensor columns); the target per row
is ``max_cycle_for_engine - current_cycle``, the standard piecewise-
linear RUL framing.

This module owns the FD001 spec + loader and exports
:func:`read_fd_table` + :func:`materialize_fd_panel` so the FD002 /
FD003 / FD004 modules can reuse the parsing path against the same
archive. The archive is fetched from the PHM Society S3 mirror per
the spec's ``source_uri`` and SHA-pinned.
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
        # PHM Society S3 mirror of NASA's Prognostics Data
        # Repository CMAPSS archive (NASA's original DASHlink URL
        # migrated; this mirror is the canonical public URL the
        # PHM community now references). The mirror ships the
        # CMAPSSData.zip inside an outer "6. Turbofan Engine
        # Degradation Simulation Data Set/" wrapper that must be
        # unwrapped before placing into the cache (this loader
        # expects the inner archive at archive_basename).
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    ),
    integrity_sha256=(
        # SHA-256 of the INNER `CMAPSSData.zip` (the file the
        # loader's `require_archive` checks). Pinned against the
        # PHM S3 mirror's archive bytes captured 2026-05-31; the
        # archive contents are NASA's 2008 release and have not
        # changed since.
        "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
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


def read_fd_table(text: str) -> pd.DataFrame:
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
    # `sep=r"\s+"` collapses consecutive whitespace, so trailing
    # whitespace at line ends produces no extra tokens. If a mirror
    # ever ships an off-shape file (extra tokens, missing leading
    # entity-id column) the misalignment surfaces as a NaN
    # `entity_id`. Guard explicitly so the failure is typed at the
    # parse boundary rather than miscast to int downstream.
    if bool(df["entity_id"].isna().any()):
        raise DatasetIOError(
            f"{_NAME}: parsed file has rows with missing entity_id; archive layout may have changed"
        )
    df["entity_id"] = df["entity_id"].astype(np.int64)
    df["cycle"] = df["cycle"].astype(np.int64)
    return df


def materialize_fd_panel(
    df: pd.DataFrame, panel_cols: tuple[str, ...]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute the RUL target and emit the panel + y in F2 contract
    shape (one row per `(entity_id, cycle)`, sorted by entity then
    cycle, panel columns match the caller's spec). Shared by every
    FD0xx variant; the spec selects which sensor columns to include
    via ``panel_cols``."""
    sorted_df: pd.DataFrame = df.sort_values(["entity_id", "cycle"]).reset_index(drop=True)
    max_cycle = sorted_df.groupby("entity_id")["cycle"].transform("max")
    rul = (max_cycle - sorted_df["cycle"]).astype(np.float64)
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
    archive = require_archive(cache_root, _NAME, _SPEC.archive_basename, _SPEC.integrity_sha256)
    with zipfile.ZipFile(archive) as zf:
        try:
            text = zf.read("train_FD001.txt").decode("utf-8")
        except KeyError as exc:
            raise DatasetIOError(
                f"{_NAME}: archive {_SPEC.archive_basename!r} does not contain 'train_FD001.txt'"
            ) from exc
    df = read_fd_table(text)
    panel, y = materialize_fd_panel(df, _SPEC.feature_real_cols)
    return PanelDataset(spec=_SPEC, panel=panel, y=y)


register_dataset(_SPEC, load)
