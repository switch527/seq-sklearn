"""C-MAPSS FD002 turbofan engine degradation dataset (NASA, OPEN).

260 engines under six operating conditions, single fault mode.
Harder than FD001 (the op-condition variation injects multi-modal
sensor signatures). Same sensor columns + RUL framing as FD001.
"""

import zipfile
from pathlib import Path

from benchmarks._io.cache import require_archive
from benchmarks._io.integrity import DatasetIOError
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.datasets.c_mapss_fd001 import (
    materialize_fd_panel,
    read_fd_table,
)
from benchmarks.registry.datasets import register_dataset

_NAME = "c_mapss_fd002"
_OP_COLS: tuple[str, ...] = ("op_setting_1", "op_setting_2", "op_setting_3")
_SENSOR_COLS: tuple[str, ...] = tuple(f"sensor_{i}" for i in range(1, 22))

_SPEC = DatasetSpec(
    name=_NAME,
    task_type="regression_point",
    access_tier="OPEN",
    size_tier="medium",
    balance="balanced",
    modality="numeric",
    source_uri=(
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    ),
    integrity_sha256=("74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"),
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
        "standard piecewise-linear RUL framing). FD002's six op "
        "conditions are exposed verbatim via op_setting_1/2/3 so "
        "the model can condition on them rather than assuming a "
        "single regime."
    ),
    densification_policy=None,
    excluded=False,
    citation=(
        "Saxena et al. 2008, Damage Propagation Modeling for "
        "Aircraft Engine Run-to-Failure Simulation, PHM Conf."
    ),
)


def load(cache_root: Path) -> PanelDataset:
    archive = require_archive(
        cache_root, "c_mapss_fd001", _SPEC.archive_basename, _SPEC.integrity_sha256
    )
    with zipfile.ZipFile(archive) as zf:
        try:
            text = zf.read("train_FD002.txt").decode("utf-8")
        except KeyError as exc:
            raise DatasetIOError(
                f"{_NAME}: archive {_SPEC.archive_basename!r} does not contain 'train_FD002.txt'"
            ) from exc
    df = read_fd_table(text)
    panel, y = materialize_fd_panel(df, _SPEC.feature_real_cols)
    return PanelDataset(spec=_SPEC, panel=panel, y=y)


register_dataset(_SPEC, load)
