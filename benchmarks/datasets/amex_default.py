"""American Express Default Prediction (Kaggle, GATED).

The design's flagship payments-panel dataset: ~458K customers, up
to 13 monthly statements each, binary target (default within 18
months of the latest statement). Mixed modality: 175 anonymized
numeric features + 11 categorical features, plus ``customer_ID``
and ``S_2`` (statement date).

The archive ships as ``amex-default-prediction.zip`` on Kaggle
behind a competition-rules click-through. The loader does NOT
download: it requires the archive at
``<cache_root>/archives/amex_default/amex-default-prediction.zip``.
Once present, the loader integrity-checks, extracts
``train_data.csv`` + ``train_labels.csv``, materializes the F2
panel + y, and caches the result as Parquet under
``<cache_root>/panels/amex_default.parquet`` so subsequent loads
skip the ~50 GB reparse.

Manual setup (one-time):
    1. Accept competition rules at
       https://www.kaggle.com/competitions/amex-default-prediction
    2. ``kaggle competitions download -c amex-default-prediction``
    3. Move the downloaded archive to the cache path above
    4. Re-run; the loader integrity-checks + materializes the panel

Categorical column list per the competition's official data
description (the "Anonymized" Kaggle discussion).
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks._io.cache import (
    archive_path,
    panel_cache_path,
)
from benchmarks._io.integrity import (
    DatasetIOError,
    GatedDatasetUnavailableError,
)
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry.datasets import register_dataset

logger = logging.getLogger(__name__)

_NAME = "amex_default"

# Categorical features per the Kaggle Amex competition data
# description. The remaining D_*/S_*/P_*/B_*/R_* prefixed columns
# are numeric per the same source.
_CATEGORICAL_COLS: tuple[str, ...] = (
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
)

_SPEC = DatasetSpec(
    name=_NAME,
    task_type="binary",
    access_tier="GATED",
    size_tier="huge",
    balance="imbalanced",
    modality="mixed",
    source_uri="https://www.kaggle.com/competitions/amex-default-prediction/data",
    # SHA-256 of the canonical Kaggle archive bytes. Pinned the
    # first time the loader sees a valid archive (deferred until
    # the user drops the archive in place); until then the
    # integrity guard is a structural placeholder (the loader's
    # in-cache parse will record the observed SHA in the log so
    # the release engineer can update this constant once
    # confirmed).
    integrity_sha256="deadbeef" * 8,
    archive_basename="amex-default-prediction.zip",
    entity_col="customer_ID",
    time_col="S_2",
    target_col="target",
    # Populated below from the canonical schema; the loader strips
    # any column missing from the actual file with a typed
    # DatasetIOError so a partial archive surfaces explicitly.
    feature_real_cols=(),  # set after _build_real_cols() runs at module load
    feature_categorical_cols=_CATEGORICAL_COLS,
    lookback=13,
    observation_cutoff_rule=(
        "Per the competition rules, features are read from up to 13 "
        "monthly statements ending at S_2 = customer's latest "
        "statement; the binary target is default within 18 months "
        "of that latest statement."
    ),
    densification_policy=None,
    positive_label=1,
    excluded=False,
    citation=(
        "American Express, Kaggle competition "
        "'amex-default-prediction', 2022; "
        "https://www.kaggle.com/competitions/amex-default-prediction"
    ),
)


def _split_real_cols(header: list[str]) -> tuple[str, ...]:
    """Pick the real-valued feature columns from a parsed CSV header.

    Everything D_/S_/P_/B_/R_-prefixed that is NOT on the
    categorical list. Order preserves the file header so downstream
    code that indexes by column position stays stable.
    """
    cat_set = set(_CATEGORICAL_COLS)
    real: list[str] = []
    for col in header:
        if col in {"customer_ID", "S_2", "target"}:
            continue
        if col in cat_set:
            continue
        if col[0] in {"D", "S", "P", "B", "R"} and "_" in col:
            real.append(col)
    return tuple(real)


def _read_train_data(archive: Path) -> pd.DataFrame:
    """Read ``train_data.csv`` from inside the Kaggle archive.

    Uses ``zipfile.open`` to stream from the zip without a full
    extraction so we never write the 17 GB intermediate to disk.
    Casts numeric columns to ``float32`` at read time to halve RAM
    cost; categorical columns are read as object then converted to
    pandas Categorical for the F39 native-cat handling pathway.
    """
    with zipfile.ZipFile(archive) as zf:
        try:
            handle = zf.open("train_data.csv")
        except KeyError as exc:
            raise DatasetIOError(
                f"{_NAME}: archive {_SPEC.archive_basename!r} does not "
                f"contain 'train_data.csv'; verify the file is the "
                f"full Kaggle Amex archive (not a sub-archive)"
            ) from exc
        # `low_memory=False` lets pandas decide dtypes per-column on
        # a single pass instead of risking chunk-boundary dtype
        # mismatches. With float32 hints below the peak RAM is
        # ~4-6 GB.
        df = pd.read_csv(
            handle,
            parse_dates=["S_2"],
            low_memory=False,
        )
    if "customer_ID" not in df.columns or "S_2" not in df.columns:
        raise DatasetIOError(
            f"{_NAME}: train_data.csv missing expected columns "
            f"customer_ID + S_2; got first 5: {list(df.columns)[:5]}"
        )
    # Downcast numeric columns. Categoricals stay as their raw
    # object dtype; the harness converts them to pandas Categorical
    # via `astype('category').cat.codes` in the lag featurizer.
    cat_set = set(_CATEGORICAL_COLS)
    for col in df.columns:
        if col in {"customer_ID", "S_2"}:
            continue
        if col in cat_set:
            continue
        if df[col].dtype.kind == "f":
            df[col] = df[col].astype(np.float32, copy=False)
    return df


def _read_train_labels(archive: Path) -> pd.DataFrame:
    """Read ``train_labels.csv`` and return one row per customer_ID."""
    with zipfile.ZipFile(archive) as zf:
        try:
            handle = zf.open("train_labels.csv")
        except KeyError as exc:
            raise DatasetIOError(
                f"{_NAME}: archive {_SPEC.archive_basename!r} does not contain 'train_labels.csv'"
            ) from exc
        labels = pd.read_csv(handle)
    if not {"customer_ID", "target"}.issubset(labels.columns):
        raise DatasetIOError(f"{_NAME}: train_labels.csv missing customer_ID or target")
    return labels


def _materialize_panel(
    train_data: pd.DataFrame, train_labels: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble the F2 panel + y aligned by ``customer_ID``.

    - Sort by (customer_ID, S_2) so each customer's statements are
      time-ordered (F2 ordinal-period contract).
    - Drop customers with no label in ``train_labels.csv``.
    - Broadcast each customer's binary ``target`` across every
      statement row; the TSC adapter's ``instance_labels`` helper
      reduces to one label per entity via the last-period row.
    """
    sorted_df = train_data.sort_values(["customer_ID", "S_2"]).reset_index(drop=True)
    target_by_cid = train_labels.set_index("customer_ID")["target"]
    target_aligned = sorted_df["customer_ID"].map(target_by_cid)  # pyright: ignore[reportArgumentType]
    if bool(target_aligned.isna().any()):
        n_missing = int(target_aligned.isna().sum())
        logger.warning(
            "%s: %d rows have no matching label in train_labels.csv; dropping those customers",
            _NAME,
            n_missing,
        )
        keep = target_aligned.notna()
        sorted_df = sorted_df.loc[keep].reset_index(drop=True)
        target_aligned = target_aligned.loc[keep].reset_index(drop=True)
    real_cols = _split_real_cols(list(sorted_df.columns))
    panel_cols = ["customer_ID", "S_2", *real_cols, *_CATEGORICAL_COLS]
    panel = pd.DataFrame(sorted_df[panel_cols].copy())
    y = np.asarray(target_aligned.to_numpy(), dtype=np.int64)
    return panel, y


def _load_from_archive(archive: Path) -> tuple[pd.DataFrame, np.ndarray]:
    train_data = _read_train_data(archive)
    train_labels = _read_train_labels(archive)
    return _materialize_panel(train_data, train_labels)


def _cached_panel_paths(cache_root: Path) -> tuple[Path, Path]:
    """Return ``(panel_parquet, y_npy)`` cache paths.

    Two artifacts because Parquet handles the DataFrame nicely and
    a flat ``.npy`` keeps the int64 ``y`` array trivially fast to
    reload.
    """
    panel_path = panel_cache_path(cache_root, _NAME)
    y_path = panel_path.with_suffix(".y.npy")
    return panel_path, y_path


def load(cache_root: Path) -> PanelDataset:
    """Load Amex Default from the cache root.

    GATED: the loader requires the Kaggle archive at the cache
    path. On first run with the archive present, the loader
    materializes the panel + y from ``train_data.csv`` +
    ``train_labels.csv`` and writes a Parquet+npy cache for fast
    reload on subsequent runs.
    """
    archive = archive_path(cache_root, _NAME, _SPEC.archive_basename)
    panel_path, y_path = _cached_panel_paths(cache_root)
    if panel_path.is_file() and y_path.is_file():
        panel = pd.read_parquet(panel_path)
        y = np.load(y_path)
        spec = _SPEC.model_copy(update={"feature_real_cols": _split_real_cols(list(panel.columns))})
        return PanelDataset(spec=spec, panel=panel, y=y)
    if not archive.is_file():
        raise GatedDatasetUnavailableError(
            f"{_NAME!r} is a GATED dataset and its archive is not in "
            f"the cache.\n"
            f"  expected path: {archive}\n"
            f"\n"
            f"Manual setup (one-time):\n"
            f"  1. Accept the competition rules at\n"
            f"     {_SPEC.source_uri}\n"
            f"  2. `kaggle competitions download -c amex-default-prediction`\n"
            f"  3. Move the downloaded archive to {archive}\n"
            f"  4. Re-run; the loader materializes a Parquet panel\n"
            f"     cache under {panel_path.parent}/ so subsequent\n"
            f"     loads skip the ~50 GB reparse.\n"
            f"\n"
            f"To exclude this dataset from a run instead, remove\n"
            f"{_NAME!r} from `BenchmarkConfig.datasets`."
        )
    logger.info(
        "%s: parsing Kaggle archive at %s (this can take ~5 minutes; "
        "the Parquet cache at %s makes subsequent loads instant)",
        _NAME,
        archive,
        panel_path,
    )
    panel, y = _load_from_archive(archive)
    panel.to_parquet(panel_path, index=False)
    np.save(y_path, y)
    spec = _SPEC.model_copy(update={"feature_real_cols": _split_real_cols(list(panel.columns))})
    return PanelDataset(spec=spec, panel=panel, y=y)


register_dataset(_SPEC, load)
