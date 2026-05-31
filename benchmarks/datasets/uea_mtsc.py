"""UEA Multivariate Time Series Classification loaders.

Three representative datasets from the UEA MTSC archive,
fetched via ``aeon.datasets.load_classification`` (which
downloads from timeseriesclassification.com on first use
and caches under ``~/.cache/aeon`` by default).

Conventions:

- One panel ENTITY per UEA instance (entity_id is the
  instance index, zero-based). One panel ROW per timestep.
  Each channel becomes a ``ch_<i>`` real-valued column.
- Class labels are integer-encoded by sorted-unique order
  (alphabetical for string labels). For binary specs, the
  positive label is the higher-sorted class.
- Labels broadcast across every row of an entity so the
  per-row F2 contract holds; the TSC adapter family's
  per-instance projection (``instance_labels``) picks the
  last-period label per entity.
- ``integrity_sha256`` pins the materialized ``X`` + ``y``
  byte sequence; aeon downloads the same archive bytes from
  timeseriesclassification.com so the hash is stable across
  reruns. A mirror or upstream change surfaces as a typed
  integrity failure.

Aeon is required at runtime; the loader imports lazily so
the module loads even when aeon is absent. The harness's
``optional_dep_missing`` machinery routes the missing-dep
case to a skip rather than crashing the run.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from benchmarks._io.integrity import DatasetIOError
from benchmarks.config import DatasetSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.registry.datasets import register_dataset


class UEAUnavailableError(RuntimeError):
    """``aeon`` is not installed; UEA datasets cannot be loaded.

    Subclasses RuntimeError so the B5/B8 driver's narrow
    adapter-error tuple routes the failure as a typed
    ``adapter_error`` skip rather than crashing the run.
    """


_AEON_CACHE = Path(".cache/aeon")


_VARIANTS: dict[str, dict[str, object]] = {
    "uea_basic_motions": {
        "aeon_name": "BasicMotions",
        "task_type": "multiclass",
        "size_tier": "small",
        # Sorted unique labels: badminton, running, standing, walking.
        # 4-class smoke test; ~80 instances total.
        "integrity_sha256": ("7af10de0f3bf254f0e3272747aa790947236cd8ffe61e19ff3c1a1cb42fd732d"),
        "n_timesteps": 100,
        "n_channels": 6,
        "citation": (
            "UEA Multivariate Time Series Classification Archive (2018), BasicMotions dataset."
        ),
    },
    "uea_heartbeat": {
        "aeon_name": "Heartbeat",
        "task_type": "binary",
        "size_tier": "medium",
        # Sorted unique labels: abnormal, normal. positive_label=1
        # → normal (higher-sorted) is the "positive" class in the
        # B5.1 framing convention.
        "integrity_sha256": ("0c6555bd2aeba46152a6878d9bdaa2aac60f13152d3c51642609e1010fd399fc"),
        "n_timesteps": 405,
        "n_channels": 61,
        "citation": (
            "UEA Multivariate Time Series Classification Archive "
            "(2018), Heartbeat dataset (PhysioNet 2016 Challenge)."
        ),
    },
    "uea_self_regulation_scp1": {
        "aeon_name": "SelfRegulationSCP1",
        "task_type": "binary",
        "size_tier": "medium",
        # Sorted unique labels: negativity, positivity.
        "integrity_sha256": ("aa3ef3c9c13d52057cb508aa3f2ed948799843d86d65e9934fae4a71ca680961"),
        "n_timesteps": 896,
        "n_channels": 6,
        "citation": (
            "UEA Multivariate Time Series Classification Archive "
            "(2018), SelfRegulationSCP1 dataset (BCI competition)."
        ),
    },
}


def _channel_cols(n_channels: int) -> tuple[str, ...]:
    return tuple(f"ch_{i}" for i in range(n_channels))


def _build_spec(name: str, params: dict[str, object]) -> DatasetSpec:
    n_channels = int(params["n_channels"])  # pyright: ignore[reportArgumentType]
    task_type: Literal["binary", "multiclass"] = params["task_type"]  # type: ignore[assignment]
    positive_label = 1 if task_type == "binary" else None
    return DatasetSpec(
        name=name,
        task_type=task_type,
        access_tier="OPEN",
        size_tier=params["size_tier"],  # type: ignore[arg-type]
        balance="balanced",
        modality="numeric",
        source_uri=(
            f"https://www.timeseriesclassification.com/description.php"
            f"?Dataset={params['aeon_name']}"
        ),
        integrity_sha256=str(params["integrity_sha256"]),
        archive_basename=f"{params['aeon_name']}.ts",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=_channel_cols(n_channels),
        feature_categorical_cols=(),
        # The expanding-window splitter's min-rows floor is
        # `n_splits + lookback`. With the harness's default
        # n_splits=5 a lookback equal to the full sequence
        # length forces every entity below the floor (the train
        # set comes back empty). Trim the lookback by a 5-fold
        # safety margin so the splitter retains every entity;
        # the TSC adapter clips to the trailing `lookback` rows
        # per the F2 trailing-window convention, losing only
        # the first ~1% of timesteps.
        lookback=int(params["n_timesteps"]) - 5,  # pyright: ignore[reportArgumentType]
        observation_cutoff_rule=(
            "Each UEA instance is a fixed-length whole-sequence sample; "
            "the label is broadcast across every timestep row of that "
            "entity (the TSC adapter's instance_labels helper projects "
            "back to per-instance via the last-period label per F2)."
        ),
        densification_policy=None,
        positive_label=positive_label,
        excluded=False,
        citation=str(params["citation"]),
    )


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, list[object]]:
    """Encode UEA string/object labels to sorted-unique int order."""
    unique = sorted(np.unique(y).tolist())
    mapping = {label: idx for idx, label in enumerate(unique)}
    encoded = np.fromiter((mapping[v] for v in y.tolist()), dtype=np.int64, count=len(y))
    return encoded, unique


def _load_via_aeon(aeon_name: str, cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from aeon.datasets import load_classification  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise UEAUnavailableError(
            f"`aeon` is not installed; install with `uv pip install aeon` to "
            f"load UEA dataset {aeon_name!r}"
        ) from exc
    extract_path = cache_root / "aeon"
    extract_path.mkdir(parents=True, exist_ok=True)
    loaded = load_classification(aeon_name, extract_path=str(extract_path))
    x, y = loaded[0], loaded[1]
    return np.asarray(x), np.asarray(y)


def _materialize_panel(
    x: np.ndarray, y_encoded: np.ndarray, n_channels: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn (n_instances, n_channels, n_timesteps) into an F2 panel.

    Each instance becomes one entity with ``n_timesteps`` rows; the
    instance's encoded label is broadcast across every row of that
    entity so the F2 per-row contract holds. The TSC adapter's
    ``instance_labels`` helper later projects back to one label per
    instance via the last-period row.
    """
    n_instances, _, n_timesteps = x.shape
    cols = _channel_cols(n_channels)
    # (instance, channel, timestep) → (instance, timestep, channel) → flatten
    transposed = np.transpose(x, (0, 2, 1)).astype(np.float32, copy=False)
    flat = transposed.reshape(n_instances * n_timesteps, n_channels)
    entity_ids = np.repeat(np.arange(n_instances, dtype=np.int64), n_timesteps)
    periods = np.tile(np.arange(n_timesteps, dtype=np.int64), n_instances)
    panel_df = pd.DataFrame(flat, columns=pd.Index(list(cols)))
    panel_df.insert(0, "period", periods)
    panel_df.insert(0, "entity_id", entity_ids)
    y_per_row = np.repeat(y_encoded, n_timesteps)
    return panel_df, y_per_row


def _make_loader(name: str, params: dict[str, object]) -> Callable[[Path], PanelDataset]:
    def load(cache_root: Path) -> PanelDataset:
        x, y_raw = _load_via_aeon(str(params["aeon_name"]), cache_root)
        # Hash raw aeon bytes (x float64 + y object/str) so the SHA
        # is stable regardless of our downstream encoding.
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(y_raw).tobytes())
        actual = h.hexdigest()
        expected = str(params["integrity_sha256"])
        if actual != expected:
            raise DatasetIOError(
                f"{name}: integrity SHA-256 mismatch; expected {expected} "
                f"but got {actual}. The UEA archive may have changed "
                f"upstream; re-pin the spec's `integrity_sha256` after "
                f"auditing."
            )
        y_encoded, _classes = _encode_labels(y_raw)
        panel, y = _materialize_panel(x, y_encoded, int(params["n_channels"]))  # type: ignore[arg-type]
        return PanelDataset(spec=_SPECS[name], panel=panel, y=y)

    return load


_SPECS: dict[str, DatasetSpec] = {
    name: _build_spec(name, params) for name, params in _VARIANTS.items()
}

for _name, _params in _VARIANTS.items():
    register_dataset(_SPECS[_name], _make_loader(_name, _params))
