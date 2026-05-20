"""Local archive cache for benchmark datasets (B7.2).

Two-level cache:

1. ``<cache_root>/archives/<dataset_name>/<basename>``: the raw
   downloaded archive bytes; integrity-checked against
   ``DatasetSpec.integrity_sha256`` on every access.
2. ``<cache_root>/panels/<dataset_name>.parquet``: the materialized
   panel after the first successful load; subsequent loads short-
   circuit to this file.

The cache root defaults to ``~/.cache/seq-sklearn-benchmarks/`` and
is overridable via the ``SEQ_SKLEARN_BENCHMARKS_CACHE`` env var or
the ``BenchmarkConfig.cache_dir`` field.

Phase B1 lands the cache-resolution + integrity-check layer. The
Parquet write-after-first-load path is wired in here so the loaders
in this phase can opt into it on completion; the optimization is
correctness-neutral and CI does not exercise it (every test
materializes from a tmp_path archive directly).
"""

import os
from pathlib import Path

from benchmarks._io.integrity import verify_sha256

_DEFAULT_CACHE = Path("~/.cache/seq-sklearn-benchmarks").expanduser()
_ENV_VAR = "SEQ_SKLEARN_BENCHMARKS_CACHE"


def resolve_cache_dir(override: Path | None = None) -> Path:
    """Resolve the active cache root.

    Precedence: explicit ``override`` argument > ``SEQ_SKLEARN_BENCHMARKS_CACHE``
    env var > the user-home default.

    Creates the directory if it does not exist.
    """
    if override is not None:
        cache = override
    else:
        env = os.environ.get(_ENV_VAR)
        cache = Path(env).expanduser() if env else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def archive_path(cache_root: Path, dataset_name: str, basename: str) -> Path:
    """Resolve the canonical archive path for a dataset, creating the
    parent directory. Does not verify integrity; that is
    :func:`require_archive`."""
    parent = cache_root / "archives" / dataset_name
    parent.mkdir(parents=True, exist_ok=True)
    return parent / basename


def require_archive(
    cache_root: Path,
    dataset_name: str,
    basename: str,
    expected_sha256: str,
) -> Path:
    """Return the path to the cached archive for ``dataset_name``,
    integrity-checked.

    Raises:
        FileNotFoundError: if the archive is not in the cache.
        DatasetIntegrityError: if its SHA-256 does not match
            ``expected_sha256``.
    """
    path = archive_path(cache_root, dataset_name, basename)
    if not path.is_file():
        raise FileNotFoundError(
            f"archive {basename!r} for dataset {dataset_name!r} not found in cache at {path}"
        )
    verify_sha256(path, expected_sha256)
    return path


def panel_cache_path(cache_root: Path, dataset_name: str) -> Path:
    """Resolve the materialized-panel Parquet path for ``dataset_name``.
    Creates the parent directory."""
    parent = cache_root / "panels"
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{dataset_name}.parquet"
