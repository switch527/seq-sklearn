"""B2.2 / B7.2 cache layer + integrity errors."""

from pathlib import Path

import pytest
from benchmarks._io.cache import (
    archive_path,
    panel_cache_path,
    require_archive,
    resolve_cache_dir,
)
from benchmarks._io.integrity import (
    DatasetIntegrityError,
    DatasetIOError,
    GatedDatasetUnavailableError,
    verify_sha256,
)


def test_resolve_cache_dir_override_wins(tmp_path: Path) -> None:
    out = resolve_cache_dir(tmp_path / "explicit")
    assert out == tmp_path / "explicit"
    assert out.is_dir()


def test_resolve_cache_dir_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQ_SKLEARN_BENCHMARKS_CACHE", str(tmp_path / "from_env"))
    out = resolve_cache_dir()
    assert out == tmp_path / "from_env"
    assert out.is_dir()


def test_resolve_cache_dir_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without override and without env var, resolves to the user-
    home default. Verified by checking the home-relative path; no
    side effect on the user's actual home directory because the
    test runs in a fresh process under pytest's mkdtemp scope and
    the directory creation is idempotent."""
    monkeypatch.delenv("SEQ_SKLEARN_BENCHMARKS_CACHE", raising=False)
    out = resolve_cache_dir()
    expected = Path("~/.cache/seq-sklearn-benchmarks").expanduser()
    assert out == expected


def test_archive_path_creates_parent_directory(tmp_path: Path) -> None:
    path = archive_path(tmp_path, "my_dataset", "archive.zip")
    assert path == tmp_path / "archives" / "my_dataset" / "archive.zip"
    assert path.parent.is_dir()
    assert not path.exists()


def test_panel_cache_path_creates_parent_directory(tmp_path: Path) -> None:
    path = panel_cache_path(tmp_path, "my_dataset")
    assert path == tmp_path / "panels" / "my_dataset.parquet"
    assert path.parent.is_dir()


def test_verify_sha256_succeeds_on_match(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"benchmark canary payload"
    path.write_bytes(payload)
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    verify_sha256(path, sha)  # returns None on success


def test_verify_sha256_raises_typed_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"actual payload")
    with pytest.raises(DatasetIntegrityError, match="integrity check failed"):
        verify_sha256(path, "0" * 64)


def test_verify_sha256_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_sha256(tmp_path / "nope.bin", "0" * 64)


def test_require_archive_returns_path_on_match(tmp_path: Path) -> None:
    target = archive_path(tmp_path, "ds", "archive.bin")
    payload = b"valid"
    target.write_bytes(payload)
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    got = require_archive(tmp_path, "ds", "archive.bin", sha)
    assert got == target


def test_require_archive_missing_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        require_archive(tmp_path, "ds", "archive.bin", "0" * 64)


def test_typed_errors_share_dataset_io_error_base() -> None:
    """The two named dataset-IO errors share a common base for
    callers that want a single `except` clause."""
    assert issubclass(DatasetIntegrityError, DatasetIOError)
    assert issubclass(GatedDatasetUnavailableError, DatasetIOError)
