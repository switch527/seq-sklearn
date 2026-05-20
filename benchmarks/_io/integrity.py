"""Typed errors + SHA-256 archive integrity check (B2.2 / B2.3).

`DatasetIntegrityError` fires when a cached archive's SHA-256 does
not match the spec; the cache layer NEVER silently uses the file and
NEVER auto-retries. `GatedDatasetUnavailableError` fires when a
GATED dataset is registered but its archive is missing from the
local cache, with a typed message naming the exact manual setup step
and the target cache path.

Both classes inherit from a common `DatasetIOError` base so a caller
can catch every dataset-IO failure with one `except` clause when
that's the right granularity.
"""

import hashlib
from pathlib import Path


class DatasetIOError(Exception):
    """Base for benchmark-suite dataset-IO errors."""


class DatasetIntegrityError(DatasetIOError):
    """A cached archive's SHA-256 does not match the spec hash.

    Raised by `verify_sha256` and by the cache layer; never silently
    swallowed; never auto-retried. The remediation is for the caller
    to delete the local file and re-download (or, for a GATED
    dataset, to re-export the archive from the source).
    """


class GatedDatasetUnavailableError(DatasetIOError):
    """A GATED dataset is registered but its archive is not in the
    local cache. The message names the exact manual setup step and
    the target cache path; the error is deterministic offline
    (B2.3)."""


_SHA256_CHUNK_BYTES = 1 << 20  # 1 MiB


def verify_sha256(path: Path, expected_sha256: str) -> None:
    """Verify the SHA-256 of ``path`` equals ``expected_sha256``.

    Reads the file in 1 MiB chunks so a huge archive does not blow
    the process RSS. Raises :class:`DatasetIntegrityError` on
    mismatch; returns ``None`` on success.

    The expected hash is a 64-character lower-case hex string
    (enforced by ``DatasetSpec.integrity_sha256`` at the schema
    layer); this function does not re-validate the format.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        DatasetIntegrityError: on hash mismatch.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_SHA256_CHUNK_BYTES):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected_sha256:
        raise DatasetIntegrityError(
            f"archive integrity check failed for {path}:\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            f"refusing to use the file; delete it and re-download "
            f"(GATED datasets: re-export from the source)"
        )
