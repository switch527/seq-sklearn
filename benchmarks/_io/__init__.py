"""I/O helpers for the benchmark harness: archive caching, integrity
verification, and typed gated-access errors (B2.2 / B2.3)."""

from benchmarks._io.integrity import (
    DatasetIntegrityError,
    GatedDatasetUnavailableError,
    verify_sha256,
)

__all__ = [
    "DatasetIntegrityError",
    "GatedDatasetUnavailableError",
    "verify_sha256",
]
