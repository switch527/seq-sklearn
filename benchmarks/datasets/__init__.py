"""Dataset loader registrations (Phase B1).

Importing this package triggers each loader module's
``register_dataset(spec, load)`` call. The harness imports this
package once at startup; subsequent ``get_dataset`` / ``get_loader``
calls hit the populated registry.

Phase B1 scope: the OPEN reference loader (`c_mapss_fd001`) plus the
GATED-pattern flagship (`amex_default`). The remaining B2 roster
entries land in Phase B1-followup iterations (one branch per
loader); the registry-invariants test asserts every registered
entry's shape, so the iteration is gated, not silent.
"""

# Importing each loader module registers its spec at module load
# time. The harness only needs to import `benchmarks.datasets` once.
# Pyright's "unused import" diagnostic fires on these symbols
# because their value is the side effect of the import (each module
# calls `register_dataset` at module load); `__all__` below is the
# contract that re-exports the names. Per-symbol suppression because
# the from-line `pyright: ignore` does not propagate.
from benchmarks.datasets import (
    amex_default,  # pyright: ignore[reportUnusedImport]
    c_mapss_fd001,  # pyright: ignore[reportUnusedImport]
)
from benchmarks.datasets._base import (
    LoaderCallable,  # pyright: ignore[reportUnusedImport]
    PanelDataset,  # pyright: ignore[reportUnusedImport]
)

__all__ = [
    "LoaderCallable",
    "PanelDataset",
    "amex_default",
    "c_mapss_fd001",
]
