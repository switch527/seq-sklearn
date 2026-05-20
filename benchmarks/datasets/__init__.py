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
# These imports are pure side-effect: each module's bottom calls
# `register_dataset` and adds itself to the live registry.
from benchmarks.datasets import amex_default, c_mapss_fd001

__all__: list[str] = ["amex_default", "c_mapss_fd001"]

# Silence "unused name" diagnostics on the side-effect modules; the
# `__all__` listing above is the contract that they are re-exported.
_ = (amex_default, c_mapss_fd001)

__all__: list[str] = []
