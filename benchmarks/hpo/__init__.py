"""Per-family HPO search-space modules (Phase B8).

Each `<family>` module exposes:

- `SEARCH_SPACE_SIZE`: int reporting the number of searchable
  hyperparameters in the default search (B6.4.0 parity disclosure).
- `sample_hyperparameters(trial, *, model_spec, dataset_spec)
  -> dict[str, Any]`: returns a kwargs dict the registered adapter
  factory consumes via its `hyperparameters=` channel.

B8 ships the `seq_sklearn` module; GBM and TSC follow when those
adapter families register (per `benchmarks/adapters/__init__.py`).
"""

# Side-effect import: pulling `benchmarks.hpo.seq_sklearn` triggers
# its `register_hpo_space(...)` call so `HPO_REGISTRY` is populated
# before the HPO uplift driver inspects it.
from benchmarks.hpo import seq_sklearn  # noqa: F401  # pyright: ignore[reportUnusedImport]
from benchmarks.hpo._base import (
    HPO_REGISTRY,
    HPOFamilyNotRegisteredError,
    HPOSpace,
    get_hpo_space,
    register_hpo_space,
)

__all__ = [
    "HPO_REGISTRY",
    "HPOFamilyNotRegisteredError",
    "HPOSpace",
    "get_hpo_space",
    "register_hpo_space",
]
