"""Registration decorators + lookup helpers for datasets and models.

Phase B0 lands the registry shape (the `_REGISTRY` mappings, the
decorators, the typed lookup errors, the `list_*` / `get_*` helpers).
Phase B1 populates the dataset registry; Phase B2 populates the
model registry; both call the decorators here.
"""

from benchmarks.registry.datasets import (
    DatasetNotRegisteredError,
    get_dataset,
    get_loader,
    list_datasets,
    register_dataset,
)
from benchmarks.registry.models import (
    ModelNotRegisteredError,
    get_model,
    list_models,
    register_model,
)

__all__ = [
    "DatasetNotRegisteredError",
    "ModelNotRegisteredError",
    "get_dataset",
    "get_loader",
    "get_model",
    "list_datasets",
    "list_models",
    "register_dataset",
    "register_model",
]
