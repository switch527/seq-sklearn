"""Registration decorators + lookup helpers for datasets and models.

Phase B0 lands the registry shape (the `_REGISTRY` mappings, the
decorators, the typed lookup errors, the `list_*` / `get_*` helpers).
Phase B1 populates the dataset registry; Phase B2 populates the
model registry; Phase B5 adds the parallel adapter-factory map so
the experiment driver can build an adapter from its registered
name.
"""

from benchmarks.registry.datasets import (
    DatasetNotRegisteredError,
    get_dataset,
    get_loader,
    list_datasets,
    register_dataset,
)
from benchmarks.registry.models import (
    AdapterFactory,
    ModelNotRegisteredError,
    get_adapter_factory,
    get_model,
    instantiate_adapter,
    list_models,
    register_adapter,
    register_adapter_factory,
    register_model,
)

__all__ = [
    "AdapterFactory",
    "DatasetNotRegisteredError",
    "ModelNotRegisteredError",
    "get_adapter_factory",
    "get_dataset",
    "get_loader",
    "get_model",
    "instantiate_adapter",
    "list_datasets",
    "list_models",
    "register_adapter",
    "register_adapter_factory",
    "register_dataset",
    "register_model",
]
