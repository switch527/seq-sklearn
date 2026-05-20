"""Dataset registry: name -> `DatasetSpec`.

Phase B1 will populate this via a `register_dataset(spec)` call at
the bottom of each `benchmarks/datasets/<name>.py` module; the
registry is import-time-populated when those modules are imported.
"""

from benchmarks.config import DatasetSpec

_REGISTRY: dict[str, DatasetSpec] = {}


class DatasetNotRegisteredError(KeyError):
    """Raised when a config asks for a dataset name that no module
    has registered. Distinct from `KeyError` for typed catching."""


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Register a dataset spec by name. Idempotent only if the spec
    is identical; a different spec under the same name raises."""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"dataset {spec.name!r} already registered with a different "
            f"spec; refusing to overwrite"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset by name; raises `DatasetNotRegisteredError`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise DatasetNotRegisteredError(
            f"no dataset registered under name {name!r}; "
            f"registered datasets: {sorted(_REGISTRY)}"
        ) from exc


def list_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
