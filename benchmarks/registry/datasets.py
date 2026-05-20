"""Dataset registry: name -> (`DatasetSpec`, loader).

Phase B1 populates this via `register_dataset(spec, loader)` calls
at the bottom of each `benchmarks/datasets/<name>.py` module; the
registry is import-time-populated when those modules are imported.

B1.5 exclusions (datasets the rubric explicitly rejects) are
registered with `excluded=True` on the spec and a noop loader; the
registry-invariants test asserts every excluded entry carries an
`exclusion_reason`.
"""

from typing import TYPE_CHECKING

from benchmarks.config import DatasetSpec

if TYPE_CHECKING:
    # `LoaderCallable` lives in `benchmarks.datasets._base`. The
    # runtime cycle goes the other way: `benchmarks.registry` is
    # imported FIRST (by tests, by `benchmarks.run`), then
    # `benchmarks.datasets.__init__` triggers loader-module imports
    # which call `register_dataset` here. If this module imports
    # `LoaderCallable` at module-top, the chain becomes
    # `registry -> datasets._base -> datasets/__init__.py -> loader
    # module -> registry.register_dataset (not yet defined) -> fail`.
    # The TYPE_CHECKING guard keeps pyright's typed-signature view
    # while the runtime import order stays acyclic.
    from benchmarks.datasets._base import LoaderCallable

_REGISTRY: dict[str, DatasetSpec] = {}
_LOADERS: "dict[str, LoaderCallable]" = {}


class DatasetNotRegisteredError(KeyError):
    """Raised when a config asks for a dataset name that no module
    has registered. Distinct from `KeyError` for typed catching."""


def register_dataset(spec: DatasetSpec, loader: "LoaderCallable") -> DatasetSpec:
    """Register a dataset spec + loader by name.

    Idempotent only if BOTH the spec and the loader object are
    identical to the existing registration; otherwise raises so a
    silent overwrite cannot happen at module-import time.
    """
    existing_spec = _REGISTRY.get(spec.name)
    existing_loader = _LOADERS.get(spec.name)
    if existing_spec is not None and existing_spec != spec:
        raise ValueError(
            f"dataset {spec.name!r} already registered with a different spec; refusing to overwrite"
        )
    if existing_loader is not None and existing_loader is not loader:
        raise ValueError(
            f"dataset {spec.name!r} already registered with a different "
            f"loader callable; refusing to overwrite"
        )
    _REGISTRY[spec.name] = spec
    _LOADERS[spec.name] = loader
    return spec


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset by name; raises `DatasetNotRegisteredError`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise DatasetNotRegisteredError(
            f"no dataset registered under name {name!r}; registered datasets: {sorted(_REGISTRY)}"
        ) from exc


def get_loader(name: str) -> "LoaderCallable":
    """Look up a loader by name; raises `DatasetNotRegisteredError`."""
    try:
        return _LOADERS[name]
    except KeyError as exc:
        raise DatasetNotRegisteredError(
            f"no loader registered for dataset {name!r}; registered datasets: {sorted(_LOADERS)}"
        ) from exc


def list_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
