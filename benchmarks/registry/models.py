"""Model registry: name -> (`ModelSpec`, adapter factory).

Phase B2 populated the `ModelSpec` half (registry of metadata).
Phase B5 adds the parallel adapter-factory map so the experiment
driver can instantiate an adapter from its registered name. Each
adapter module (`benchmarks/adapters/seq_sklearn.py`,
`adapters/gbm.py`, ...) calls both `register_model(spec)` and
`register_adapter_factory(name, factory)` at module-import time;
the two halves stay in lockstep because the no-deep-imports test
in `tests/benchmarks/test_scaffold.py` imports the adapters package
which triggers both registrations.

The factory signature is intentionally narrow per B3.2.1: the
harness calls `factory(spec=dataset_spec, hyperparameters=...)` and
treats the result as a `SeqSklearnAdapter`. Concrete adapter
classes either match this signature directly (the dataclass kwargs
shape suffices) or wrap themselves in a small `_make_*` adapter
function declared next to the class.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from benchmarks.config import DatasetSpec, ModelSpec

if TYPE_CHECKING:
    # Type-only import: importing `benchmarks.adapters._base` at module
    # top would pull in `benchmarks.adapters.__init__` (which itself
    # imports each adapter module, which imports `register_adapter_
    # factory` from this file), creating a partial-init circular
    # import. The factory's runtime contract is duck-typed; pyright
    # gets the Protocol annotation through the TYPE_CHECKING guard.
    from benchmarks.adapters._base import SeqSklearnAdapter

# The factory protocol: a callable that returns an adapter ready to
# run protocol introspection (`name`, `family`, `task_types`,
# `supports_proba`) before `fit` is called. The dataset spec is
# REQUIRED because adapters carry per-dataset state (`tabular_config`
# is built from the spec at the B3.2.2 boundary). Hyperparameters
# are optional; the HPO driver (B8) feeds suggested params here.
AdapterFactory = Callable[..., "SeqSklearnAdapter"]


_REGISTRY: dict[str, ModelSpec] = {}
_ADAPTER_FACTORIES: dict[str, AdapterFactory] = {}


class ModelNotRegisteredError(KeyError):
    """Raised when a config asks for a model name that no adapter
    module has registered. Distinct from `KeyError` for typed
    catching."""


def register_model(spec: ModelSpec) -> ModelSpec:
    """Register a model spec by name. Idempotent only if the spec is
    identical; a different spec under the same name raises."""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"model {spec.name!r} already registered with a different spec; refusing to overwrite"
        )
    _REGISTRY[spec.name] = spec
    return spec


def register_adapter_factory(name: str, factory: AdapterFactory) -> AdapterFactory:
    """Register a factory callable for the model named `name`.

    Idempotent only if the factory object is identical; a different
    factory under the same name raises so a silent overwrite during
    module-import-time registration cannot happen.

    Raises:
        ModelNotRegisteredError: no `ModelSpec` is registered under
            this name. The factory and the spec must be co-located
            in the same adapter module; this guard catches the case
            where one registration is forgotten.
    """
    if name not in _REGISTRY:
        raise ModelNotRegisteredError(
            f"register_adapter_factory: no ModelSpec registered under "
            f"{name!r}; register the spec first (the spec carries the "
            f"task_types / supports_proba metadata the harness reads "
            f"before calling the factory)"
        )
    existing = _ADAPTER_FACTORIES.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(
            f"adapter factory for {name!r} already registered with a "
            f"different callable; refusing to overwrite"
        )
    _ADAPTER_FACTORIES[name] = factory
    return factory


def get_model(name: str) -> ModelSpec:
    """Look up a model by name; raises `ModelNotRegisteredError`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ModelNotRegisteredError(
            f"no model registered under name {name!r}; registered models: {sorted(_REGISTRY)}"
        ) from exc


def get_adapter_factory(name: str) -> AdapterFactory:
    """Look up the adapter factory for `name`.

    Raises:
        ModelNotRegisteredError: no factory is registered. Either
            the model name is wrong or the adapter module hasn't been
            imported (registrations are import-time side effects).
    """
    try:
        return _ADAPTER_FACTORIES[name]
    except KeyError as exc:
        raise ModelNotRegisteredError(
            f"no adapter factory registered under name {name!r}; "
            f"registered factories: {sorted(_ADAPTER_FACTORIES)}"
        ) from exc


def instantiate_adapter(
    name: str,
    *,
    spec: DatasetSpec,
    hyperparameters: dict[str, Any] | None = None,
) -> "SeqSklearnAdapter":
    """Build an adapter by registered name.

    Thin convenience over `get_adapter_factory(name)(spec=..., ...)`
    so the experiment driver can write
    `instantiate_adapter("tft_classifier", spec=ds_spec)` without
    threading the factory variable.
    """
    return get_adapter_factory(name)(spec=spec, hyperparameters=hyperparameters or {})


def register_adapter(spec: ModelSpec, factory: AdapterFactory) -> ModelSpec:
    """Register a model spec + adapter factory atomically.

    `register_model` and `register_adapter_factory` can be called
    separately, but a partial registration (spec without factory)
    leaves `_REGISTRY` and `_ADAPTER_FACTORIES` out of sync. This
    helper wraps the pair: if `register_adapter_factory` raises, the
    spec registration is rolled back so re-attempting the registration
    starts from a clean state.

    Adapter modules should prefer this helper over the two-call form
    so a future I/O step between the two registrations cannot leave
    the registry inconsistent.
    """
    is_new_spec = spec.name not in _REGISTRY
    register_model(spec)
    try:
        register_adapter_factory(spec.name, factory)
    except Exception:
        if is_new_spec:
            _REGISTRY.pop(spec.name, None)
        raise
    return spec


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
