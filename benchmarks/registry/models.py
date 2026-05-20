"""Model registry: name -> `ModelSpec`.

Phase B2 will populate this via `register_model(spec)` calls inside
each adapter module (`benchmarks/adapters/seq_sklearn.py`,
`adapters/gbm.py`, `adapters/tsc.py`, ...). The registry is
import-time-populated when those modules are imported.
"""

from benchmarks.config import ModelSpec

_REGISTRY: dict[str, ModelSpec] = {}


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
            f"model {spec.name!r} already registered with a different "
            f"spec; refusing to overwrite"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_model(name: str) -> ModelSpec:
    """Look up a model by name; raises `ModelNotRegisteredError`."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ModelNotRegisteredError(
            f"no model registered under name {name!r}; "
            f"registered models: {sorted(_REGISTRY)}"
        ) from exc


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
