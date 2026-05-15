"""Tier 4 ``extra`` escape hatch and deprecation-alias machinery (per A4).

``ExtraDict`` is the ALPHA-tier landing zone carried by every family
sub-config. It is restricted to a JSON-safe value union and stored as a
sorted hashable tuple so the frozen pydantic models stay hashable (same
precedent as ``categorical_embed_dims`` at
:mod:`seq_sklearn.config.tabular`).

:func:`extract_deprecated_extras` is the single helper every family
factory routes through so the ALPHA -> BETA promotion alias contract
lands once instead of being re-implemented per factory.
"""

import warnings
from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, BeforeValidator

from seq_sklearn.errors import ConfigError

__all__ = [
    "ExtraDict",
    "ExtraValue",
    "extract_deprecated_extras",
]

ExtraValue = str | int | float | bool | None


def _normalize_extras(v: object) -> tuple[tuple[str, ExtraValue], ...]:
    """Coerce dict / Mapping / iterable-of-pairs input to a sorted hashable tuple.

    Mirrors the pattern at
    :func:`seq_sklearn.config.tabular._normalize_embed_dims`.

    Raises:
        TypeError: for keys that are not ``str`` or values outside the
            documented ``ExtraValue`` union, so unserializable types
            fail at construction, not at save time.
    """
    if v in (None, (), {}):
        return ()
    items = list(v.items()) if isinstance(v, Mapping) else list(v)  # type: ignore[attr-defined]
    for k, val in items:
        if not isinstance(k, str):
            raise TypeError(f"extra key must be str, got {type(k).__name__}")
        # `bool` is a subclass of `int`, so the four-type tuple covers all
        # five documented ExtraValue types without redundancy.
        if not isinstance(val, (str, int, float, type(None))):
            raise TypeError(
                f"extra value for key {k!r} must be one of "
                f"str/int/float/bool/None; got {type(val).__name__}. "
                "Nested structures and custom objects are not supported "
                "in the `extra` escape hatch."
            )
    return tuple(sorted(items))


ExtraDict = Annotated[
    tuple[tuple[str, ExtraValue], ...],
    BeforeValidator(_normalize_extras),
]


_PROMOTED_KEYS_BY_FAMILY: dict[str, dict[str, str]] = {
    "optimizer": {
        # Populated as ALPHA keys are promoted to typed fields.
        # Format: "<extra-key>": "<typed-field-name>".
        # Example after a future promotion:
        #     "amsgrad": "amsgrad",
    },
    "scheduler": {},
    "loss": {},
    "sampler": {},
}


def extract_deprecated_extras(
    cfg: BaseModel,
    family: str,
) -> dict[str, ExtraValue]:
    """Return ``extra`` as a dict, routing promoted keys to the typed field.

    Every family factory (``build_optimizer``, ``build_scheduler``,
    ``build_loss``, ``build_sampler``) calls this helper instead of
    ``dict(cfg.extra)`` so the deprecation-alias contract lands once. A
    maintainer promoting an ALPHA key to a typed field adds one entry to
    ``_PROMOTED_KEYS_BY_FAMILY`` and the alias behavior fires
    automatically.

    Known limitation, and a hard pre-condition for registering the
    first promotion: when a promoted key is supplied via ``extra`` and
    the typed field is still at its default, this helper drops the key
    from the returned dict but cannot write the supplied value onto the
    frozen ``cfg``. The value is therefore discarded rather than routed
    to the typed field. ``_PROMOTED_KEYS_BY_FAMILY`` is empty, so this
    branch never executes and the loop body is unreachable today. Before
    any entry is added, this must be fixed (return a
    ``cfg.model_copy(update=...)`` alongside the cleaned dict, or have
    the family factory read the renamed key from the returned dict).
    ``_PROMOTED_KEYS_BY_FAMILY`` must stay empty until then.

    Raises:
        ConfigError: if a promoted key is supplied via both ``extra``
            and its typed field (ambiguous configuration).
    """
    extra = dict(cfg.extra)  # type: ignore[attr-defined]
    promoted = _PROMOTED_KEYS_BY_FAMILY[family]
    for extra_key, typed_name in promoted.items():
        if extra_key in extra:
            existing_typed = getattr(cfg, typed_name)
            warnings.warn(
                f"Passing {extra_key!r} via {type(cfg).__name__}.extra is "
                f"deprecated; use the typed {type(cfg).__name__}.{typed_name} "
                f"field. The dict path remains a permanent alias.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Promoted fields must have an explicit default (the registry
            # meta-test enforces this). If the typed value differs from
            # its default, the caller set BOTH the typed field AND the
            # extra key, which is ambiguous and rejected. Otherwise the
            # key is removed from the returned dict (see the pre-condition
            # in the docstring: the value is currently discarded, not
            # routed onto the frozen cfg).
            typed_default = type(cfg).model_fields[typed_name].default
            if existing_typed != typed_default:
                raise ConfigError(
                    f"{extra_key!r} provided via both extra and the typed "
                    f"{typed_name} field; remove one."
                )
            extra.pop(extra_key)
    return extra
