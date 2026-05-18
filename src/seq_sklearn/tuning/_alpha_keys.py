"""Curated per-family ALPHA-key search list (A16 ``search_extras``).

``suggest_params(..., search_extras=True)`` samples from the family's
curated ALPHA-key list (escape-hatch ``extra`` keys promoted to ALPHA
stability for tuning). v1 ships this empty by design: every v1 knob is
already a typed STABLE field, so there is no ALPHA key to sweep and the
flag is a structural no-op until a future family registers keys here.
"""

from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig

# Keyed by concrete config class, mirroring _ADAPTER_MAP_BY_CONFIG. Each
# value is the tuple of `extra`-dict keys eligible for the search_extras
# sweep. Empty in v1: no ALPHA keys exist yet.
_ALPHA_KEYS_BY_CONFIG: dict[type[BaseModelConfig], tuple[str, ...]] = {
    TFTConfig: (),
}


def alpha_keys_for(config_cls: type[BaseModelConfig]) -> tuple[str, ...]:
    """Return the curated ALPHA-key list for ``config_cls`` (``()`` if none)."""
    return _ALPHA_KEYS_BY_CONFIG.get(config_cls, ())
