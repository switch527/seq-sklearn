"""Frozen-config to estimator-kwargs bridge (A16; not in ``__all__``).

A sampled :class:`BaseModelConfig` is immutable pydantic; an estimator
``__init__`` takes the mutable :class:`sklearn.base.BaseEstimator`
adapters (A4 step 3). The round-trip is non-obvious, so it lives in one
documented place rather than being open-coded in every ``objective``.
Importable directly (like ``check_combo``) but kept out of the
``tuning`` package's headline ``__all__``.

Per-model dispatch: each concrete config class registers its nested
sub-config to adapter map in ``_ADAPTER_MAP_BY_CONFIG``. v2 / v3 add
their own entry; the helper itself never changes.
"""

from sklearn.base import BaseEstimator

from seq_sklearn.config._adapters import (
    LossParams,
    OptimizerParams,
    SamplerParams,
    SchedulerParams,
    TabularConfigParams,
    TFTAdvancedParams,
)
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.errors import ConfigError

# Nested pydantic sub-config field name -> mutable adapter class.
_TFT_ADAPTER_MAP: dict[str, type[BaseEstimator]] = {
    "tabular_config": TabularConfigParams,
    "optimizer": OptimizerParams,
    "scheduler": SchedulerParams,
    "loss": LossParams,
    "sampler": SamplerParams,
    "advanced": TFTAdvancedParams,
}

# Registry keyed by concrete config class. v2 / v3 estimators register
# their per-model adapter map here when they ship.
_ADAPTER_MAP_BY_CONFIG: dict[type[BaseModelConfig], dict[str, type[BaseEstimator]]] = {
    TFTConfig: _TFT_ADAPTER_MAP,
}


def adapter_map_for(config_cls: type[BaseModelConfig]) -> dict[str, type[BaseEstimator]]:
    """Return the sub-config to adapter map for ``config_cls``."""
    return _ADAPTER_MAP_BY_CONFIG[config_cls]


def config_to_estimator_kwargs(config: BaseModelConfig) -> dict[str, object]:
    """Convert a frozen config dump into the kwargs an estimator accepts.

    Every nested sub-config named in the model's adapter map is popped
    from ``model_dump(mode="json")`` and wrapped in its matching mutable
    adapter; flat fields pass through unchanged. ``mode="json"`` is the
    pinned serialization mode (F7 save/load contract): it fixes the
    ``extra`` and ``categorical_embed_dims`` tuple shapes on the wire,
    and each adapter's pydantic-side BeforeValidator re-normalizes the
    iterable-of-pairs form on ``to_pydantic``.
    """
    raw = config.model_dump(mode="json")
    adapter_map = adapter_map_for(type(config))
    kwargs: dict[str, object] = {}
    for field_name, adapter_cls in adapter_map.items():
        if field_name not in raw:
            raise ConfigError(
                f"{type(config).__name__}'s adapter map names sub-config "
                f"{field_name!r}, but it is absent from the config dump; "
                "the adapter map and the config schema are out of sync."
            )
        sub_dict = raw.pop(field_name)
        kwargs[field_name] = adapter_cls(**sub_dict)
    return {**raw, **kwargs}
