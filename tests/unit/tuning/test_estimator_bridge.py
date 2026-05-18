"""config_to_estimator_kwargs A16 contract (Phase 8, no training).

The frozen-pydantic-dump to mutable-adapter round-trip is non-obvious,
so it is pinned directly: every adapter slot survives, and the
``extra`` / ``categorical_embed_dims`` tuple shapes survive the
``mode="json"`` serialization the F7 save/load contract pins.
"""

import pytest
from sklearn.base import BaseEstimator

from seq_sklearn.config._adapters import (
    LossParams,
    OptimizerParams,
    SamplerParams,
    SchedulerParams,
    TabularConfigParams,
    TFTAdvancedParams,
)
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.errors import ConfigError
from seq_sklearn.tuning._estimator_bridge import (
    _ADAPTER_MAP_BY_CONFIG,
    _TFT_ADAPTER_MAP,
    adapter_map_for,
    config_to_estimator_kwargs,
)

_EXPECTED_ADAPTERS = {
    "tabular_config": TabularConfigParams,
    "optimizer": OptimizerParams,
    "scheduler": SchedulerParams,
    "loss": LossParams,
    "sampler": SamplerParams,
    "advanced": TFTAdvancedParams,
}


def _config() -> TFTConfig:
    return TFTConfig(
        task_type="binary",
        loss=LossConfig(strategy="cross_entropy"),
        hidden_size=64,
        attention_heads=8,
        tabular_config=TabularToSequenceConfig(
            id_col="id",
            time_col="t",
            time_varying_real_cols=("x0",),
        ),
    )


def test_adapter_map_for_returns_six_slot_map() -> None:
    amap = adapter_map_for(TFTConfig)
    assert dict(amap.items()) == _EXPECTED_ADAPTERS
    # The registry hands back the same object each call (v2 / v3 register
    # their own map; the helper is a pure lookup, not a rebuild).
    assert adapter_map_for(TFTConfig) is amap


def test_adapter_map_for_unknown_config_raises() -> None:
    with pytest.raises(KeyError):
        adapter_map_for(OptimizerConfig)  # type: ignore[arg-type]


def test_config_to_estimator_kwargs_round_trips_all_adapters() -> None:
    cfg = _config()
    kwargs = config_to_estimator_kwargs(cfg)

    # Every nested slot is wrapped in its matching mutable adapter.
    for slot, adapter_cls in _EXPECTED_ADAPTERS.items():
        assert isinstance(kwargs[slot], adapter_cls)
        assert isinstance(kwargs[slot], BaseEstimator)

    # Flat fields pass through unchanged (not wrapped).
    assert kwargs["task_type"] == "binary"
    assert kwargs["hidden_size"] == 64
    assert kwargs["attention_heads"] == 8

    # Each adapter rebuilds its exact frozen sub-config.
    assert kwargs["tabular_config"].to_pydantic() == cfg.tabular_config  # type: ignore[union-attr]
    assert kwargs["optimizer"].to_pydantic() == cfg.optimizer  # type: ignore[union-attr]
    assert kwargs["scheduler"].to_pydantic() == cfg.scheduler  # type: ignore[union-attr]
    assert kwargs["loss"].to_pydantic() == cfg.loss  # type: ignore[union-attr]
    assert kwargs["sampler"].to_pydantic() == cfg.sampler  # type: ignore[union-attr]
    assert kwargs["advanced"].to_pydantic() == cfg.advanced  # type: ignore[union-attr]


def test_config_to_estimator_kwargs_extra_tuple_type_survives() -> None:
    cfg = TFTConfig(
        task_type="binary",
        loss=LossConfig(strategy="cross_entropy"),
        optimizer=OptimizerConfig(extra=(("warmup_pct", 0.1),)),
        tabular_config=TabularToSequenceConfig(
            id_col="id",
            time_col="t",
            time_varying_real_cols=("x0",),
            categorical_embed_dims=(("region", 4),),
        ),
    )
    # Pre-condition: pydantic normalized both to sorted hashable tuples.
    assert cfg.optimizer.extra == (("warmup_pct", 0.1),)
    assert cfg.tabular_config.categorical_embed_dims == (("region", 4),)

    kwargs = config_to_estimator_kwargs(cfg)
    rebuilt_opt = kwargs["optimizer"].to_pydantic()  # type: ignore[union-attr]
    rebuilt_tab = kwargs["tabular_config"].to_pydantic()  # type: ignore[union-attr]

    # The mode="json" list-of-pairs survives back to the frozen tuple.
    assert rebuilt_opt.extra == (("warmup_pct", 0.1),)
    assert isinstance(rebuilt_opt.extra, tuple)
    assert rebuilt_tab.categorical_embed_dims == (("region", 4),)
    assert isinstance(rebuilt_tab.categorical_embed_dims, tuple)
    assert rebuilt_opt == cfg.optimizer
    assert rebuilt_tab == cfg.tabular_config


def test_config_to_estimator_kwargs_raises_on_adapter_map_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the adapter map names a sub-config the schema dump lacks
    # (adapter map vs schema drift), the bridge must raise a typed
    # ConfigError, not a bare KeyError.
    drifted = {**_TFT_ADAPTER_MAP, "no_such_subconfig": OptimizerParams}
    monkeypatch.setitem(_ADAPTER_MAP_BY_CONFIG, TFTConfig, drifted)
    with pytest.raises(ConfigError, match="out of sync"):
        config_to_estimator_kwargs(_config())
