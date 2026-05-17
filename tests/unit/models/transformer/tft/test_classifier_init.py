"""TFTClassifier A4-adapter __init__ contract (Phase 7, no training).

Pins that ``__init__`` mirrors every TFTConfig model-shape field, that
the six adapters surface as the canonical double-underscore flat keys
in ``get_params(deep=True)``, that unknown kwargs raise (keyword-only),
that ``clone`` is independent, and that ``_build_config`` produces a
``TFTConfig`` carrying the model-shape overrides.
"""

import inspect

import pytest
import sklearn.base

from seq_sklearn.config._adapters import OptimizerParams, TabularConfigParams
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier

# The TFTConfig fields that are NOT part of the shared BaseModelConfig
# surface and are NOT nested adapters: the flat model-shape knobs the
# concrete __init__ must mirror 1:1.
_MODEL_SHAPE_FIELDS = {
    "hidden_size",
    "attention_heads",
    "dropout",
    "variable_selection_dropout",
    "prediction_readout",
}


def _clf(**kw: object) -> TFTClassifier:
    params: dict[str, object] = {"task_type": "binary"}
    params.update(kw)
    return TFTClassifier(**params)  # type: ignore[arg-type]


def test_init_mirrors_every_tft_model_shape_field() -> None:
    init_params = set(inspect.signature(TFTClassifier.__init__).parameters) - {"self"}
    assert init_params >= _MODEL_SHAPE_FIELDS
    # every model-shape param is keyword-only (A4: BETA promotion safe)
    for name, p in inspect.signature(TFTClassifier.__init__).parameters.items():
        if name in _MODEL_SHAPE_FIELDS:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
    # and they are exactly the non-BaseModelConfig, non-adapter TFTConfig
    # fields (so a future TFTConfig model-shape field can't be silently
    # unexposed).
    tft_only = set(TFTConfig.model_fields) - set(BaseModelConfig.model_fields)
    assert tft_only - {"tabular_config", "advanced"} == _MODEL_SHAPE_FIELDS


def test_model_shape_params_stored_verbatim() -> None:
    est = _clf(hidden_size=64, attention_heads=8, prediction_readout="mean_pool")
    assert est.hidden_size == 64
    assert est.attention_heads == 8
    assert est.prediction_readout == "mean_pool"
    # defaults mirror TFTConfig defaults
    d = _clf()
    assert d.hidden_size == 128
    assert d.attention_heads == 4
    assert d.dropout == 0.1
    assert d.variable_selection_dropout == 0.1
    assert d.prediction_readout == "last_valid"


def test_get_params_deep_exposes_adapter_flat_keys() -> None:
    est = _clf(tabular_config=TabularConfigParams(lookback=6))
    params = est.get_params(deep=True)
    assert params["tabular_config__lookback"] == 6
    assert "optimizer__learning_rate" in params
    assert "scheduler__warmup_steps" in params
    assert "advanced__extra" in params
    assert params["hidden_size"] == 128  # flat model-shape key present
    assert params["loss"] is None  # F5 verbatim-None loss adapter


def test_set_params_chains_through_adapter_and_clone_independent() -> None:
    opt = OptimizerParams(learning_rate=5e-4)
    est = _clf(optimizer=opt, hidden_size=32)
    est.set_params(optimizer__learning_rate=3e-4, tabular_config__lookback=9)
    assert est.optimizer.learning_rate == 3e-4
    assert est.tabular_config.lookback == 9
    clone = sklearn.base.clone(est)
    assert clone.optimizer is not est.optimizer
    assert clone.hidden_size == 32
    clone.optimizer.learning_rate = 1.0
    assert est.optimizer.learning_rate == 3e-4


@pytest.mark.parametrize("bad_kw", ["bogus", "n_heads", "hidden_dim"])
def test_unknown_kwarg_raises(bad_kw: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TFTClassifier(task_type="binary", **{bad_kw: 1})  # type: ignore[arg-type]


def test_positional_args_rejected() -> None:
    # keyword-only __init__: nothing binds positionally
    with pytest.raises(TypeError):
        TFTClassifier("binary")  # type: ignore[misc]


def test_build_config_produces_tftconfig_with_overrides() -> None:
    est = _clf(hidden_size=64, attention_heads=8)
    cfg = est._build_config()
    assert isinstance(cfg, TFTConfig)
    assert cfg.hidden_size == 64
    assert cfg.attention_heads == 8
    assert cfg.tabular_config is not None  # required field re-supplied
    assert est._config_cls is TFTConfig


def test_build_config_invalid_heads_wraps_configerror() -> None:
    from seq_sklearn.errors import ConfigError

    # attention_heads must divide hidden_size (TFTConfig validator);
    # _build_config wraps the pydantic ValidationError as ConfigError.
    est = _clf(hidden_size=10, attention_heads=3)
    with pytest.raises(ConfigError):
        est._build_config()
