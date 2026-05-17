"""TFTRegressor A4-adapter __init__ contract (Phase 7, no training).

Mirrors test_classifier_init.py but for TFTRegressor: the MRO diverges
after the shared `_TFTEstimatorMixin`
(`TFTRegressor -> _TFTEstimatorMixin -> TransformerSequenceEstimator
.Regressor -> BaseSequenceRegressor -> RegressorMixin ->
BaseSequenceEstimator`), so the classifier's clone / get_params /
keyword-only / verbatim-None-loss assertions do NOT transfer and need a
regressor-rooted pin. Adds the quantile-mode clone round-trip
(`quantiles` + `task_type` survive `sklearn.base.clone`).
"""

import inspect

import pytest
import sklearn.base

from seq_sklearn.config._adapters import OptimizerParams, TabularConfigParams
from seq_sklearn.config.base import BaseModelConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor

_MODEL_SHAPE_FIELDS = {
    "hidden_size",
    "attention_heads",
    "dropout",
    "variable_selection_dropout",
    "prediction_readout",
}


def _reg(**kw: object) -> TFTRegressor:
    params: dict[str, object] = {"task_type": "regression_point"}
    params.update(kw)
    return TFTRegressor(**params)  # type: ignore[arg-type]


def test_init_mirrors_every_tft_model_shape_field() -> None:
    init_params = set(inspect.signature(TFTRegressor.__init__).parameters) - {"self"}
    assert init_params >= _MODEL_SHAPE_FIELDS
    for name, p in inspect.signature(TFTRegressor.__init__).parameters.items():
        if name in _MODEL_SHAPE_FIELDS:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
    tft_only = set(TFTConfig.model_fields) - set(BaseModelConfig.model_fields)
    assert tft_only - {"tabular_config", "advanced"} == _MODEL_SHAPE_FIELDS


def test_model_shape_params_stored_verbatim() -> None:
    est = _reg(hidden_size=64, attention_heads=8, prediction_readout="mean_pool")
    assert est.hidden_size == 64
    assert est.attention_heads == 8
    assert est.prediction_readout == "mean_pool"
    d = _reg()
    assert d.hidden_size == 128
    assert d.attention_heads == 4
    assert d.dropout == 0.1
    assert d.variable_selection_dropout == 0.1
    assert d.prediction_readout == "last_valid"


def test_get_params_deep_exposes_adapter_flat_keys() -> None:
    est = _reg(tabular_config=TabularConfigParams(lookback=6))
    params = est.get_params(deep=True)
    assert params["tabular_config__lookback"] == 6
    assert "optimizer__learning_rate" in params
    assert "scheduler__warmup_steps" in params
    assert "advanced__extra" in params
    assert params["hidden_size"] == 128
    assert params["loss"] is None  # F5 verbatim-None loss adapter (regressor MRO)


def test_set_params_chains_through_adapter_and_clone_independent() -> None:
    opt = OptimizerParams(learning_rate=5e-4)
    est = _reg(optimizer=opt, hidden_size=32)
    est.set_params(optimizer__learning_rate=3e-4, tabular_config__lookback=9)
    assert est.optimizer.learning_rate == 3e-4
    assert est.tabular_config.lookback == 9
    clone = sklearn.base.clone(est)
    assert clone.optimizer is not est.optimizer
    assert clone.hidden_size == 32
    clone.optimizer.learning_rate = 1.0
    assert est.optimizer.learning_rate == 3e-4


def test_quantile_mode_clone_round_trips_quantiles_and_task_type() -> None:
    # The regressor-specific surface: quantile mode params must survive
    # a sklearn clone through the regressor MRO.
    est = _reg(task_type="regression_quantile", quantiles=(0.1, 0.5, 0.9))
    clone = sklearn.base.clone(est)
    assert clone.task_type == "regression_quantile"
    assert clone.quantiles == (0.1, 0.5, 0.9)
    assert clone.get_params(deep=True)["quantiles"] == (0.1, 0.5, 0.9)


@pytest.mark.parametrize("bad_kw", ["bogus", "n_heads", "hidden_dim"])
def test_unknown_kwarg_raises(bad_kw: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TFTRegressor(task_type="regression_point", **{bad_kw: 1})  # type: ignore[arg-type]


def test_positional_args_rejected() -> None:
    with pytest.raises(TypeError):
        TFTRegressor("regression_point")  # type: ignore[misc]


def test_build_config_produces_tftconfig_with_overrides() -> None:
    est = _reg(hidden_size=64, attention_heads=8)
    cfg = est._build_config()
    assert isinstance(cfg, TFTConfig)
    assert cfg.hidden_size == 64
    assert cfg.attention_heads == 8
    assert cfg.tabular_config is not None
    assert est._config_cls is TFTConfig


def test_build_config_invalid_heads_wraps_configerror() -> None:
    from seq_sklearn.errors import ConfigError

    est = _reg(hidden_size=10, attention_heads=3)
    with pytest.raises(ConfigError):
        est._build_config()
