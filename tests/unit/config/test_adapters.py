"""Tests for the BaseEstimator adapter family at architecture A4 step 3.

The single-adapter contract (``TabularConfigParams``) generalizes to
six adapters; the keyword-only-init meta-test pins the ``*`` marker
across all of them so a BETA promotion cannot shift positional argument
order.
"""

import inspect

import pytest
import sklearn.base
from pydantic import ValidationError

from seq_sklearn.config.adapters import (
    LossParams,
    OptimizerParams,
    SamplerParams,
    SchedulerParams,
    TabularConfigParams,
    TFTAdvancedParams,
)
from seq_sklearn.config.loss import LossConfig
from seq_sklearn.config.optimizer import OptimizerConfig
from seq_sklearn.config.sampler import SamplerConfig
from seq_sklearn.config.scheduler import SchedulerConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTAdvancedConfig

_ALL_ADAPTERS = [
    TabularConfigParams,
    OptimizerParams,
    SchedulerParams,
    LossParams,
    SamplerParams,
    TFTAdvancedParams,
]


# ---- TabularConfigParams construction, get/set_params, clone, to_pydantic


def test_tabular_config_params_default_construction_mirrors_pydantic_defaults() -> None:
    adapter = TabularConfigParams()
    assert adapter.id_col == "id"
    assert adapter.time_col == "time"
    assert adapter.lookback == 12
    # F3 / mandatory test #2: contemporaneous default at the adapter
    # layer too (dual-layer default-is-0 guard); and it must survive
    # the to_pydantic() round-trip.
    assert adapter.prediction_step == 0
    assert adapter.to_pydantic().prediction_step == 0
    assert adapter.scaling_real == "standard"


def test_tabular_config_params_get_params_returns_flat_dict() -> None:
    adapter = TabularConfigParams(lookback=6)
    params = adapter.get_params(deep=False)
    assert params["lookback"] == 6
    assert params["id_col"] == "id"


def test_tabular_config_params_set_params_mutates_in_place() -> None:
    adapter = TabularConfigParams()
    adapter.set_params(lookback=24, prediction_step=2)
    assert adapter.lookback == 24
    assert adapter.prediction_step == 2


def test_tabular_config_params_clone_is_independent() -> None:
    original = TabularConfigParams(lookback=6)
    cloned = sklearn.base.clone(original)
    assert isinstance(cloned, TabularConfigParams)
    assert cloned is not original
    cloned.set_params(lookback=24)
    assert original.lookback == 6
    assert cloned.lookback == 24


def test_to_pydantic_produces_frozen_config() -> None:
    adapter = TabularConfigParams(lookback=6, prediction_step=2)
    cfg = adapter.to_pydantic()
    assert isinstance(cfg, TabularToSequenceConfig)
    assert cfg.lookback == 6
    assert cfg.prediction_step == 2


def test_to_pydantic_propagates_validation_errors() -> None:
    """A4 step 4: outer estimators wrap this into ConfigError at _build_config."""
    adapter = TabularConfigParams(lookback=0)  # below the ge=1 floor
    with pytest.raises(ValidationError):
        adapter.to_pydantic()


def test_tabular_config_params_embed_dims_none_default_is_empty_mapping() -> None:
    adapter = TabularConfigParams()
    cfg = adapter.to_pydantic()
    assert dict(cfg.categorical_embed_dims) == {}


def test_tabular_config_params_embed_dims_dict_round_trips() -> None:
    adapter = TabularConfigParams(categorical_embed_dims={"industry": 16})
    cfg = adapter.to_pydantic()
    assert dict(cfg.categorical_embed_dims) == {"industry": 16}


def test_tabular_config_params_set_params_chains_via_clone() -> None:
    original = TabularConfigParams()
    cloned = sklearn.base.clone(original)
    cloned.set_params(lookback=18)
    assert cloned.lookback == 18


# ---- Clone independence: each adapter produces an independent copy


def test_optimizer_params_clone_is_independent() -> None:
    original = OptimizerParams(learning_rate=0.05)
    cloned = sklearn.base.clone(original)
    assert cloned is not original
    cloned.set_params(learning_rate=0.001)
    assert original.learning_rate == 0.05
    assert cloned.learning_rate == 0.001


def test_scheduler_params_clone_is_independent() -> None:
    original = SchedulerParams(warmup_steps=200)
    cloned = sklearn.base.clone(original)
    assert cloned is not original
    cloned.set_params(warmup_steps=50)
    assert original.warmup_steps == 200
    assert cloned.warmup_steps == 50


def test_loss_params_clone_is_independent() -> None:
    original = LossParams(strategy="focal", focal_gamma=3.0)
    cloned = sklearn.base.clone(original)
    assert cloned is not original
    cloned.set_params(focal_gamma=1.0)
    assert original.focal_gamma == 3.0
    assert cloned.focal_gamma == 1.0


def test_sampler_params_clone_is_independent() -> None:
    original = SamplerParams(strategy="oversample_minority", oversample_ratio=2.0)
    cloned = sklearn.base.clone(original)
    assert cloned is not original
    cloned.set_params(oversample_ratio=1.5)
    assert original.oversample_ratio == 2.0
    assert cloned.oversample_ratio == 1.5


def test_tft_advanced_params_clone_is_independent() -> None:
    original = TFTAdvancedParams(extra={"knob": 1})
    cloned = sklearn.base.clone(original)
    assert cloned is not original
    cloned.set_params(extra={"knob": 2})
    assert original.extra == {"knob": 1}
    assert cloned.extra == {"knob": 2}


# ---- Keyword-only init: no parameter can be bound positionally


@pytest.mark.parametrize("adapter_cls", _ALL_ADAPTERS)
def test_all_adapters_have_keyword_only_init(adapter_cls: type) -> None:
    """Every non-self __init__ parameter must be KEYWORD_ONLY.

    Not merely POSITIONAL_OR_KEYWORD with a default: a parameter with a
    default still accepts positional binding, so a BETA promotion that
    inserts a typed field ahead of it would silently shift positional
    callers. The ``*`` marker forbids that.
    """
    params = list(inspect.signature(adapter_cls.__init__).parameters.values())
    non_self = [p for p in params if p.name != "self"]
    assert non_self, f"{adapter_cls.__name__}.__init__ has no params besides self"
    for p in non_self:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{adapter_cls.__name__}.__init__ parameter {p.name!r} is "
            f"{p.kind}, expected KEYWORD_ONLY"
        )


# ---- to_pydantic produces the matching frozen pydantic config


def test_optimizer_params_to_pydantic_produces_correct_config() -> None:
    cfg = OptimizerParams(name="sgd", learning_rate=0.01).to_pydantic()
    assert isinstance(cfg, OptimizerConfig)
    assert cfg.name == "sgd"
    assert cfg.learning_rate == 0.01


def test_scheduler_params_to_pydantic_produces_correct_config() -> None:
    cfg = SchedulerParams(name="one_cycle", warmup_steps=42).to_pydantic()
    assert isinstance(cfg, SchedulerConfig)
    assert cfg.name == "one_cycle"
    assert cfg.warmup_steps == 42


def test_loss_params_to_pydantic_produces_correct_config() -> None:
    cfg = LossParams(strategy="huber", huber_delta=2.5).to_pydantic()
    assert isinstance(cfg, LossConfig)
    assert cfg.strategy == "huber"
    assert cfg.huber_delta == 2.5


def test_sampler_params_to_pydantic_produces_correct_config() -> None:
    cfg = SamplerParams(strategy="undersample_majority").to_pydantic()
    assert isinstance(cfg, SamplerConfig)
    assert cfg.strategy == "undersample_majority"


def test_tft_advanced_params_to_pydantic_produces_correct_config() -> None:
    cfg = TFTAdvancedParams(extra={"beta_knob": 7}).to_pydantic()
    assert isinstance(cfg, TFTAdvancedConfig)
    assert dict(cfg.extra) == {"beta_knob": 7}


# ---- Outer-estimator clone deep-clones every nested adapter (A4)


def test_outer_estimator_clone_does_not_alias_adapter_instances() -> None:
    """``sklearn.base.clone`` of the estimator yields fresh adapters.

    The A4-draft "clone each adapter in __init__" was dropped because it
    breaks ``sklearn.base.clone``; correctness now relies on clone's own
    deep-clone of nested estimators. This pins that: every one of the six
    adapter slots on a cloned estimator must be a distinct object from
    the original's (no shared mutable state) while comparing equal by
    params.
    """
    from tests._test_models._dummy_estimator import _DummySequenceClassifier

    original = _DummySequenceClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(time_varying_real_cols=("tr",), lookback=3),
        optimizer=OptimizerParams(name="sgd", learning_rate=0.01),
        scheduler=SchedulerParams(name="one_cycle", warmup_steps=5),
        loss=LossParams(strategy="huber", huber_delta=2.5),
        sampler=SamplerParams(strategy="undersample_majority"),
        advanced=TFTAdvancedParams(extra={"knob": 1}),
    )
    cloned = sklearn.base.clone(original)

    for slot in ("tabular_config", "optimizer", "scheduler", "loss", "sampler", "advanced"):
        orig_adapter = getattr(original, slot)
        clone_adapter = getattr(cloned, slot)
        assert clone_adapter is not orig_adapter, f"{slot} adapter aliased after clone"
        assert type(clone_adapter) is type(orig_adapter)
        assert clone_adapter.get_params(deep=True) == orig_adapter.get_params(deep=True)
